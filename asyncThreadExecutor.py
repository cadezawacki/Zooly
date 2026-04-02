


from __future__ import annotations

import asyncio
import threading
from weakref import WeakValueDictionary, WeakKeyDictionary
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Optional

from app.helpers.taskContext import TaskContext
from app.logs.logging import log
from app.helpers.loop_helpers import set_uvloop

def iswrappedcoroutine(x):
    return hasattr(x, '__call__') and asyncio.iscoroutinefunction(x.__call__)

class AsyncThreadExecutor:
    def __init__(self, *, loop=None, name: str = "async-executor", max_workers: Optional[int] = None):
        self._name = name
        self._loop: Optional[asyncio.AbstractEventLoop] = loop
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._shutdown = False
        self._start_exc: Optional[BaseException] = None
        self._loop_ref = WeakKeyDictionary()

        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{name}-worker",
        )

        self.ctx = TaskContext()

        # Track inner asyncio.Tasks spawned on the executor loop, keyed by
        # the concurrent.futures.Future returned to the caller.  This allows
        # true cancellation of the work running on the executor thread.
        self._inner_tasks: dict[asyncio.futures.Future, asyncio.Task] = {}
        self._inner_lock = threading.Lock()

    def _track_inner(self, cf_future, inner_task: asyncio.Task, loop:asyncio.AbstractEventLoop):
        with self._inner_lock:
            self._inner_tasks[cf_future] = inner_task
            self._loop_ref[cf_future] = loop

    def _untrack_inner(self, cf_future):
        with self._inner_lock:
            self._inner_tasks.pop(cf_future, None)
            self._loop_ref.pop(cf_future, None)

    def cancel_inner(self, cf_future):
        """Cancel the inner asyncio.Task running on the executor loop for the
        given concurrent.futures.Future.  This is the only reliable way to
        stop work that is already executing on the executor thread."""
        with self._inner_lock:
            inner = self._inner_tasks.pop(cf_future, None)
            loop = self._loop_ref.pop(cf_future, None)
        if inner is not None and not inner.done():
            loop = loop if loop is not None else self._loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(inner.cancel)

    def _thread_main(self):
        try:
            if self._loop is None:
                self._loop = set_uvloop()

            asyncio.set_event_loop(self._loop)
            self.ctx.set_loop(self._loop)

            # Make the thread pool the loop's default executor so run_in_executor(None, ...) uses it.
            try:
                self._loop.set_default_executor(self._pool)
            except NotImplementedError:
                # Some custom loops may not support set_default_executor; fall back to explicit executor usage.
                pass

        except BaseException as exc:
            self._start_exc = exc
            self._ready.set()
            return

        self._ready.set()

        try:
            self._loop.run_forever()
        finally:
            try:
                # Drain async generators cleanly.
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            finally:
                # Ensure the default executor is shut down (this will call executor.shutdown()).
                shutdown_default = getattr(self._loop, "shutdown_default_executor", None)
                if shutdown_default is not None:
                    try:
                        self._loop.run_until_complete(shutdown_default())
                    except BaseException:
                        # Best-effort; loop is closing anyway.
                        pass

                asyncio.set_event_loop(None)
                try:
                    self._loop.close()
                finally:
                    # If the loop didn't own the pool (e.g. set_default_executor unsupported), shut it down here.
                    try:
                        self._pool.shutdown(wait=True, cancel_futures=False)
                    except TypeError:
                        self._pool.shutdown(wait=True)

    def start(self):
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Executor is shut down")

            if self._thread is not None and self._thread.is_alive():
                return

            self._ready.clear()
            self._start_exc = None

            self._thread = threading.Thread(
                target=self._thread_main,
                name=self._name,
                daemon=False,
            )
            self._thread.start()

        self._ready.wait()
        if self._start_exc is not None:
            raise RuntimeError("Failed to start executor thread") from self._start_exc

    # ---------- submission ----------

    def submit(self, func, /, *args, **kwargs):
        """
        Accepts:
          - coroutine function + args/kwargs  -> created/awaited on executor loop thread
          - sync callable + args/kwargs       -> executed on thread pool via loop.run_in_executor
          - coroutine object (legacy)         -> awaited on executor loop thread (no args/kwargs allowed)
        Returns a concurrent.futures.Future.

        Cancellation: call cancel_inner(future) to cancel the inner asyncio.Task
        on the executor loop.  Simply calling future.cancel() on the returned
        concurrent.futures.Future is NOT sufficient to stop already-running
        coroutines.
        """
        loop = kwargs.pop('__loop', self._loop)

        if self._shutdown:
            raise RuntimeError("Executor is shut down")

        if loop is None or self._thread is None or not self._thread.is_alive():
            self.start()

        loop = loop if loop is not None else self._loop

        if self._start_exc is not None:
            raise RuntimeError("Executor thread is not healthy") from self._start_exc

        executor_ref = self

        if asyncio.iscoroutine(func):
            if args or kwargs:
                raise TypeError("submit(coro) does not accept args/kwargs; pass a coroutine function instead")
            coro_obj = func

            async def _runner():
                inner_task = executor_ref.ctx.spawn(coro_obj, loop=loop)
                # Store inner task reference so cancel_inner() can reach it.
                # cf_future is captured from the enclosing scope after
                # run_coroutine_threadsafe returns.
                executor_ref._track_inner(cf_future, inner_task, loop)
                try:
                    return await inner_task
                except asyncio.CancelledError:
                    inner_task.cancel()
                    raise
                finally:
                    executor_ref._untrack_inner(cf_future)

            cf_future = asyncio.run_coroutine_threadsafe(_runner(), loop)
            return cf_future

        if asyncio.iscoroutinefunction(func) or iswrappedcoroutine(func):
            async def _runner():
                coro_obj = func(*args, **kwargs)  # create inside executor loop thread
                inner_task = executor_ref.ctx.spawn(coro_obj, loop=loop)
                executor_ref._track_inner(cf_future, inner_task, loop)
                try:
                    return await inner_task
                except asyncio.CancelledError:
                    loop.call_soon_threadsafe(inner_task.cancel)
                    raise
                finally:
                    executor_ref._untrack_inner(cf_future)

            cf_future = asyncio.run_coroutine_threadsafe(_runner(), loop)
            return cf_future

        if not callable(func):
            raise TypeError("submit() expects a callable, coroutine function, or coroutine object")

        call = partial(func, *args, **kwargs) if kwargs else partial(func, *args)
        async def _runner():
            # Use the loop's default executor (our pool) when possible.
            try:
                return await loop.run_in_executor(None, call)
            except NotImplementedError:
                return await loop.run_in_executor(self._pool, call)
        return asyncio.run_coroutine_threadsafe(_runner(), loop)

    # ---------- convenience ----------

    def run(self, func, /, *args, **kwargs):
        # Avoid deadlocking if called from an event loop thread.
        try:
            loop = kwargs.get('__loop', asyncio.get_running_loop())
        except RuntimeError:
            pass
            #log.warning("run() called from within a running event loop; use submit() + await wrap_future()")

        future = self.submit(func, *args, **kwargs)
        return future.result()

    # ---------- shutdown ----------

    def shutdown(self, cancel_pending: bool = True):
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

            loop = self._loop
            thread = self._thread

        if loop is None or thread is None:
            return

        if cancel_pending:
            self._cancel_all_tasks(loop)

        try:
            loop.call_soon_threadsafe(loop.stop)
        except BaseException:
            pass

        thread.join()

        with self._lock:
            self._thread = None
            self._loop = None
            with self._inner_lock:
                self._inner_tasks.clear()

    def _cancel_all_tasks(self, _loop):
        # Cancel all tracked inner tasks directly.
        with self._inner_lock:
            for fut, inner in self._inner_tasks.items():
                loop = self._loop_ref.pop(fut, _loop)
                if not inner.done():
                    try:
                        loop.call_soon_threadsafe(inner.cancel)
                    except RuntimeError:
                        pass
            self._inner_tasks.clear()
            self._loop_ref.clear()

        async def _close_ctx():
            await self.ctx.close()

        try:
            loop.call_soon_threadsafe(_close_ctx).result()
        except BaseException:
            # If loop is already stopping/closed, best-effort.
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


if __name__ == "__main__":
    async def work(x):
        return x * 2

    def blocking_work(x):
        return x + 100

    executor = AsyncThreadExecutor()

    # 1) Sync (runs in thread pool)
    future = executor.submit(blocking_work, 21)
    print("1) ", future.result())

    # 2) Async (runs on executor's loop thread)
    future = executor.submit(work, 21)
    print("2) ", future.result())

    # 3) Fire and forget
    executor.submit(work, 10)
    executor.submit(blocking_work, 20)
    executor.submit(work, 30)
    print("3) ", "done!")

    # 4) Async caller awaiting result
    async def caller():
        fut = executor.submit(work, 5)
        result = await asyncio.wrap_future(fut)
        print("4) ", result)

    asyncio.run(caller())
    executor.shutdown()

    # 5) Context-manager
    with AsyncThreadExecutor() as ex:
        r1 = ex.run(work, 1)          # runs on loop thread, blocks caller
        r2 = ex.run(blocking_work, 2) # runs in pool, blocks caller
        print("5) ", r1, r2)



