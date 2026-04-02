
import asyncio
import threading
from weakref import WeakSet

class TaskContext:
    _register_lock = threading.Lock()
    register = WeakSet()
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()
        self._loop = None

    def set_loop(self, loop):
        self._loop = loop

    def spawn(self, coro, name=None, loop=None):
        loop = loop if loop is not None else self._loop
        if loop is None:
            task = asyncio.create_task(coro, name=name)
        else:
            task = loop.create_task(coro, name=name)

        self._tasks.add(task)
        with TaskContext._register_lock:
            TaskContext.register.add(task)
        task.add_done_callback(self.discard)
        return task

    def add(self, task, name=None):
        self._tasks.add(task)
        with TaskContext._register_lock:
            TaskContext.register.add(task)
        task.add_done_callback(self.discard)
        return task

    def discard(self, task):
        self._tasks.discard(task)
        with TaskContext._register_lock:
            TaskContext.register.discard(task)

    async def close(self, force_timeout=10):
        if not self._tasks: return
        for t in list(self._tasks): t.cancel()
        try: await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), force_timeout)
        except asyncio.TimeoutError: pass
        except asyncio.CancelledError: pass
        with TaskContext._register_lock:
            TaskContext.register = TaskContext.register.difference(self._tasks)
        self._tasks.clear()

    @classmethod
    async def shutdown(cls, force_timeout=10):
        with cls._register_lock:
            if not TaskContext.register: return
            tasks = list(TaskContext.register)
        for t in tasks: t.cancel()
        try: await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), force_timeout)
        except asyncio.TimeoutError: pass
        except asyncio.CancelledError: pass
        with cls._register_lock:
            TaskContext.register.clear()


