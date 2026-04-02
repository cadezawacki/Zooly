
from __future__ import annotations
import numpy as np
try:
    import re2 as re
except ImportError:
    import re

from app.helpers.pandas_helpers import pd
import polars as pl
import pytz
from zoneinfo import ZoneInfo
from datetime import datetime, date, timedelta, time
from datetime import time as datetimetime
from datetime import timezone
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
import holidays
from typing import Union, Optional, List, Any, Dict, Tuple, Set, TYPE_CHECKING
import warnings
from functools import wraps, lru_cache
import calendar
import bisect
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import time as time_module
import tzlocal

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

# Optional imports
try:
    from temporal_adjuster import TemporalAdjuster
    HAS_TEMPORAL_ADJUSTER = True
except ImportError:
    HAS_TEMPORAL_ADJUSTER = False

try:
    import whenever
    from whenever import Instant, PlainDateTime
    HAS_WHENEVER = True
except ImportError:
    HAS_WHENEVER = False

try:
    from friendlydateparser import parse_date as friendly_parse_date, parse_datetime as friendly_parse_datetime
    HAS_FRIENDLY_PARSER = True
except ImportError:
    from dateutil.parser import parse as friendly_parse_date
    friendly_parse_datetime = friendly_parse_date
    HAS_FRIENDLY_PARSER = False

try:
    import dateparser
    HAS_DATEPARSER = True
except ImportError:
    HAS_DATEPARSER = False

# Default settle days
DEFAULT_SETTLE_DAYS = 1

# Constants
DEFAULT_TZ = pytz.UTC
EXCEL_EPOCH = datetime(1899, 12, 30, tzinfo=DEFAULT_TZ)
KDB_EPOCH = datetime(2000, 1, 1, tzinfo=DEFAULT_TZ)

EXCEL_EPOCH_DATE = date(1899, 12, 30)
KDB_EPOCH_DATE = date(2000, 1, 1)

# Backwards compatibility - US Business Day
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
ENABLE_FINANCIAL_CALENDARS = True

# Credit market enums
class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"
    THIRTY_360_EUROPEAN = "30E/360"
    THIRTY_360_ISDA = "30/360 ISDA"
    ACT_365_FIXED = "ACT/365 Fixed"
    ACT_364 = "ACT/364"
    BUS_252 = "BUS/252"


class SettlementConvention(Enum):
    T_PLUS_0 = "T+0"
    T_PLUS_1 = "T+1"
    T_PLUS_2 = "T+2"
    T_PLUS_3 = "T+3"
    T_PLUS_5 = "T+5"
    SAME_DAY = "SAME_DAY"
    NEXT_DAY = "NEXT_DAY"


class InstrumentType(Enum):
    GOVERNMENT_BOND = "GOVERNMENT_BOND"
    CORPORATE_BOND = "CORPORATE_BOND"
    CORPORATE_BOND_144A = "CORPORATE_BOND_144A"
    MUNICIPAL_BOND = "MUNICIPAL_BOND"
    TREASURY_BILL = "TREASURY_BILL"
    COMMERCIAL_PAPER = "COMMERCIAL_PAPER"
    CERTIFICATE_OF_DEPOSIT = "CERTIFICATE_OF_DEPOSIT"
    CDS_SINGLE_NAME = "CDS_SINGLE_NAME"
    CDS_INDEX = "CDS_INDEX"
    REPO = "REPO"
    REVERSE_REPO = "REVERSE_REPO"
    MONEY_MARKET = "MONEY_MARKET"
    EQUITY = "EQUITY"
    FX_SPOT = "FX_SPOT"
    FX_FORWARD = "FX_FORWARD"


# Holiday calendars - cached globally
_HOLIDAY_CACHE = {}


def _normalize_years(years):
    if isinstance(years, int):
        years = [years]
    years = sorted(int(y) for y in years)
    return years

def _dates_to_yyyy_mm_dd_dict(dates, years):
    if not dates:
        return {}
    year_set = set(years)
    out = {}
    for x in dates:
        d = _as_date(x)
        if d is None or d.year not in year_set:
            continue
        out[d] = d.strftime("%Y-%m-%d")
    return out

def _as_date(x):
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()

    # numpy.datetime64 support without hard dependency
    # (np.datetime64 -> python datetime/date via .astype or .item where available)
    try:
        # numpy datetime64 has .astype and sometimes .item
        if hasattr(x, "item"):
            v = x.item()
            if isinstance(v, datetime):
                return v.date()
            if isinstance(v, date):
                return v
        if hasattr(x, "astype"):
            v = x.astype("datetime64[D]").astype(datetime)
            return v.date() if isinstance(v, datetime) else v
    except Exception:
        pass

    try:
        if hasattr(x, "to_pydatetime"):
            v = x.to_pydatetime()
            if isinstance(v, datetime):
                return v.date()
    except Exception:
        pass

    # last resort: if it quacks like a date
    if hasattr(x, "year") and hasattr(x, "month") and hasattr(x, "day"):
        try:
            return date(int(x.year), int(x.month), int(x.day))
        except Exception:
            return None

    return None

def _nth_weekday_of_month(year, month, weekday, n):
    # weekday: Mon=0 .. Sun=6
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=delta)
    return first_occurrence + timedelta(days=7 * (n - 1))


def _thanksgiving_day(year):
    # US Thanksgiving: 4th Thursday of November
    return _nth_weekday_of_month(year, 11, 3, 4)


def _observed_if_weekend(d):
    # Common US observed convention (Sat->Fri, Sun->Mon)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nyse_early_closes_for_year(year):
    out = set()

    bf = _thanksgiving_day(year) + timedelta(days=1)
    if bf.weekday() < 5:
        out.add(bf)

    xmas = date(year, 12, 25)
    xmas_obs = _observed_if_weekend(xmas)
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5 and xmas_eve != xmas_obs:
        out.add(xmas_eve)

    # Pre-Independence Day early close patterns:
    # - If July 4 is Tue: Mon Jul 3 early close
    # - If July 4 is Thu: Wed Jul 3 early close
    # - If July 4 is Fri: Thu Jul 3 early close
    # - If July 4 is Mon: Fri Jul 1 early close
    july4 = date(year, 7, 4)
    if july4.weekday() == 0:  # Monday
        d = date(year, 7, 1)
        if d.weekday() < 5:
            out.add(d)
    elif july4.weekday() in (1, 3, 4):  # Tue/Thu/Fri
        d = date(year, 7, 3)
        if d.weekday() < 5:
            out.add(d)

    return out

def _try_financial_calendar(calendar_name, years):
    if not ENABLE_FINANCIAL_CALENDARS:
        return None

    # python-holidays may or may not ship a "financial" module depending on version.
    # We attempt a few common names safely.
    try:
        from holidays import financial as _fin  # type: ignore
    except Exception:
        return None

    name = calendar_name.lower()
    candidates = []
    if name == "nyse":
        candidates = ["NYSE", "NewYorkStockExchange"]
    elif name == "sifma":
        candidates = ["SIFMA", "SIFMAUS", "SIFMA_US", "SIFMA_USA"]

    for cls_name in candidates:
        cls = getattr(_fin, cls_name, None)
        if cls is None:
            continue
        try:
            return cls(years=years)
        except Exception:
            continue

    return None

def _build_early_closes(years):
    dates = []
    for y in years:
        dates.extend(sorted(_nyse_early_closes_for_year(y)))
    return _dates_to_yyyy_mm_dd_dict(dates, years)

def _build_us_fallback(years):
    return holidays.country_holidays("US", years=years)

def _build_nyse_fallback_from_us(years):
    us = holidays.country_holidays("US", years=years)
    for y in years:
        easter = _western_easter_date(y)
        good_friday = easter - timedelta(days=2)
        us[good_friday] = "Good Friday"
    return us

def _western_easter_date(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def _fake_dict_from_np_list(x, years):
  if not x: return
  if not isinstance(x[0], np.datetime64): return {y:y.strftime("%Y-%m-%d") for y in x if y.year in years}
  return {y.item():y.item().strftime("%Y-%m-%d") for y in x if y.item().year in years}

def _get_holiday_calendar(calendar_name: str, years: Union[int, List[int]]):
    years = _normalize_years(years)
    cache_key = f"{calendar_name.lower()}_{'_'.join(map(str, years))}"

    cached = _HOLIDAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    name = calendar_name.lower()

    try:
        if name == "early_closes":
            cal = _build_early_closes(years)
            _HOLIDAY_CACHE[cache_key] = cal
            return cal

        fin = _try_financial_calendar(name, years)
        if fin is not None:
            _HOLIDAY_CACHE[cache_key] = fin
            return fin

        # Fallbacks
        if name == "nyse":
            cal = _build_nyse_fallback_from_us(years)
        elif name == "sifma":
            cal = _build_us_fallback(years)
        else:
            cal = _build_us_fallback(years)

        _HOLIDAY_CACHE[cache_key] = cal
        return cal

    except Exception:
        # Last-resort fallback: US holidays only
        cal = _build_us_fallback(years)
        _HOLIDAY_CACHE[cache_key] = cal
        return cal


class LocaleSupport:
    """Support for international date formats and conventions"""

    def __init__(self):
        self.locale_formats = {
            'en_US': {
                'date_formats': ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d'],
                'datetime_formats': ['%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p'],
                'first_weekday': 6,  # Sunday
            },
            'en_GB': {
                'date_formats': ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d'],
                'datetime_formats': ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y %I:%M:%S %p'],
                'first_weekday': 0,  # Monday
            },
            'de_DE': {
                'date_formats': ['%d.%m.%Y', '%d-%m-%Y', '%Y-%m-%d'],
                'datetime_formats': ['%d.%m.%Y %H:%M:%S', '%d.%m.%Y %H:%M'],
                'first_weekday': 0,  # Monday
            },
            'fr_FR': {
                'date_formats': ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d'],
                'datetime_formats': ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M'],
                'first_weekday': 0,  # Monday
            },
            'ja_JP': {
                'date_formats': ['%Y/%m/%d', '%Y-%m-%d', '%Y年%m月%d日'],
                'datetime_formats': ['%Y/%m/%d %H:%M:%S', '%Y年%m月%d日 %H:%M:%S'],
                'first_weekday': 0,  # Monday
            },
            'zh_CN': {
                'date_formats': ['%Y/%m/%d', '%Y-%m-%d', '%Y年%m月%d日'],
                'datetime_formats': ['%Y/%m/%d %H:%M:%S', '%Y年%m月%d日 %H:%M:%S'],
                'first_weekday': 0,  # Monday
            }
        }

    def parse_date_locale(self, date_str: str, locale_code: str = 'en_US') -> Optional[datetime]:
        """Parse date string using locale-specific formats"""
        locale_info = self.locale_formats.get(locale_code, self.locale_formats['en_US'])

        for fmt in locale_info['date_formats']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def format_date_locale(self, date_obj: datetime, locale_code: str = 'en_US') -> str:
        """Format date according to locale conventions"""
        locale_info = self.locale_formats.get(locale_code, self.locale_formats['en_US'])
        return date_obj.strftime(locale_info['date_formats'][0])


class CreditInstrumentRules:
    """Credit market specific settlement and convention rules"""

    def __init__(self):
        self.settlement_rules = {
            InstrumentType.GOVERNMENT_BOND: {
                'US': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_ACT},
                'GB': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_ACT},
                'DE': {'convention': SettlementConvention.T_PLUS_2, 'day_count': DayCountConvention.ACT_ACT},
                'JP': {'convention': SettlementConvention.T_PLUS_2, 'day_count': DayCountConvention.ACT_365},
            },
            InstrumentType.CORPORATE_BOND: {
                'US': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.THIRTY_360},
                'GB': {'convention': SettlementConvention.T_PLUS_2, 'day_count': DayCountConvention.ACT_ACT},
                'DE': {'convention': SettlementConvention.T_PLUS_2, 'day_count': DayCountConvention.ACT_ACT},
                'JP': {'convention': SettlementConvention.T_PLUS_3, 'day_count': DayCountConvention.ACT_365},
            },
            InstrumentType.CORPORATE_BOND_144A: {
                'US': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.THIRTY_360},
                'GB': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_ACT},
                'DE': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_ACT},
                'JP': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_365},
            },
            InstrumentType.CDS_SINGLE_NAME: {
                'US': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_360},
                'GB': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_360},
                'DE': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_360},
                'JP': {'convention': SettlementConvention.T_PLUS_1, 'day_count': DayCountConvention.ACT_360},
            },
            InstrumentType.CDS_INDEX: {
                'US': {'convention': SettlementConvention.T_PLUS_3, 'day_count': DayCountConvention.ACT_360},
                'GB': {'convention': SettlementConvention.T_PLUS_3, 'day_count': DayCountConvention.ACT_360},
                'DE': {'convention': SettlementConvention.T_PLUS_3, 'day_count': DayCountConvention.ACT_360},
                'JP': {'convention': SettlementConvention.T_PLUS_3, 'day_count': DayCountConvention.ACT_360},
            },
            InstrumentType.REPO: {
                'US': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_360},
                'GB': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_365},
                'DE': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_360},
                'JP': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_365},
            },
            InstrumentType.MONEY_MARKET: {
                'US': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_360},
                'GB': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_365},
                'DE': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_360},
                'JP': {'convention': SettlementConvention.SAME_DAY, 'day_count': DayCountConvention.ACT_365},
            }
        }

    def get_settlement_convention(self, instrument_type: InstrumentType, country: str = 'US') -> SettlementConvention:
        """Get settlement convention for instrument type and country"""
        rules = self.settlement_rules.get(instrument_type, {})
        country_rules = rules.get(country, rules.get('US', {}))
        return country_rules.get('convention', SettlementConvention.T_PLUS_2)

    def get_day_count_convention(self, instrument_type: InstrumentType, country: str = 'US') -> DayCountConvention:
        """Get day count convention for instrument type and country"""
        rules = self.settlement_rules.get(instrument_type, {})
        country_rules = rules.get(country, rules.get('US', {}))
        return country_rules.get('day_count', DayCountConvention.ACT_360)

# Global configuration
class DateConfig:
    """Global configuration for date utilities"""

    def __init__(self):
        self.default_tz = DEFAULT_TZ
        self.system_tz = tzlocal.get_localzone()
        self.default_locale = 'en_US'
        self.default_settlement_country = 'US'
        self.strict_parsing = False
        self.cache_size = 1000
        # Use lru_cache for more robust caching
        self._cache = {}

        # Initialize support classes
        self.locale_support = LocaleSupport()

    def set_timezone(self, tz: Union[str, pytz.BaseTzInfo]):
        """Set default timezone"""
        if isinstance(tz, str):
            self.default_tz = pytz.timezone(tz)
        else:
            self.default_tz = tz

    def set_locale(self, locale_code: str):
        """Set default locale"""
        if locale_code in self.locale_support.locale_formats:
            self.default_locale = locale_code
        else:
            warnings.warn(f"Locale {locale_code} not supported, using en_US")
            self.default_locale = 'en_US'


# Global config instance
config = DateConfig()

# Decorators
def robust_date_handler(func):
    """Decorator to add robust error handling to functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if config.strict_parsing:
                raise
            warnings.warn(f"Date processing error in {func.__name__}: {e}")
            return None

    return wrapper


# ===================================================
# Business Days - Backwards Compatibility
# ===================================================

def latest_biz_datetime(date_input=None, utc=True):
    if (date_input is None) or isinstance(date_input, str):
        date_input = parse_single_datetime(date_input, biz_days=False, utc=utc)
    dt = pd.Timestamp(date_input)
    return dt.to_pydatetime() if is_business_day(dt) else (dt - US_BUSINESS_DAY).to_pydatetime()

def latest_biz_date(date_input=None, utc=True):
    return latest_biz_datetime(date_input, utc).date()

def next_biz_date(date_input=None, n=1):
    """Get next business date"""
    if date_input is None:
        date_input = get_today(utc=False)

    if (date_input is None) or isinstance(date_input, str):
        date_input = parse_single_datetime(date_input)

    date_pd = pd.Timestamp(date_input)
    return (date_pd + n*US_BUSINESS_DAY).to_pydatetime().date()

def next_date(date_input=None, n=1, biz_days=True):
    """Get next calendar date"""
    if (date_input is None) or isinstance(date_input, str):
        date_input = parse_single_date(date_input, biz_days=biz_days)

    dt = pd.Timestamp(date_input).to_pydatetime()
    return (dt + timedelta(days=n)).date()

def next_biz_date_from_today(n=1, utc=True):
    """Get nth business date from today"""
    return next_biz_date(get_today(utc=utc), n)

def prev_biz_date_from_today(utc=True):
    """Get previous business date from today"""
    return next_biz_date(get_today(utc=utc), n=-1)

def next_settle_date_from_today(n=DEFAULT_SETTLE_DAYS, utc=True):
    """Get settlement date from today"""
    return next_biz_date(get_today(utc=utc), n)


# ===================================================
# BVAL - Backwards Compatibility
# ===================================================

def get_utc_bval_mappings(today: datetime = None):
    """Get UTC BVAL mappings (backwards compatibility)"""
    if not today: today = get_today(utc=True)
    if isinstance(today, datetime):
        date_obj = today.date()
    else:
        date_obj = today

    mapping = {
        "NY 3PM": ("America/New_York", datetimetime(15, 0)),
        "NY 4PM": ("America/New_York", datetimetime(16, 0)),
        "LO 12PM": ("Europe/London", datetimetime(12, 0)),
        "LO 3PM": ("Europe/London", datetimetime(15, 0)),
        "LO 415PM": ("Europe/London", datetimetime(16, 15)),
        "LO 4PM": ("Europe/London", datetimetime(16, 0)),
        "TO 3PM": ("Asia/Tokyo", datetimetime(15, 0)),
        "TO 4PM": ("Asia/Tokyo", datetimetime(16, 0)),
        "TO 5PM": ("Asia/Tokyo", datetimetime(17, 0)),
        "SH 5PM": ("Asia/Shanghai", datetimetime(17, 0)),
        "SY 5PM": ("Australia/Sydney", datetimetime(17, 0)),
        "NY 3PM HOLIDAY": ("UTC", datetimetime(17, 0)),
        "NY 4PM HOLIDAY": ("UTC", datetimetime(18, 0)),
    }

    result = {}
    for mnemonic, (tzname, local_time) in mapping.items():
        # Use pytz for localization to be consistent
        tz = pytz.timezone(tzname)
        local_dt = tz.localize(datetime.combine(date_obj, local_time))
        utc_dt = local_dt.astimezone(pytz.UTC)
        utc_str = utc_dt.strftime("%H:%M:%S")

        # Apply transformations (Leaving intentional edge case as is)
        if mnemonic == "LO 4PM":
            mnemonic = "LO 4:15PM"
        elif mnemonic == "NY 3PM HOLIDAY":
            mnemonic = "NY 3PM H"
        elif mnemonic == "NY 4PM HOLIDAY":
            mnemonic = "NY 4PM H"

        result[utc_str] = mnemonic

    return result


def get_bval_snap(dtime):
    """Get BVAL snapshot name for given datetime"""
    if isinstance(dtime, str):
        dtime = datetime.fromisoformat(dtime)

    mappings = get_utc_bval_mappings(dtime)
    return mappings.get(dtime.strftime("%H:%M:%S"), None)


# ===================================================
# Helpers - Backwards Compatibility
# ===================================================

def seq_biz_days(sd, ed):
    """Generate sequence of business days"""
    return [x.to_pydatetime().date() for x in pd.bdate_range(start=sd, end=ed)]


# ===================================================
# Parsers - Backwards Compatibility
# ===================================================

def parse_single_date(d, biz_days=True, utc=False):
  my_tz = "utc" if utc else tzlocal.get_localzone()
  my_date = parse_date(d, tz=my_tz, return_format='datetime')
  return latest_biz_date(my_date) if biz_days else my_date.date()

def parse_single_datetime(d, biz_days=True, utc=False):
  try:
    my_tz = "utc" if utc else None
    my_date = parse_date(d, tz=my_tz, return_format='date')
    my_datetime = date_to_datetime(my_date)
    try:
        my_time = parse_single_time(d) if isinstance(d, datetime) else my_datetime.time()
    except:
        my_time = my_datetime.time()
    my_next_date = latest_biz_date(my_datetime) if biz_days else my_datetime.date()
    res = datetime.combine(my_next_date, my_time)
    return res.replace(tzinfo=timezone.utc) if utc else res
  except Exception as e:
    return _parse_single_datetime(d)

def parse_single_time(time_str_or_epoch, fmt="%H:%M:%S.%f"):
    """Parse single time"""
    if isinstance(time_str_or_epoch, str):
        try:
            return datetime.strptime(time_str_or_epoch, fmt).time()
        except ValueError:
            try:
                return datetime.strptime(time_str_or_epoch, "%H:%M:%S").time()
            except ValueError:
                raise ValueError("Invalid time format. Expected format: 'HH:MM:SS.sss' or 'HH:MM:SS'")
    elif isinstance(time_str_or_epoch, (int, float)):
        return kdb_epoch_to_time(time_str_or_epoch)
    else:
        raise TypeError("Input must be a string or an epoch timestamp.")


def _parse_single_datetime(datetime_str_or_epoch, fmt="%Y.%m.%dT%H:%M:%S.%f"):
    """Parse single datetime"""
    if isinstance(datetime_str_or_epoch, str):
        # Try multiple formats
        formats = [
            fmt,
            "%Y.%m.%dT%H:%M:%S",
            "%Y.%m.%d %H:%M:%S"
        ]

        for f in formats:
            try:
                return datetime.strptime(datetime_str_or_epoch, f)
            except ValueError:
                continue

        # Try whenever if available
        if HAS_WHENEVER:
            try:
                return Instant.parse_common_iso(datetime_str_or_epoch).py_datetime()
            except Exception:
                pass

        # Try friendly parser
        try:
            return friendly_parse_datetime(datetime_str_or_epoch)
        except Exception:
            raise ValueError("Invalid datetime format. Expected format: 'YYYY.MM.DDTHH:MM:SS.sss'")

    elif isinstance(datetime_str_or_epoch, (int, float)):
        return kdb_epoch_to_datetime(datetime_str_or_epoch)
    else:
        raise TypeError("Input must be a string or an epoch timestamp.")


# ===================================================
# Today
# ===================================================

def get_today(utc=True):
    """Get today's date"""
    return datetime.now(timezone.utc).date() if utc else date.today()

def is_today(x, utc=True):
    """Check if date is today"""
    if x is None: return True
    return parse_single_date(x, utc=utc, biz_days=False) == get_today(utc)

def isonow(utc=True):
    """Get current datetime as ISO string"""
    if utc: return datetime.now(tz=timezone.utc).isoformat()
    return datetime.now().isoformat()

def now_date(utc=True, fmt="%Y-%m-%d"):
    if utc: return datetime.now(tz=timezone.utc).strftime(fmt)
    return datetime.now().strftime(fmt)

def now_time(utc=True, fmt:str|None="%H:%M:%S"):
    base = datetime.now(tz=timezone.utc) if utc else datetime.now()
    return base.strftime(fmt) if fmt is not None else base

def now_datetime(utc=True):
    if utc:
        return datetime.now(tz=timezone.utc).replace(microsecond=0)
    return datetime.now().replace(microsecond=0)

def date_to_datetime(dt, time=None, utc=True, tz=None, biz=True):
    my_time = now_time(utc, fmt=None) if time is None else time

    if isinstance(my_time, str):
        my_time = parse_time_only(my_time)
    if isinstance(my_time, datetime):
        my_time = my_time.time()

    my_date = parse_date(dt) if not isinstance(dt, date) else dt
    if biz:
        latest_biz_date(my_date, utc)

    res = datetime.combine(my_date, my_time)
    if utc:
        res = res.replace(tzinfo=timezone.utc)
    elif tz:
        res.replace(tzinfo=tz)
    return res



# ===================================================
# Timezones - Backwards Compatibility
# ===================================================

def is_datetime_obj(x):
    """Check if object is datetime"""
    return isinstance(x, datetime)


def is_date_obj(x):
    """Check if object is date (but not datetime)"""
    return isinstance(x, date) and not isinstance(x, datetime)


def get_local_timezone():
    """Get local timezone"""
    return tzlocal.get_localzone()


def convert_date_to_utc(d: date):
    """Convert date to UTC by assuming it's a local date and finding its UTC representation."""
    if not isinstance(d, date) or isinstance(d, datetime):
        return d  # Only convert date objects

    local_tz = get_local_timezone()
    # Localize the start of the day and then convert to UTC
    return local_tz.localize(datetime.combine(d, time.min)).astimezone(pytz.UTC).date()


def _zero_time():
    """Get zero time"""
    return time.min


def _to_whenever(d):
    """Convert to whenever object"""
    if not HAS_WHENEVER:
        return d

    if isinstance(d, (whenever.Date, whenever.Time, Instant, PlainDateTime, whenever.ZonedDateTime)):
        return d
    if isinstance(d, str):
        d = parse_single_datetime(d)
    d = datetime.combine(d, _zero_time()) if is_date_obj(d) else d
    return PlainDateTime.from_py_datetime(d)


def convert_to_tz(d, new_tz_str: str):
    """Convert to timezone"""
    new_tz = pytz.timezone(new_tz_str)

    # Fallback implementation if `whenever` is not used
    if not HAS_WHENEVER:
        if isinstance(d, str):
            d = parse_single_datetime(d)
        if is_date_obj(d):
            d = datetime.combine(d, _zero_time())

        if d.tzinfo is None:
            d = pytz.utc.localize(d)

        return d.astimezone(new_tz)

    i = _to_whenever(d)
    if not isinstance(i, whenever.ZonedDateTime):
        i = i.assume_tz('UTC')

    return i.to_tz(new_tz_str)


def _get_tz(tz_name: str) -> ZoneInfo:
    return ZoneInfo(tz_name)

def _ensure_aware_utc_if_naive(d: datetime) -> datetime:
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        return d.replace(tzinfo=timezone.utc)
    return d


def all_timezones():
    """Get all available timezones"""
    if HAS_WHENEVER:
        return whenever.available_timezones()
    else:
        return pytz.all_timezones


# ===================================================
# KDB - Backwards Compatibility
# ===================================================

def ms_since_epoch(t=None):
    """Get milliseconds since epoch"""
    t = t or time_module.time()
    if isinstance(t, str):
        t = datetime.fromisoformat(t).timestamp()
    return int(round(t * 1000))


def to_kdb_date(d):
    """Convert to KDB date format"""
    return parse_single_date(d).strftime("%Y.%m.%d")


def kdb_epoch_to_datetime(kdb_timestamp):
    """Convert KDB timestamp to datetime"""
    seconds_since_2000 = kdb_timestamp / 1_000_000_000.0
    return KDB_EPOCH + timedelta(seconds=seconds_since_2000)


def kdb_epoch_to_time(kdb_timestamp):
    """Convert KDB timestamp to time"""
    return kdb_epoch_to_datetime(kdb_timestamp).time()


def kdb_epoch_to_date(kdb_timestamp):
    """Convert KDB timestamp to date"""
    return kdb_epoch_to_datetime(kdb_timestamp).date()


# ===================================================
# Misc - Backwards Compatibility
# ===================================================

def time_format_guess(s):
    """Guess time format"""
    patterns = [
        # Full Date+Time patterns, with optional ms and TZ
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3,6}([Zz]|[+-]\d{2}:?\d{2})$', '%Y-%m-%dT%H:%M:%S.%f%z'),
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3,6}$', '%Y-%m-%dT%H:%M:%S.%f'),
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([Zz]|[+-]\d{2}:?\d{2})$', '%Y-%m-%dT%H:%M:%S%z'),
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$', '%Y-%m-%dT%H:%M:%S'),
        # Date and time with space separator
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3,6}$', '%Y-%m-%d %H:%M:%S.%f'),
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
        (r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$', '%d/%m/%Y %H:%M:%S'),
        (r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}$', '%d-%m-%Y %H:%M:%S'),
        # Date and time with AM/PM
        (r'^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2} [APMapm]{2}$', '%Y-%m-%d %I:%M:%S %p'),
        (r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APMapm]{2}$', '%m/%d/%Y %I:%M %p'),
        # Time patterns
        (r'^\d{2}:\d{2}:\d{2}.\d{3,6}$', '%H:%M:%S.%f'),
        (r'^\d{2}:\d{2}:\d{2},\d{3,6}$', '%H:%M:%S,%f'),
        (r'^\d{2}:\d{2}:\d{2}.\d{3,6} ?([Zz]|[+-]\d{2}:?\d{2})$', '%H:%M:%S.%f%z'),
        (r'^\d{2}:\d{2}:\d{2} ?([Zz]|[+-]\d{2}:?\d{2})$', '%H:%M:%S%z'),
        (r'^\d{2}:\d{2}:\d{2}$', '%H:%M:%S'),
        (r'^\d{1,2}:\d{2}:\d{2}$', '%H:%M:%S'),
        (r'^\d{2}:\d{2}.\d{1,6}$', '%H:%M.%f'),
        (r'^\d{2}:\d{2}$', '%H:%M'),
        (r'^\d{1,2}:\d{2}$', '%H:%M'),
        (r'^\d{1,2}:\d{2}:\d{2} [APMapm]{2}$', '%I:%M:%S %p'),
        (r'^\d{1,2}:\d{2} [APMapm]{2}$', '%I:%M %p'),
        (r'^\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2} [APMapm]{2}$', '%m/%d/%Y %I:%M %p'),
        # Date only patterns
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
        (r'^\d{1,2}/\d{1,2}/\d{4}$', '%m/%d/%Y'),
        (r'^\d{1,2}-\d{1,2}-\d{4}$', '%d-%m-%Y'),
    ]

    s = s.strip()
    for patt, fmt in patterns:
        if re.match(patt, s):
            return fmt
    return None


# ===================================================
# Temporal Adjustments - Backwards Compatibility
# ===================================================

def first_day_of_week(d=None):
    """Get first day of week"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        days_since_monday = d.weekday()
        return d - timedelta(days=days_since_monday)
    return TemporalAdjuster.first_day_of_week(d or get_today())


def last_day_of_week(d=None):
    """Get last day of week"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        days_until_sunday = 6 - d.weekday()
        return d + timedelta(days=days_until_sunday)
    return TemporalAdjuster.last_day_of_week(d or get_today())


def first_day_of_last_week(d=None):
    """Get first day of last week"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        return first_day_of_week(d - timedelta(weeks=1))
    return TemporalAdjuster.first_day_of_last_week(d or get_today())


def last_day_of_last_week(d=None):
    """Get last day of last week"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        return last_day_of_week(d - timedelta(weeks=1))
    return TemporalAdjuster.last_day_of_last_week(d or get_today())


def first_day_of_next_week(d=None):
    """Get first day of next week"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        return first_day_of_week(d + timedelta(weeks=1))
    return TemporalAdjuster.first_day_of_next_week(d or get_today())


def last_day_of_next_week(d=None):
    """Get last day of next week"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        return last_day_of_week(d + timedelta(weeks=1))
    return TemporalAdjuster.last_day_of_next_week(d or get_today())


def first_day_of_month(d=None):
    """Get first day of month"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        return d.replace(day=1)
    return TemporalAdjuster.first_day_of_month(d or get_today())


def last_day_of_month(d=None):
    """Get last day of month"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        last_day = calendar.monthrange(d.year, d.month)[1]
        return d.replace(day=last_day)
    return TemporalAdjuster.last_day_of_month(d or get_today())


def first_day_of_last_month(d=None):
    """Get first day of last month"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        prev_month = d - relativedelta(months=1)
        return prev_month.replace(day=1)
    return TemporalAdjuster.first_day_of_last_month(d or get_today())


def last_day_of_last_month(d=None):
    """Get last day of last month"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        prev_month = d - relativedelta(months=1)
        last_day = calendar.monthrange(prev_month.year, prev_month.month)[1]
        return prev_month.replace(day=last_day)
    return TemporalAdjuster.last_day_of_last_month(d or get_today())


def first_day_of_next_month(d=None):
    """Get first day of next month"""
    if not HAS_TEMPORAL_ADJUSTER:
        d = d or get_today()
        d = parse_single_date(d) if isinstance(d, str) else d
        next_month = d + relativedelta(months=1)
        return next_month.replace(day=1)
    return TemporalAdjuster.first_day_of_next_month(d or get_today())


# ===================================================
# Core Date Library Functions
# ===================================================

#@robust_date_handler
def parse_date(
    date_input: Any,
    tz: Optional[Union[str, pytz.BaseTzInfo]] = None,
    biz: bool = False,
    return_format: str = 'date'
) -> Optional[Union[datetime, date, str, int]]:

    if date_input is None:
        date_input = 'T'

    target_tz = config.default_tz
    if tz:
        target_tz = pytz.timezone(tz) if isinstance(tz, str) else tz

    if not isinstance(date_input, datetime):
        dt = _parse_to_datetime(date_input, target_tz)
    else:
        dt = date_input

    if biz:
        dt = latest_biz_datetime(dt, is_utc(tz))

    if dt is None:
        return None

    return _format_output(dt, return_format)

def is_utc(tz):
    if tz is None: return True
    return tz is timezone.utc

def _parse_to_datetime(date_input: Any, tz: pytz.BaseTzInfo) -> Optional[datetime]:
    """Internal function to parse various inputs to a timezone-aware datetime"""
    if (date_input is None) or (date_input == ''):
        date_input = datetime.now(tz)

    # Handle datetime objects
    if isinstance(date_input, datetime):
        return date_input.astimezone(tz) if date_input.tzinfo else tz.localize(date_input)

    # Handle date objects
    if isinstance(date_input, date):
        from datetime import time
        if hasattr(tz, 'localize'):
            return tz.localize(datetime.combine(date_input, time.min))
        return datetime.combine(date_input, time.min).astimezone(tz)

    # Handle numeric inputs (timestamps, excel dates, etc.)
    if isinstance(date_input, (int, float, np.integer, np.floating)):
        my_date = _parse_numeric_date(date_input, tz)
        if not isinstance(my_date, datetime):
            my_date = date_to_datetime(my_date, tz=tz, biz=False)
        return my_date

    # Handle string inputs
    if isinstance(date_input, str):
        my_date = _parse_string_date(date_input, tz)
        if not isinstance(my_date, datetime):
            my_date = date_to_datetime(my_date, tz=tz, biz=False)
        return my_date

    # Handle pandas Timestamp
    if hasattr(date_input, 'to_pydatetime'):
        dt = date_input.to_pydatetime()
        # pandas may return a naive datetime or a UTC datetime
        return dt.astimezone(tz) if dt.tzinfo else tz.localize(dt)

    return None


def _parse_numeric_date(num_input: Union[int, float], tz: pytz.BaseTzInfo) -> Optional[datetime]:
    """Parse numeric date inputs (timestamps, excel dates, etc.)"""
    # Handle Unix timestamps (seconds)
    if 10 ** 9 <= num_input < 3 * 10 ** 9:  # reasonable range for seconds
        return datetime.fromtimestamp(num_input, tz=tz)
    # Handle millisecond timestamps
    if 10 ** 12 <= num_input < 3 * 10 ** 12:  # reasonable range for ms
        return datetime.fromtimestamp(num_input / 1000, tz=tz)
    # Handle Excel dates (days since 1899-12-30)
    if 1 <= num_input <= 2958465:
        # Excel epoch is naive, add timedelta and then localize
        naive_dt = datetime(1899, 12, 30) + timedelta(days=num_input)
        return tz.localize(naive_dt)
    # Handle KDB dates (days since 2000-01-01)
    if isinstance(num_input, int) and -73000 <= num_input <= 73000:  # ~200 years range
        return KDB_EPOCH + timedelta(days=num_input)

    return None


def _parse_string_date(str_input: str, tz: pytz.BaseTzInfo) -> Optional[datetime]:
    """Parse string date inputs with various formats"""
    str_input = str_input.strip()

    if re.match(r'(?i)^T([+-]\d+)?$', str_input):
        return _parse_t_notation(str_input, tz)
    if re.match(r'(i?)^\d+[DWMYH]$', str_input):
        return _parse_relative_date(str_input, tz)
    if re.match(r'(i?)^\.z\.d([+-]\d+)?$', str_input):
        return _parse_kdb_format(str_input, tz)

    # Try dateparser for NLP parsing first if available
    if HAS_DATEPARSER:
        try:
            # Tell dateparser to assume the target timezone for naive strings
            parsed = dateparser.parse(str_input, settings={'TO_TIMEZONE': str(tz)})
            if parsed:
                return parsed.astimezone(tz)
        except Exception:
            pass

    # Try dateutil parser as a robust fallback
    try:
        parsed = dateutil_parser.parse(str_input)
        return parsed.astimezone(tz) if parsed.tzinfo else tz.localize(parsed)
    except (ValueError, TypeError):
        pass

    return None


def _parse_t_notation(t_str: str, tz: pytz.BaseTzInfo) -> datetime:
    """Parse T notation (T, T+1, T-1, etc.) as business days"""
    base_date = datetime.now(tz)
    offset = 0
    if len(t_str) > 1:
        offset = int(t_str[1:])
        return add_business_days(base_date, offset)
    return base_date


def _parse_relative_date(rel_str: str, tz: pytz.BaseTzInfo) -> datetime:
    """Parse relative date strings (30D, 1M, 2Y, etc.)"""
    match = re.match(r'(i?)[/+-]*(\d+)([DWMYH])', rel_str)
    if not match: return datetime.now(tz)

    amount, unit = match.groups()
    amount = int(amount)
    base_date = datetime.now(tz)
    unit = unit.upper()

    delta = timedelta()
    if unit == 'D':
        delta = timedelta(days=amount)
    elif unit == 'W':
        delta = timedelta(weeks=amount)
    elif unit == 'H':
        delta = timedelta(hours=amount)
    elif unit == 'M':
        return base_date + relativedelta(months=amount)
    elif unit == 'Y':
        return base_date + relativedelta(years=amount)

    return base_date + delta


def _parse_kdb_format(kdb_str: str, tz: pytz.BaseTzInfo) -> datetime:
    """Parse KDB format (.z.d, .z.d-1, etc.)"""
    offset = 0
    if len(kdb_str) > 4:
        offset = int(kdb_str[4:])

    return datetime.now(tz) + timedelta(days=offset)


def _format_output(dt: datetime, return_format: str) -> Any:
    """Format datetime output to requested format"""
    if return_format == 'datetime': return dt
    if return_format == 'date': return dt.date() if is_datetime_obj(dt) else dt
    if return_format == 'string': return dt.isoformat()
    if return_format == 'timestamp': return int(dt.timestamp())

    # Ensure correct epoch handling for timezone-aware dates
    if return_format == 'excel':
        k = EXCEL_EPOCH.date() if not is_datetime_obj(dt) else EXCEL_EPOCH
        return (dt - k).days + (dt - k).seconds / 86400.0
    if return_format == 'kdb':
        k = KDB_EPOCH.date() if not is_datetime_obj(dt) else KDB_EPOCH
        return (dt - k).days
    if "%" in return_format:
        try:
            if hasattr(dt, 'strftime'):
                return dt.strftime(return_format)
            if hasattr(dt, "format"):
                return dt.format(return_format)
        except Exception:
            pass
    return dt


@robust_date_handler
def is_business_day(date_input: Any, holiday_calendar: str = 'nyse') -> bool:
    """Check if date is a business day"""
    dt = parse_date(date_input, config.default_tz, return_format='date')
    if dt is None: return False

    # Check if it's a weekend
    if dt.weekday() >= 5:  # Monday is 0 and Sunday is 6
        return False

    # Check if it's a holiday
    holidays_cal = _get_holiday_calendar(holiday_calendar, dt.year)
    return dt not in holidays_cal


@robust_date_handler
def add_business_days(date_input: Any, days: int, holiday_calendar: str = 'nyse') -> Optional[datetime]:
    """Add business days to a date - corrected for year boundaries."""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None
    if days == 0:
        # still roll to a business day if the start date isn't one
        return roll_date(dt, rule='following', holiday_calendar=holiday_calendar)

    current_date = dt.date() if is_datetime_obj(dt) else dt
    step = timedelta(days=1) if days > 0 else timedelta(days=-1)
    days_to_move = abs(days)

    # Get holidays for a reasonable range of years to avoid repeated lookups
    start_year = dt.year
    # Estimate end year; add buffer
    end_year = (dt + timedelta(days=days * 1.5) + timedelta(days=365)).year
    year_range = list(range(min(start_year, end_year), max(start_year, end_year) + 1))
    holidays_cal = _get_holiday_calendar(holiday_calendar, year_range)

    while days_to_move > 0:
        current_date += step
        if current_date.weekday() < 5 and current_date not in holidays_cal:
            days_to_move -= 1

    if is_datetime_obj(dt):
        return datetime.combine(current_date, dt.time(), tzinfo=dt.tzinfo)
    return current_date


@robust_date_handler
def get_business_days_between(start_date: Any, end_date: Any, holiday_calendar: str = 'nyse') -> int:
    """Get number of business days between two dates"""
    start_dt = parse_date(start_date, config.default_tz)
    end_dt = parse_date(end_date, config.default_tz)
    if start_dt is None or end_dt is None: return 0

    if start_dt > end_dt: start_dt, end_dt = end_dt, start_dt

    # Get holidays for all years in the range
    years = list(range(start_dt.year, end_dt.year + 1))
    holidays_cal = _get_holiday_calendar(holiday_calendar, years)

    business_days = pd.bdate_range(start_dt.date(), end_dt.date())

    return len([d for d in business_days if d.date() not in holidays_cal])


@robust_date_handler
def add_time(date_input: Any,
             years: int = 0, months: int = 0, days: int = 0,
             hours: int = 0, minutes: int = 0, seconds: int = 0,
             business_days: int = 0,
             holiday_calendar: str = 'nyse') -> Optional[datetime]:
    """Add various time units to a date"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None

    dt += relativedelta(years=years, months=months, days=days, hours=hours, minutes=minutes, seconds=seconds)
    if business_days != 0:
        dt = add_business_days(dt, business_days, holiday_calendar)

    return dt


@robust_date_handler
def roll_date(date_input: Any, rule: str = 'following', holiday_calendar: str = 'nyse') -> Optional[datetime]:
    """Roll date according to business day convention"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None

    if is_business_day(dt, holiday_calendar): return dt

    if rule == 'following':
        return add_business_days(dt, 1, holiday_calendar)
    elif rule == 'preceding':
        return add_business_days(dt, -1, holiday_calendar)
    elif rule == 'modified_following':
        next_bd = add_business_days(dt, 1, holiday_calendar)
        return add_business_days(dt, -1, holiday_calendar) if next_bd and next_bd.month != dt.month else next_bd

    return dt


@robust_date_handler
def get_imm_date(year: int, month: int) -> Optional[datetime]:
    """Get IMM date (3rd Wednesday of March, June, September, December)"""
    if month not in [3, 6, 9, 12]:
        raise ValueError("IMM months are March, June, September, December")

    # Third Wednesday of the month
    first_day = date(year, month, 1)
    # weekday() -> Monday is 0 and Sunday is 6. Wednesday is 2.
    days_to_wed = (2 - first_day.weekday() + 7) % 7
    first_wed = first_day + timedelta(days=days_to_wed)
    third_wed = first_wed + timedelta(weeks=2)

    return datetime.combine(third_wed, time.min, tzinfo=config.default_tz)

def add_date_to_time(date, time, utc=True):
    my_tz = config.default_tz if utc else None
    my_date = parse_date(date)
    my_time = parse_single_time(time)
    return datetime.combine(my_date, my_time, tzinfo=my_tz)

class BVALSnapshotTimes:
    def __init__(self):
        self.snapshot_times = {
            'TOKYO': {'TO3PM': (15, 0), 'TO4PM': (16, 0), 'TO5PM': (17, 0)},
            'SYDNEY': {'SY5PM': (17, 0)},
            'SHANGHAI': {'SH5PM': (17, 0)},
            'LONDON': {'LO12PM': (12, 0), 'LO3PM': (15, 0), 'LO4PM': (16, 0), 'LO415': (16, 15)},
            'NEW_YORK': {'NY3PM': (15, 0), 'NY4PM': (16, 0)}
        }
        self.market_timezones = {
            'TOKYO': 'Asia/Tokyo', 'SYDNEY': 'Australia/Sydney', 'SHANGHAI': 'Asia/Shanghai',
            'LONDON': 'Europe/London', 'NEW_YORK': 'America/New_York'
        }
        self.early_close_times = {'NY3PM': (13, 0), 'NY4PM': (13, 0)}
        self.monikers = {y.replace("PM","") for x in self.snapshot_times.values() for y in list(x.keys())}

        self.time_to_moniker = {}
        from zoneinfo import ZoneInfo
        for lbl, times in self.snapshot_times.items():
            for mon, (hr, mn) in times.items():
                tz_aware_time = datetime(year=1990, month=1, day=1, hour=hr, minute=mn, tzinfo=ZoneInfo(self.market_timezones[lbl]))
                utc_time = tz_aware_time.astimezone(pytz.UTC).time()
                self.time_to_moniker[utc_time] = mon
        self.moniker_to_time = {v:k for k,v in self.time_to_moniker.items()}

bval = BVALSnapshotTimes()


def _is_sifma_early_close(date_input: Any) -> bool:
    """Check if date is a SIFMA early close day"""
    dt = parse_date(date_input, return_format='date')
    if dt is None: return False
    sifma_holidays = _get_holiday_calendar('early_closes', dt.year)
    return dt in sifma_holidays.keys()


@robust_date_handler
def get_bval_snapshot_times(date_input: Any, market: str = 'NEW_YORK') -> Dict[str, datetime]:
    """Get all BVAL snapshot times for a given date and market"""
    dt = parse_date(date_input)
    if dt is None: return {}

    market = market.upper()
    if market not in bval.snapshot_times:
        raise ValueError(f"Unsupported market: {market}")

    holiday_cal_name = 'sifma' if market == 'NEW_YORK' else 'london'
    if not is_business_day(dt, holiday_cal_name):
        return {}

    market_tz = pytz.timezone(bval.market_timezones[market])
    market_date = dt.astimezone(market_tz)
    is_early_close = market == 'NEW_YORK' and _is_sifma_early_close(market_date)

    snapshot_times = {}
    for moniker, (hour, minute) in bval.snapshot_times[market].items():
        use_hour, use_minute = (hour, minute)
        if is_early_close and moniker in bval.early_close_times:
            use_hour, use_minute = bval.early_close_times[moniker]

        snapshot_time = market_date.replace(hour=use_hour, minute=use_minute, second=0, microsecond=0)
        snapshot_times[moniker] = snapshot_time

    return snapshot_times


# Credit market functions
@robust_date_handler
def calculate_day_count_fraction(start_date: Any, end_date: Any,
                                 convention: DayCountConvention = DayCountConvention.ACT_360) -> Optional[Decimal]:
    """Calculate day count fraction between two dates"""
    start_dt = parse_date(start_date, config.default_tz)
    end_dt = parse_date(end_date, config.default_tz)
    if start_dt is None or end_dt is None: return None
    if start_dt > end_dt: start_dt, end_dt = end_dt, start_dt

    if convention == DayCountConvention.ACT_360:
        return Decimal((end_dt - start_dt).days) / Decimal(360)
    elif convention in (DayCountConvention.ACT_365, DayCountConvention.ACT_365_FIXED):
        return Decimal((end_dt - start_dt).days) / Decimal(365)
    elif convention == DayCountConvention.ACT_ACT:
        return _calculate_act_act(start_dt, end_dt)
    elif convention == DayCountConvention.THIRTY_360:
        return _calculate_30_360(start_dt, end_dt)
    elif convention == DayCountConvention.THIRTY_360_EUROPEAN:
        return _calculate_30e_360(start_dt, end_dt)
    elif convention == DayCountConvention.BUS_252:
        return Decimal(get_business_days_between(start_dt, end_dt)) / Decimal(252)

    return Decimal((end_dt - start_dt).days) / Decimal(360)


def _calculate_act_act(start_dt: datetime, end_dt: datetime) -> Decimal:
    """Calculate ACT/ACT day count fraction"""
    if start_dt.year == end_dt.year or not calendar.isleap(start_dt.year):
        days_in_year = 366 if calendar.isleap(start_dt.year) else 365
        return Decimal((end_dt - start_dt).days) / Decimal(days_in_year)

    # Handle period crossing a leap year boundary
    year_end = datetime(start_dt.year, 12, 31, tzinfo=start_dt.tzinfo)
    days_in_first_year = 366 if calendar.isleap(start_dt.year) else 365
    frac1 = Decimal((year_end - start_dt).days + 1) / Decimal(days_in_first_year)

    year_start = datetime(end_dt.year, 1, 1, tzinfo=end_dt.tzinfo)
    days_in_second_year = 366 if calendar.isleap(end_dt.year) else 365
    frac2 = Decimal((end_dt - year_start).days + 1) / Decimal(days_in_second_year)

    return frac1 + frac2 + Decimal(end_dt.year - start_dt.year - 1)


def _calculate_30_360(start_dt: datetime, end_dt: datetime) -> Decimal:
    """Calculate 30/360 day count fraction"""
    d1, m1, y1 = start_dt.day, start_dt.month, start_dt.year
    d2, m2, y2 = end_dt.day, end_dt.month, end_dt.year
    if d1 == 31: d1 = 30
    if d2 == 31 and d1 == 30: d2 = 30
    return Decimal(360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / Decimal(360)


def _calculate_30e_360(start_dt: datetime, end_dt: datetime) -> Decimal:
    """Calculate 30E/360 (European) day count fraction"""
    d1, m1, y1 = start_dt.day, start_dt.month, start_dt.year
    d2, m2, y2 = end_dt.day, end_dt.month, end_dt.year
    if d1 == 31: d1 = 30
    if d2 == 31: d2 = 30
    return Decimal(360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / Decimal(360)



@robust_date_handler
def get_cds_payment_dates(start_date: Any, maturity_date: Any) -> List[datetime]:
    """Get CDS payment dates (20th of Mar, Jun, Sep, Dec)"""
    start_dt = parse_date(start_date)
    maturity_dt = parse_date(maturity_date)
    if start_dt is None or maturity_dt is None: return []

    payment_dates = []
    payment_months = [3, 6, 9, 12]

    # Start generating potential dates from the quarter of the start_date
    current_date = start_dt.replace(day=1)

    while True:
        # Find the next payment month
        month_found = False
        for month in payment_months:
            if month >= current_date.month:
                current_date = current_date.replace(month=month)
                month_found = True
                break
        if not month_found:
            current_date = current_date.replace(year=current_date.year + 1, month=payment_months[0])

        payment_date = current_date.replace(day=20, tzinfo=start_dt.tzinfo)

        # Ensure the candidate date is after the start date
        if payment_date <= start_dt:
            current_date += relativedelta(months=3)  # Move to next quarter
            continue

        rolled_date = roll_date(payment_date, 'following')

        if rolled_date > maturity_dt:
            break

        payment_dates.append(rolled_date)
        current_date += relativedelta(months=3)  # Move to next quarter

    return payment_dates


@robust_date_handler
def calculate_accrued_interest(start_date: Any, end_date: Any,
                               annual_rate: Decimal,
                               convention: DayCountConvention = DayCountConvention.ACT_360,
                               principal: Decimal = Decimal('1000000')) -> Optional[Decimal]:
    """Calculate accrued interest between two dates"""
    fraction = calculate_day_count_fraction(start_date, end_date, convention)
    return principal * annual_rate * fraction if fraction is not None else None


@robust_date_handler
def calculate_bond_equivalent_yield(start_date: Any, end_date: Any,
                                    discount_rate: Decimal) -> Optional[Decimal]:
    """Calculate bond equivalent yield for money market instruments"""
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if start_dt is None or end_dt is None: return None

    days = (end_dt - start_dt).days
    if days <= 0: return None

    discount_rate = Decimal(discount_rate)
    days_dec = Decimal(days)

    # BEY = (365 * Discount Rate) / (360 - (Days * Discount Rate))
    denominator = Decimal(360) - (days_dec * discount_rate)
    if denominator <= 0:
        return None  # Avoid division by zero or negative

    bey = (Decimal(365) * discount_rate) / denominator
    return bey


@robust_date_handler
def get_bval_snapshot_time(date_input: Any, moniker: str, market: str = None) -> Optional[datetime]:
    """Get specific BVAL snapshot time by moniker"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None:
        return None

    # Auto-detect market from moniker if not provided
    if market is None:
        upper_moniker = moniker.upper()
        if upper_moniker.startswith('TO'):
            market = 'TOKYO'
        elif upper_moniker.startswith('SY'):
            market = 'SYDNEY'
        elif upper_moniker.startswith('SH'):
            market = 'SHANGHAI'
        elif upper_moniker.startswith('LO'):
            market = 'LONDON'
        elif upper_moniker.startswith('NY'):
            market = 'NEW_YORK'
        else:
            raise ValueError(f"Cannot auto-detect market from moniker: {moniker}")

    market = market.upper()
    snapshot_times = get_bval_snapshot_times(dt, market)

    return snapshot_times.get(moniker)


@robust_date_handler
def is_bval_snapshot_time(date_input: Any, moniker: str = None, market: str = None, tolerance_minutes: int = 0) -> bool:
    """Check if given datetime is a BVAL snapshot time"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None:
        return False

    # If moniker specified, check specific snapshot
    if moniker:
        snapshot_time = get_bval_snapshot_time(dt, moniker, market)
        if snapshot_time is None: return False
        time_diff = abs((dt - snapshot_time).total_seconds())
        return time_diff <= tolerance_minutes * 60

    # Check all markets if no moniker specified
    for market_name in bval.snapshot_times.keys():
        snapshot_times = get_bval_snapshot_times(dt, market_name)
        market_tz = pytz.timezone(bval.market_timezones[market_name])
        market_dt = dt.astimezone(market_tz)

        for snapshot_time in snapshot_times.values():
            time_diff = abs((market_dt - snapshot_time).total_seconds())
            if time_diff <= tolerance_minutes * 60:
                return True

    return False


@robust_date_handler
def get_bval_snapshot_moniker(date_input: Any, market: str = 'NEW_YORK') -> Optional[str]:
    """Get the moniker of the BVAL snapshot for a given time"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None

    market = market.upper()
    if market not in bval.snapshot_times: return None

    snapshot_times = get_bval_snapshot_times(dt, market)
    market_tz = pytz.timezone(bval.market_timezones[market])
    market_dt = dt.astimezone(market_tz)

    # Find closest snapshot within 1 minute
    for moniker, snapshot_time in snapshot_times.items():
        time_diff = abs((market_dt - snapshot_time).total_seconds())
        if time_diff <= 60:  # Within 1 minute
            return moniker

    return None


@robust_date_handler
def get_next_bval_snapshot(date_input: Any, market: str = 'NEW_YORK') -> Optional[Tuple[str, datetime]]:
    """Get the next BVAL snapshot time and moniker after the given datetime"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None

    market = market.upper()
    market_tz = pytz.timezone(bval.market_timezones[market])
    market_dt = dt.astimezone(market_tz)

    # Get today's snapshot times, sorted by time
    today_snapshots = sorted(get_bval_snapshot_times(market_dt, market).items(), key=lambda x: x[1])

    # Find next snapshot today
    for moniker, snapshot_time in today_snapshots:
        if snapshot_time > market_dt:
            return moniker, snapshot_time

    # No more snapshots today, get first snapshot of next business day
    next_business_day = add_business_days(market_dt, 1)
    if next_business_day:
        next_day_snapshots = sorted(get_bval_snapshot_times(next_business_day, market).items(), key=lambda x: x[1])
        if next_day_snapshots:
            return next_day_snapshots[0]

    return None


@robust_date_handler
def get_previous_bval_snapshot(date_input: Any, market: str = 'NEW_YORK') -> Optional[Tuple[str, datetime]]:
    """Get the previous BVAL snapshot time and moniker before the given datetime"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None

    market = market.upper()
    market_tz = pytz.timezone(bval.market_timezones[market])
    market_dt = dt.astimezone(market_tz)

    # Get today's snapshot times, sorted in reverse
    today_snapshots = sorted(get_bval_snapshot_times(market_dt, market).items(), key=lambda x: x[1], reverse=True)

    # Find previous snapshot today
    for moniker, snapshot_time in today_snapshots:
        if snapshot_time < market_dt:
            return moniker, snapshot_time

    # No earlier snapshots today, get last snapshot of previous business day
    prev_business_day = add_business_days(market_dt, -1)
    if prev_business_day:
        prev_day_snapshots = sorted(get_bval_snapshot_times(prev_business_day, market).items(), key=lambda x: x[1], reverse=True)
        if prev_day_snapshots:
            return prev_day_snapshots[0]

    return None


@robust_date_handler
def get_all_bval_snapshots(date_input: Any) -> Dict[str, Dict[str, datetime]]:
    """Get all BVAL snapshots across all markets for a given date"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return {}

    all_snapshots = {}
    for market in bval.snapshot_times.keys():
        market_snapshots = get_bval_snapshot_times(dt, market)
        if market_snapshots:
            all_snapshots[market] = market_snapshots

    return all_snapshots


@robust_date_handler
def get_bval_snapshots_between(start_date: Any, end_date: Any, market: str = 'NEW_YORK') -> Dict[str, List[datetime]]:
    """Get all BVAL snapshot times between two dates, grouped by moniker"""
    start_dt = parse_date(start_date, config.default_tz)
    end_dt = parse_date(end_date, config.default_tz)
    if start_dt is None or end_dt is None: return {}
    if start_dt > end_dt: start_dt, end_dt = end_dt, start_dt

    snapshots_by_moniker = {}
    current_date = start_dt
    while current_date <= end_dt:
        day_snapshots = get_bval_snapshot_times(current_date, market)
        for moniker, snapshot_time in day_snapshots.items():
            if start_dt <= snapshot_time <= end_dt:
                snapshots_by_moniker.setdefault(moniker, []).append(snapshot_time)
        current_date += timedelta(days=1)

    return snapshots_by_moniker


@robust_date_handler
def is_early_close_day(date_input: Any) -> bool:
    """Check if date is a SIFMA early close day"""
    return _is_sifma_early_close(date_input)


@robust_date_handler
def get_ny_snapshot_actual_time(date_input: Any, moniker: str) -> Optional[datetime]:
    """Get actual NY snapshot time (accounting for early close)"""
    if not moniker.upper().startswith('NY'):
        raise ValueError("This function is only for NY monikers")
    return get_bval_snapshot_time(date_input, moniker, 'NEW_YORK')


@robust_date_handler
def convert_bval_to_utc(date_input: Any, moniker: str, market: str = None) -> Optional[datetime]:
    """Convert BVAL snapshot time to UTC"""
    snapshot_time = get_bval_snapshot_time(date_input, moniker, market)
    return snapshot_time.astimezone(pytz.UTC) if snapshot_time else None


@robust_date_handler
def get_bval_snapshot_window(date_input: Any, moniker: str, market: str = None,
                             window_minutes: int = 5) -> Optional[Tuple[datetime, datetime]]:
    """Get time window around a BVAL snapshot"""
    snapshot_time = get_bval_snapshot_time(date_input, moniker, market)
    if snapshot_time is None: return None
    window_start = snapshot_time - timedelta(minutes=window_minutes)
    window_end = snapshot_time + timedelta(minutes=window_minutes)
    return window_start, window_end


@robust_date_handler
def get_market_snapshots_summary(date_input: Any) -> Dict[str, Any]:
    """Get summary of all market snapshots for a given date"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return {}

    summary = {}
    for market in bval.snapshot_times.keys():
        market_snapshots = get_bval_snapshot_times(dt, market)
        if market_snapshots:
            utc_snapshots = {
                moniker: {'local_time': st, 'utc_time': st.astimezone(pytz.UTC)}
                for moniker, st in market_snapshots.items()
            }
            summary[market] = {
                'snapshots': utc_snapshots,
                'count': len(market_snapshots),
                'early_close': market == 'NEW_YORK' and _is_sifma_early_close(dt)
            }
    return summary


def parse_datetime(datetime_str):
    """Backwards compatibility alias for parse_single_datetime"""
    return _parse_single_datetime(datetime_str)


def parse_date_friendly(date_str):
    """Backwards compatibility alias for friendly parsing"""
    return friendly_parse_date(date_str)


@robust_date_handler
def time_elapsed(start_date: Any, end_date: Any = None, unit: str = 'seconds') -> Optional[float]:
    """Calculate time elapsed between two dates"""
    start_dt = parse_date(start_date, config.default_tz)
    end_dt = parse_date(end_date or datetime.now(), config.default_tz)
    if start_dt is None or end_dt is None: return None

    delta = end_dt - start_dt
    unit = unit.lower()
    if unit == 'seconds': return delta.total_seconds()
    if unit == 'minutes': return delta.total_seconds() / 60
    if unit == 'hours': return delta.total_seconds() / 3600
    if unit == 'days': return delta.total_seconds() / 86400
    if unit == 'weeks': return delta.total_seconds() / 604800
    return delta.total_seconds()


@robust_date_handler
def get_repo_maturity_date(start_date: Any, tenor: str) -> Optional[datetime]:
    """Get repo maturity date based on tenor"""
    start_dt = parse_date(start_date, config.default_tz)
    if start_dt is None: return None

    tenor = tenor.upper()
    if tenor in ('ON', 'O/N'): return add_business_days(start_dt, 1)
    if tenor in ('TN', 'T/N'): return add_business_days(start_dt, 2)
    if tenor in ('SN', 'S/N'): return add_business_days(start_dt, 3)

    match = re.match(r'(\d+)([WMY])', tenor)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        if unit == 'W': return add_business_days(start_dt, num * 5)
        if unit == 'M': return add_time(start_dt, months=num)
        if unit == 'Y': return add_time(start_dt, years=num)

    return None


def _get_settlement_days(convention: SettlementConvention) -> int:
    """Convert settlement convention to number of days"""
    return {
        SettlementConvention.SAME_DAY: 0, SettlementConvention.NEXT_DAY: 1,
        SettlementConvention.T_PLUS_0: 0, SettlementConvention.T_PLUS_1: 1,
        SettlementConvention.T_PLUS_2: 2, SettlementConvention.T_PLUS_3: 3,
        SettlementConvention.T_PLUS_5: 5
    }.get(convention, 2)  # Default T+2



@robust_date_handler
def is_fed_holiday(date_input: Any) -> bool:
    """Check if date is a US Federal Reserve holiday"""
    dt = parse_date(date_input, return_format='date')
    if dt is None: return False
    fed_holidays = _get_holiday_calendar('nyse', dt.year)
    return dt in fed_holidays

@robust_date_handler
def is_holiday(date_input: Any) -> bool:
  """Check if date is a US Federal Reserve holiday"""
  dt = parse_date(date_input, return_format='date')
  if dt is None: return False
  sifma_holidays = _get_holiday_calendar('sifma', dt.year)
  return dt in sifma_holidays

@robust_date_handler
def get_next_fomc_meeting_date(reference_date: Any = None) -> Optional[datetime]:
    """Get next FOMC meeting date (approximate - 8 times per year)"""
    ref_dt = parse_date(reference_date or datetime.now(), config.default_tz)
    if ref_dt is None: return None

    # FOMC typically meets 8 times per year
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    year = ref_dt.year

    for month in fomc_months:
        if year == ref_dt.year and month < ref_dt.month: continue
        # Assume meetings are typically mid-month, around the 15th
        meeting_date = datetime(year, month, 15, tzinfo=ref_dt.tzinfo)
        if meeting_date > ref_dt:
            return meeting_date

    # If no meeting found this year, return January of next year
    return datetime(year + 1, 1, 15, tzinfo=ref_dt.tzinfo)


@robust_date_handler
def get_earnings_season_dates(year: int, quarter: int) -> List[datetime]:
    """Get approximate earnings season dates"""
    if quarter not in [1, 2, 3, 4]:
        raise ValueError("Quarter must be 1, 2, 3, or 4")

    quarter_end_month = 3 * quarter
    last_day_of_month = calendar.monthrange(year, quarter_end_month)[1]
    quarter_end = datetime(year, quarter_end_month, last_day_of_month, tzinfo=config.default_tz)

    season_start = add_business_days(quarter_end, 10)  # ~2 weeks after quarter end
    season_end = add_business_days(season_start, 30)  # ~6 weeks duration
    return [season_start, season_end]


@robust_date_handler
def validate_trade_date(trade_date: Any, instrument_type: InstrumentType, country: str = 'US') -> bool:
    """Validate if trade date is valid for specific instrument type"""
    trade_dt = parse_date(trade_date, config.default_tz)
    if trade_dt is None or not is_business_day(trade_dt): return False

    if instrument_type in [InstrumentType.GOVERNMENT_BOND, InstrumentType.CORPORATE_BOND, InstrumentType.CDS_SINGLE_NAME, InstrumentType.CDS_INDEX]:
        return not is_fed_holiday(trade_dt) if country.upper() == 'US' else True

    return True




@robust_date_handler
def get_week_of_year(date_input: Any) -> Optional[int]:
    """Get week of year for a date"""
    dt = parse_date(date_input, config.default_tz)
    return dt.isocalendar()[1] if dt else None


@robust_date_handler
def get_age(birth_date: Any, reference_date: Any = None) -> Optional[int]:
    """Calculate age in years"""
    birth_dt = parse_date(birth_date, config.default_tz)
    ref_dt = parse_date(reference_date or datetime.now(), config.default_tz)
    if birth_dt is None or ref_dt is None: return None

    return ref_dt.year - birth_dt.year - ((ref_dt.month, ref_dt.day) < (birth_dt.month, birth_dt.day))


@robust_date_handler
def is_weekend(date_input: Any) -> bool:
    """Check if date is a weekend"""
    dt = parse_date(date_input, config.default_tz)
    return dt.weekday() >= 5 if dt else False


@robust_date_handler
def format_date(date_input: Any, format_str: str = '%Y-%m-%d') -> Optional[str]:
    """Format date as string"""
    dt = parse_date(date_input, config.default_tz)
    return dt.strftime(format_str) if dt else None


class DateRange:
    """Date range with set operations"""

    def __init__(self, start_date: Any, end_date: Any):
        self.start_date = parse_date(start_date, config.default_tz)
        self.end_date = parse_date(end_date, config.default_tz)
        if self.start_date and self.end_date and self.start_date > self.end_date:
            self.start_date, self.end_date = self.end_date, self.start_date

    def __contains__(self, date_input: Any) -> bool:
        dt = parse_date(date_input, config.default_tz)
        if dt is None or self.start_date is None or self.end_date is None: return False
        return self.start_date <= dt <= self.end_date

    def intersection(self, other: 'DateRange') -> Optional['DateRange']:
        if not self.overlaps(other): return None
        start = max(self.start_date, other.start_date)
        end = min(self.end_date, other.end_date)
        return DateRange(start, end)

    def union(self, other: 'DateRange') -> 'DateRange':
        if self.start_date is None: return other
        if other.start_date is None: return self
        start = min(self.start_date, other.start_date)
        end = max(self.end_date, other.end_date)
        return DateRange(start, end)

    def difference(self, other: 'DateRange') -> List['DateRange']:
        if not self.overlaps(other): return [self] if self.start_date else []

        ranges = []
        # Left part
        if self.start_date < other.start_date:
            ranges.append(DateRange(self.start_date, other.start_date - timedelta(microseconds=1)))
        # Right part
        if self.end_date > other.end_date:
            ranges.append(DateRange(other.end_date + timedelta(microseconds=1), self.end_date))
        return ranges

    def overlaps(self, other: 'DateRange') -> bool:
        if self.start_date is None or self.end_date is None or other.start_date is None or other.end_date is None:
            return False
        return self.start_date <= other.end_date and self.end_date >= other.start_date

    def duration_days(self) -> int:
        if self.start_date is None or self.end_date is None: return 0
        return (self.end_date - self.start_date).days + 1

    def business_days(self, holiday_calendar: str = 'nyse') -> int:
        if self.start_date is None or self.end_date is None: return 0
        return get_business_days_between(self.start_date, self.end_date, holiday_calendar)


@robust_date_handler
def resample_dates(dates: List[Any], to_freq: str, method: str = 'last') -> List[datetime]:
    """Resample date series to different frequency"""
    if not dates: return []

    dt_list = [d for d in (parse_date(d, config.default_tz) for d in dates) if d is not None]
    if not dt_list: return []

    series = pd.Series(range(len(dt_list)), index=pd.DatetimeIndex(dt_list))
    resampled = series.resample(to_freq)

    if method == 'first':
        resampled = resampled.first()
    elif method == 'mean':
        resampled = resampled.mean()
    else:
        resampled = resampled.last()  # Default

    return [dt.to_pydatetime() for dt in resampled.index]


@robust_date_handler
def generate_date_sequence(start_date: Any,
                           end_date: Any,
                           frequency: str = 'D',
                           holiday_calendar: str = 'nyse') -> List[datetime]:
    """Generate a sequence of dates between start and end"""
    start_dt = parse_date(start_date, config.default_tz)
    end_dt = parse_date(end_date, config.default_tz)
    if start_dt is None or end_dt is None: return []

    if frequency.upper().startswith('B'):
        holidays_cal = _get_holiday_calendar(holiday_calendar, list(range(start_dt.year, end_dt.year + 1)))
        date_range = pd.bdate_range(start_dt, end_dt)
        return [dt.to_pydatetime() for dt in date_range if dt.date() not in holidays_cal]
    else:
        date_range = pd.date_range(start_dt, end_dt, freq=frequency)
        return [dt.to_pydatetime() for dt in date_range]


@robust_date_handler
def convert_timezone(date_input: Any, to_tz: Union[str, pytz.BaseTzInfo],
                     from_tz: Optional[Union[str, pytz.BaseTzInfo]] = None) -> Optional[datetime]:
    """Convert date from one timezone to another"""
    if from_tz:
        from_tz = pytz.timezone(from_tz) if isinstance(from_tz, str) else from_tz

    dt = parse_date(date_input, from_tz or config.default_tz)
    if dt is None: return None

    to_tz_obj = pytz.timezone(to_tz) if isinstance(to_tz, str) else to_tz
    return dt.astimezone(to_tz_obj)


@robust_date_handler
def get_quarter(date_input: Any) -> Optional[int]:
    """Get quarter for a date (1-4)"""
    dt = parse_date(date_input, config.default_tz)
    return (dt.month - 1) // 3 + 1 if dt else None


@robust_date_handler
def get_month_end(date_input: Any) -> Optional[datetime]:
    """Get last day of month for a date"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return dt.replace(day=last_day)


@robust_date_handler
def get_month_start(date_input: Any) -> Optional[datetime]:
    """Get first day of month for a date"""
    dt = parse_date(date_input, config.default_tz)
    return dt.replace(day=1) if dt else None


def set_default_timezone(tz: Union[str, pytz.BaseTzInfo]):
    """Set the default timezone for all operations"""
    config.set_timezone(tz)


def set_default_locale(locale_code: str):
    """Set default locale for date formatting"""
    config.set_locale(locale_code)


def set_strict_parsing(strict: bool):
    """Enable/disable strict parsing mode"""
    config.strict_parsing = strict


@robust_date_handler
def to_excel_date(date_input: Any) -> Optional[float]:
    """Convert date to Excel format (as a float)"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None
    delta = dt - EXCEL_EPOCH
    return float(delta.days) + float(delta.seconds) / 86400.0


@robust_date_handler
def from_excel_date(excel_date: Union[int, float]) -> Optional[datetime]:
    """Convert Excel date to datetime"""
    # Excel's epoch is naive, so the result is naive and needs to be localized
    naive_dt = datetime(1899, 12, 30) + timedelta(days=excel_date)
    return config.default_tz.localize(naive_dt)


@robust_date_handler
def to_kdb_days(date_input: Any) -> Optional[int]:
    """Convert date to KDB days since epoch"""
    dt = parse_date(date_input, config.default_tz)
    if dt is None: return None
    return (dt - KDB_EPOCH).days


@robust_date_handler
def from_kdb_days(kdb_date: int) -> Optional[datetime]:
    """Convert KDB date (days) to datetime"""
    return KDB_EPOCH + timedelta(days=kdb_date)


# --- Coupon & Payment Schedule Generation ---
@robust_date_handler
def generate_payment_schedule(
        start_date: Any,
        maturity_date: Any,
        frequency: str,
        convention: str = 'modified_following',
        holiday_calendar: str = 'sifma'
) -> List[datetime]:
  start_dt = parse_date(start_date)
  maturity_dt = parse_date(maturity_date)
  if not all([start_dt, maturity_dt]):
    return []

  freq_map = {
    'annual': relativedelta(months=12),
    'semi-annual': relativedelta(months=6),
    'quarterly': relativedelta(months=3),
    'monthly': relativedelta(months=1)
  }
  delta = freq_map.get(frequency.lower())
  if not delta:
    raise ValueError("Invalid frequency. Use 'annual', 'semi-annual', 'quarterly', or 'monthly'.")

  schedule = []
  current_date = maturity_dt
  while current_date > start_dt:
    adjusted_date = roll_date(current_date, rule=convention, holiday_calendar=holiday_calendar)
    schedule.append(adjusted_date)
    current_date -= delta

  schedule.reverse()
  return schedule


# --- Vectorized (Pandas & Polars) Operations ---
def add_business_days_series(
        date_series: Union[pd.Series, "pl.Series"],
        days: int,
        holiday_calendar: str = 'sifma'
) -> Union[pd.Series, "pl.Series"]:

  if isinstance(date_series, pd.Series):
    dt_series = pd.to_datetime(date_series, errors='coerce')
    min_year, max_year = dt_series.min().year, dt_series.max().year
    est_end_year = max_year + int(days / 252) + 2
    years = list(range(min_year, est_end_year))
    holidays_cal = _get_holiday_calendar(holiday_calendar, years)
    offset = pd.tseries.offsets.CustomBusinessDay(n=days, holidays=holidays_cal.keys())
    return dt_series + offset

  try:
    import polars as pl
    if isinstance(date_series, pl.Series):
      # Polars' native offset_by doesn't support custom holiday lists, so we use apply.
      return date_series.apply(lambda d: add_business_days(d, days, holiday_calendar))
  except ImportError:
    pass

  raise TypeError("Input must be a pandas or Polars Series.")


# --- Advanced Calendar and Holiday Logic ---
@robust_date_handler
def is_joint_business_day(date_input: Any, calendars: List[str]) -> bool:
  """
  Checks if a date is a business day across multiple calendars simultaneously.
  """
  if not calendars:
    return is_business_day(date_input, holiday_calendar='sifma')
  return all(is_business_day(date_input, holiday_calendar=cal) for cal in calendars)


@robust_date_handler
def add_joint_business_days(date_input: Any, days: int, calendars: List[str]) -> Optional[datetime]:
  """
  Adds business days, only counting days when all specified markets are open.
  """
  dt = parse_date(date_input, config.default_tz)
  if dt is None: return None

  current_date = dt
  step = timedelta(days=1) if days > 0 else timedelta(days=-1)
  days_to_move = abs(days)

  while days_to_move > 0:
    current_date += step
    if is_joint_business_day(current_date, calendars):
      days_to_move -= 1

  return current_date


# --- Tenor Parsing and Credit-Specific Helpers ---
@robust_date_handler
def parse_tenor(tenor_str: str) -> relativedelta:
  """
  Parses a financial tenor string like '3M', '6W', or '1Y6M' into a relativedelta object.
  """
  tenor_str = tenor_str.upper()
  total_delta = relativedelta()
  pattern = re.compile(r'(\d+)([YMWD])')
  matches = pattern.findall(tenor_str)

  if not matches:
    raise ValueError(f"Invalid tenor string format: {tenor_str}")

  for amount, unit in matches:
    num = int(amount)
    if unit == 'Y':
      total_delta += relativedelta(years=num)
    elif unit == 'M':
      total_delta += relativedelta(months=num)
    elif unit == 'W':
      total_delta += relativedelta(weeks=num)
    elif unit == 'D':
      total_delta += relativedelta(days=num)

  return total_delta


@robust_date_handler
def get_accrual_period(
        settlement_date: Any,
        maturity_date: Any,
        frequency: str,
        **kwargs
) -> Optional[Tuple[datetime, datetime]]:
  """
  Finds the coupon accrual period for a given settlement date.
  """
  settle_dt = parse_date(settlement_date)
  # Use a sensible start date guess to generate the schedule
  start_guess = settle_dt - parse_tenor(f"1{frequency[0].upper()}") * 2

  # Ensure the default calendar is SIFMA if not provided
  kwargs.setdefault('holiday_calendar', 'sifma')

  coupon_schedule = generate_payment_schedule(
    start_date=start_guess,
    maturity_date=maturity_date,
    frequency=frequency,
    **kwargs
  )
  if not coupon_schedule: return None

  idx = bisect.bisect_left(coupon_schedule, settle_dt)
  if idx == 0:
    # If settlement is before the first coupon, the period starts on the issue date.
    # This requires an issue_date parameter, for now we raise an error.
    raise ValueError("Settlement date is before the first calculated coupon payment.")

  period_end = coupon_schedule[idx]
  period_start = coupon_schedule[idx - 1]

  return period_start, period_end


@robust_date_handler
def get_cds_roll_dates(start_date: Any, end_date: Any) -> List[datetime]:
  """
  Generates standard CDS index roll dates (Mar 20, Sep 20) in a date range.
  """
  start_dt = parse_date(start_date)
  end_dt = parse_date(end_date)
  roll_dates = []

  for year in range(start_dt.year, end_dt.year + 2):
    for month in [3, 9]:
      candidate = roll_date(datetime(year, month, 20), "following", "sifma")
      if start_dt <= candidate <= end_dt:
        roll_dates.append(candidate)

  return sorted(roll_dates)


# --- Generic Utilities and Conversions ---
def get_fiscal_period(
        date_series: Union[pd.Series, "pl.Series"],
        fiscal_year_start_month: int = 1
) -> Union[pd.DataFrame, "pl.DataFrame"]:
  """
  Calculates the fiscal year and quarter for a Series of dates.
  """
  if isinstance(date_series, pd.Series):
    dt_series = pd.to_datetime(date_series, errors='coerce')
    offset = (fiscal_year_start_month - 1) * pd.tseries.offsets.MonthBegin()
    fiscal_year = (dt_series - offset).dt.year
    fiscal_quarter = ((dt_series.dt.month - fiscal_year_start_month + 12) % 12) // 3 + 1
    return pd.DataFrame({'fiscal_year': fiscal_year, 'fiscal_quarter': fiscal_quarter})

  if isinstance(date_series, pl.Series):
    year = date_series.dt.year()
    month = date_series.dt.month()
    fiscal_year = pl.when(month >= fiscal_year_start_month).then(year).otherwise(year - 1)
    adj_month = (month - fiscal_year_start_month + 12) % 12
    fiscal_quarter = (adj_month // 3 + 1)
    return pl.DataFrame({'fiscal_year': fiscal_year, 'fiscal_quarter': fiscal_quarter})

  raise TypeError("Input must be a pandas or Polars Series.")


def snap_to_interval(
        date_series: Union[pd.Series, "pl.Series"],
        interval: str = '1h',
        direction: str = 'down'
) -> Union[pd.Series, "pl.Series"]:
  """
  Snaps or rounds a datetime Series to a specified interval.
  """
  if isinstance(date_series, pd.Series):
    ts_series = pd.to_datetime(date_series, errors='coerce')
    if direction == 'down': return ts_series.dt.floor(interval)
    if direction == 'up': return ts_series.dt.ceil(interval)
    if direction == 'nearest': return ts_series.dt.round(interval)

    if isinstance(date_series, pl.Series):
      if direction == 'down': return date_series.dt.truncate(interval)
      if direction == 'up': return date_series.dt.truncate(interval, offset=f"-{interval}").dt.offset_by(interval)
      if direction == 'nearest': return date_series.dt.round(interval)

  raise TypeError("Input must be a pandas or Polars Series.")

import datetime as _dt
import re as _re
import dateparser as _dateparser

_MONTH_WORDS_RE = _re.compile(
    r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|"
    r"sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\b",
    _re.IGNORECASE,
)
_TIME_HINT_RE = _re.compile(
    r"(?ix)"
    r"("                         # any strong time indicator:
    r"\b(am|pm)\b"               # am/pm
    r"|:"                        # colon time 12:34
    r"|\bnoon\b|\bmidnight\b"    # common words
    r"|\b\d{1,2}\s*(?:h|hr|hrs)\b"  # 13h, 2hrs
    r")"
)
_DATEY_RE = _re.compile(
    r"(?ix)"
    r"("                                 # obvious date separators/patterns
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"   # 2026-02-26
    r"|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b" # 02/26/2026
    r"|[-/]\d{1,2}[-/]"                  # contains /mm/ or -mm-
    r")"
)


def parse_time_only(text: Union[str, datetime.time], *, base_dt: _dt.datetime | None = None) -> _dt.time:
    """
    Parse a time-like string into a datetime.time using dateparser.parse.

    Examples:
      "12:00"   -> 12:00:00
      "13"      -> 13:00:00  (i.e., 1pm in 12h terms)
      "1pm"     -> 13:00:00
      "noon"    -> 12:00:00

    Notes:
      - Rejects inputs that look like dates (e.g., "2026-02-26", "02/26/2026", month names).
      - Treats a bare hour like "13" or "7" as HH:00.
      - Returns a naive datetime.time (no tzinfo).
    """
    import datetime as dt
    if isinstance(text, dt.time): return text
    if isinstance(text, dt.datetime): return text.time()
    if text is None:
        raise ValueError("text must be a non-empty string")

    s = text.strip()
    if not s:
        raise ValueError("text must be a non-empty string")

    # Fast reject: obvious date-like strings.
    if _MONTH_WORDS_RE.search(s) or _DATEY_RE.search(s):
        raise ValueError(f"input looks like a date, not a time: {text!r}")

    # Ensure there's at least some reason to think it's a time.
    # Allow pure digits as "hour" as well.
    if not (_TIME_HINT_RE.search(s) or s.isdigit()):
        raise ValueError(f"input does not look like a time: {text!r}")

    # If it's just digits, interpret as an hour.
    # "13" => "13:00", "7" => "07:00"
    if s.isdigit():
        hour = int(s)
        if not (0 <= hour <= 24):
            raise ValueError(f"hour out of range in {text!r}")
        if hour == 24:
            return _dt.time(0, 0, 0)
        return _dt.time(hour, 0, 0)

    base = base_dt if base_dt is not None else _dt.datetime.now()

    dt = _dateparser.parse(
        s,
        settings={
            # Anchor parsing to a stable reference date to avoid weird rollovers.
            "RELATIVE_BASE": base,
            # Bias toward times when ambiguous (still can parse dates, hence our rejects above).
            "PREFER_DATES_FROM": "current_period",
            # Keep naive; we only return time.
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if dt is None:
        raise ValueError(f"could not parse time from {text!r}")

    # If dateparser interpreted a date-heavy meaning, guard against it.
    # (Heuristic: if input didn't strongly indicate time and date differs from base.)
    if not _TIME_HINT_RE.search(s) and dt.date() != base.date():
        raise ValueError(f"input parsed as date-like, not time-only: {text!r}")

    t = dt.time()

    # Normalize "24:00" edge (some parsers can yield 00:00 next day; dateparser usually won't,
    # but keep normalization trivial and safe).
    if t.hour == 0 and t.minute == 0 and t.second == 0:
        # If user typed "24:00" specifically, accept as midnight.
        if s.replace(" ", "") in ("24:00", "24:00:00"):
            return _dt.time(0, 0, 0)

    return t
