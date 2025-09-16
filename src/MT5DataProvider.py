from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Dict

import pandas as pd
import pytz

try:
    import MetaTrader5 as mt5
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "无法导入 MetaTrader5 模块。请先安装: pip install MetaTrader5，并确保已安装并登录本地MT5终端。"
    ) from exc


TimeInput = Union[str, datetime]

UTC = pytz.UTC
TZ_SH = pytz.timezone("Asia/Shanghai")


def _parse_time(value: TimeInput) -> datetime:
    if isinstance(value, datetime):
        # 若无时区，视为上海时间
        if value.tzinfo is None:
            value = TZ_SH.localize(value)
        return value
    # 尝试多种常见格式
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
            # 视为上海时间
            return TZ_SH.localize(parsed)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {value}")


def _build_timeframe_map() -> Dict[str, int]:
    # 参考 MT5 可用周期；
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


TIMEFRAME_MAP = _build_timeframe_map()


@dataclass
class MT5Config:
    terminal_path: Optional[str] = None  # 指定本地MT5终端路径（可选）
    timeout: int = 30  # 秒


class MT5DataProvider:
    def __init__(self, config: Optional[MT5Config] = None) -> None:
        self.config = config or MT5Config()
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        kwargs = {}
        if self.config.terminal_path:
            kwargs["path"] = self.config.terminal_path
        # 建议设置超时，避免卡住
        success = mt5.initialize(timeout=self.config.timeout, **kwargs)
        if not success:
            last_error = mt5.last_error()
            raise RuntimeError(f"MT5初始化失败: {last_error}")
        self._initialized = True

    def shutdown(self) -> None:
        if self._initialized:
            mt5.shutdown()
            self._initialized = False

    def __enter__(self) -> "MT5DataProvider":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    @staticmethod
    def _resolve_timeframe(timeframe: str) -> int:
        key = timeframe.upper()
        if key not in TIMEFRAME_MAP:
            supported = ", ".join(TIMEFRAME_MAP.keys())
            raise ValueError(f"不支持的时间周期: {timeframe}；可选: {supported}")
        return TIMEFRAME_MAP[key]

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        count: Optional[int] = None,
        start_time: Optional[TimeInput] = None,
        end_time: Optional[TimeInput] = None,
    ) -> pd.DataFrame:
        """
        抓取MT5行情K线数据。

        参数：
        - symbol: 品种，例如 "EURUSD"、"XAUUSD" 等。
        - timeframe: 周期字符串，如 "M1"、"M5"、"H1"、"H4"、"D1" 等。
        - count: K线数量（优先于时间范围）。
        - start_time: 开始时间（包含）。
        - end_time: 结束时间（不包含），可选；缺省为“最新”。

        规则：k线数量 与 开始/结束时间 二选一，k线数量优先。

        返回：pandas DataFrame，时间索引，包含 open/high/low/close, tick_volume, real_volume, spread。
        """
        self.initialize()

        tf = self._resolve_timeframe(timeframe)

        if count is not None and count > 0:
            # 以当前UTC为结束点
            now_utc = datetime.now(tz=UTC)
            rates = mt5.copy_rates_from(symbol, tf, now_utc, count)
        else:
            if start_time is None:
                raise ValueError("当未指定k线数量时，必须提供开始时间。")
            start_dt_local = _parse_time(start_time)
            end_dt_local = datetime.now(tz=TZ_SH) if end_time is None else _parse_time(end_time)
            # 转成UTC给MT5
            start_dt = start_dt_local.astimezone(UTC)
            end_dt = end_dt_local.astimezone(UTC)
            if end_dt <= start_dt:
                raise ValueError("结束时间必须大于开始时间。")
            rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            last_error = mt5.last_error()
            raise RuntimeError(f"未获取到数据或MT5返回空。last_error={last_error}")

        df = pd.DataFrame(rates)
        if "time" in df.columns:
            # MT5返回的时间为UTC秒，转换为UTC带tz，再转上海时区
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ_SH)
            df.set_index("time", inplace=True)

        # 确保列顺序友好
        ordered_cols = [c for c in [
            "open", "high", "low", "close"
        ] if c in df.columns]
        df = df[ordered_cols]

        return df


# 便捷函数式接口
_default_provider: Optional[MT5DataProvider] = None


def get_rates(
    symbol: str,
    timeframe: str,
    count: Optional[int] = None,
    start_time: Optional[TimeInput] = None,
    end_time: Optional[TimeInput] = None,
    terminal_path: Optional[str] = None,
) -> pd.DataFrame:
    global _default_provider
    if _default_provider is None:
        _default_provider = MT5DataProvider(MT5Config(terminal_path=terminal_path))
    with _default_provider:
        return _default_provider.fetch(
            symbol=symbol,
            timeframe=timeframe,
            count=count,
            start_time=start_time,
            end_time=end_time,
        )


