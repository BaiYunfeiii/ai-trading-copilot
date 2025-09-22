from __future__ import annotations

from pathlib import Path

from src.MT5DataProvider import get_rates
import logging


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("fetch_mt5_samples")

    symbols = ["XAUUSDm", "BTCUSDm", "ETHUSDm"]
    timeframe_to_count = {"D1": 100, "H1": 300, "M5": 300}
    timeframes = ["D1", "H1", "M5"]

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始抓取行情样本 | symbols=%s | timeframes=%s | timeframe_to_count=%s", symbols, timeframes, timeframe_to_count)

    for symbol in symbols:
        for tf in timeframes:
            count = timeframe_to_count[tf]
            logger.info("获取数据 | symbol=%s | timeframe=%s | count=%d", symbol, tf, count)
            df = get_rates(symbol=symbol, timeframe=tf, count=count)
            out_path = output_dir / f"{symbol}_{tf}_{count}.csv"
            df.to_csv(out_path, index=True)
            logger.info("已保存: %s", out_path)


if __name__ == "__main__":
    main()


