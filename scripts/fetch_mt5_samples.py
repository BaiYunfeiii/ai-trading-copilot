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
    count = 100
    timeframes = ["D1", "H1", "M5"]

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始抓取行情样本 | symbols=%s | timeframes=%s | count=%d", symbols, timeframes, count)

    for symbol in symbols:
        for tf in timeframes:
            logger.info("获取数据 | symbol=%s | timeframe=%s | count=%d", symbol, tf, count)
            df = get_rates(symbol=symbol, timeframe=tf, count=count)
            out_path = output_dir / f"{symbol}_{tf}_{count}.csv"
            df.to_csv(out_path, index=True)
            logger.info("已保存: %s", out_path)


if __name__ == "__main__":
    main()


