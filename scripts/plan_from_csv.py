from __future__ import annotations

from pathlib import Path
import os
from dotenv import load_dotenv
from typing import List

import pandas as pd
import logging

from src import OpenAIClient, OpenAIConfig


def load_recent_rows(csv_path: Path, n: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    if n > 0:
        df = df.tail(n)
    return df


def _summarize_df(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=True)
    return csv_text


def make_prompt_merged(symbol: str, tf_to_df: dict[str, pd.DataFrame]) -> str:
    sections = []
    for tf, df in tf_to_df.items():
        sections.append(f"[周期 {tf}]\n" + _summarize_df(df))
    joined = "\n\n".join(sections)
    return (
    f"""
我是一名“Albrooks 价格行为学”的交易者，偏好按照H1读取Context，M5交易。  
以下是 {symbol} H1和M5的k线数据。  
- 请分析当前的Context和可以采取的交易计划.
- 请给出行动清单

注：
1. 最新的K线仍在进行中，包括这次和后续的数据
2. 如有多个交易计划，请给出优先级。

数据如下：
{joined}
    """
    )


def main() -> None:
    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("plan_from_csv")

    # 加载 .env（若存在），便于本地开发与部署环境变量管理
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logging.info("已加载环境变量文件: %s", env_path)
    else:
        logging.info("未检测到 .env 文件，使用系统环境变量")

    # 配置
    symbols: List[str] = ["XAUUSDm"]
    timeframes: List[str] = ["H1", "M5"]
    rows = 100

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    plans_dir = output_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    # 从环境变量读取 OpenAI 兼容接口配置，避免硬编码敏感信息
    client = OpenAIClient(OpenAIConfig(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    ))

    logger.info("启动交易计划生成 | symbols=%s | timeframes=%s | rows=%d", symbols, timeframes, rows)

    for symbol in symbols:
        tf_to_df: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            csv_path = output_dir / f"{symbol}_{tf}_{rows}.csv"
            try:
                logger.info("读取CSV: %s", csv_path)
                df = load_recent_rows(csv_path, rows)
            except FileNotFoundError:
                # 若恰好文件名中的行数不同，尝试模糊匹配最新文件
                logger.warning("未找到精确CSV，尝试回退匹配: %s", csv_path.name)
                candidates = sorted(output_dir.glob(f"{symbol}_{tf}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    logging.error("缺少CSV: %s-%s，请先运行 scripts/fetch_mt5_samples.py", symbol, tf)
                    continue
                csv_path = candidates[0]
                logger.info("使用最新候选CSV: %s", csv_path)
                df = load_recent_rows(csv_path, rows)
            tf_to_df[tf] = df

        if not tf_to_df:
            logger.warning("跳过 %s：无可用数据帧", symbol)
            continue

        prompt = make_prompt_merged(symbol, tf_to_df)
        logger.info("构建提示完成 | symbol=%s | 提示字符数=%d", symbol, len(prompt))
        content = client.chat([
            {"role": "system", "content": "你是专业的量化交易顾问，严格按照Al Brooks价格行为学的逻辑进行交易。"},
            {"role": "user", "content": prompt},
        ])
        logger.info("收到模型响应 | symbol=%s | 响应字符数=%d", symbol, len(content) if isinstance(content, str) else -1)
        print("=" * 80)
        print(f"Unified Plan for {symbol} (timeframes: {', '.join(timeframes)}):")
        print(content)

        # 保存为 Markdown 文件
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = plans_dir / f"plan_{symbol}_{ts}.md"
        header = (
            f"# Trading Plan\n\n"
            f"- symbol: {symbol}\n"
            f"- timeframes: {', '.join(timeframes)}\n"
            f"- rows: {rows}\n"
            f"- generated_at: {ts}\n\n"
        )
        body = content if isinstance(content, str) else ""
        md_path.write_text(header + body, encoding="utf-8")
        logger.info("已保存计划: %s", md_path)


if __name__ == "__main__":
    main()


