from __future__ import annotations

from pathlib import Path
import os
import json
import argparse
from dotenv import load_dotenv
from typing import List

import pandas as pd
import logging
from datetime import datetime

from src import OpenAIClient, OpenAIConfig

SYSTEM_PROMPT = """
你是一位精通Al Brooks价格行为学体系的专业交易分析师。你的任务是分析所提供时间框架的K线数据，首先识别当前市场周期，然后基于该周期特征制定相应的交易计划。

核心分析框架：

一、市场周期识别（按Al Brooks体系）
1. Strong Breakout（强突破）
   - 连续的趋势棒，很少回调
   - 突破后快速远离突破点
   
2. Breakout（突破）
   - 突破重要支撑/阻力
   - 需要评估突破的强度和后续跟进

3. Always In Long/Short（始终做多/做空）
   - 明确的单边趋势
   - 回调浅且被快速买入/卖出
   
4. Broad Channel（宽通道）
   - 趋势存在但有较深回调
   - 通道线清晰可见
   
5. Tight Channel（窄通道）
   - 缓慢稳定的趋势
   - 小幅波动，很少超出通道线
   
6. Trading Range（交易区间）
   - 明确的支撑和阻力
   - 价格在区间内震荡
   
7. Breakout Mode（突破模式）
   - 区间收窄，即将选择方向
   - 需要等待确认

二、价格行为模式识别
- Major Reversal（主要反转）
- Minor Reversal（次要反转）
- Failed Breakout（假突破）
- Measured Move（等距运动）
- Wedge/Triangle（楔形/三角形）
- Double Top/Bottom（双顶/双底）
- High/Low 1,2,3,4（高低点序列）

三、Bar-by-Bar分析要点
- Signal Bar质量评估
- Entry Bar确认
- Follow Through评估
- 趋势棒vs区间棒识别

四、交易计划制定原则
根据识别的市场周期，采用相应策略：
- Always In: 只做趋势方向，回调入场
- Channel: 通道边界交易，注意假突破
- Trading Range: 区间顶底交易，中间区域不交易
- Breakout: 评估突破强度，决定追入还是等回调

输出格式要求：
1. 先明确当前市场周期及判断依据
2. 基于周期特征制定具体交易策略
3. 标注关键价格水平和结构
4. 给出明确的交易设置和管理方案
"""


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


def parse_symbol_args(symbol_arg: str) -> tuple[str, str, int]:
    """
    解析 --symbol 参数，支持以下格式：
    - SYMBOL
    - SYMBOL,TIMEFRAME
    - SYMBOL,TIMEFRAME,ROWS

    示例：
    - XAUUSDm
    - XAUUSDm,M5
    - XAUUSDm,M5,300
    """
    parts = [p.strip() for p in symbol_arg.split(',') if p.strip()]
    if len(parts) == 0 or len(parts) > 3:
        raise ValueError(
            f"参数格式错误: {symbol_arg}，应为 SYMBOL 或 SYMBOL,TIMEFRAME 或 SYMBOL,TIMEFRAME,ROWS"
        )

    symbol = parts[0]
    timeframe = "M5" if len(parts) < 2 else parts[1]
    if len(parts) < 3:
        rows = 300
    else:
        try:
            rows = int(parts[2])
        except ValueError:
            raise ValueError(f"行数必须是整数: {parts[2]}")

    return symbol, timeframe, rows


def make_prompt_merged(symbol: str, tf_to_df: dict[str, pd.DataFrame]) -> str:
    sections = []
    for tf, df in tf_to_df.items():
        sections.append(f"[周期 {tf}]\n" + _summarize_df(df))
    joined = "\n\n".join(sections)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
    f"""
请使用Al Brooks价格行为学方法分析以下[{symbol}]的数据：

{joined}

分析要求：

第一部分：市场周期诊断
1. 当前市场处于什么周期？（Breakout/Always In/Channel/Trading Range等）
2. 判断依据是什么？（具体指出关键的价格行为特征）
3. 这个周期已经持续了多久？是否有转换迹象？

第二部分：关键结构识别
1. 标注最重要的3-5个支撑/阻力位
2. 识别最近的Swing High/Low
3. 是否存在明显的趋势线或通道线？
4. 有无重要的价格行为模式？（如wedge, triangle, measured move等）

第三部分：当前局势评估
1. 最近20根K线的价格行为总结
2. 当前是否有"Always In"方向？
3. 买方和卖方的相对强弱
4. 是否处于"Breakout Mode"（即将突破）？

第四部分：交易计划
基于识别的市场周期，提供：
1. 适合当前周期的交易策略
2. 具体入场设置（包括Signal Bar特征）
3. 止损位置（基于市场结构）
4. 第一目标和第二目标
5. 交易管理（何时移动止损、部分止盈）

第五部分：风险提示
1. 当前设置的概率评估（高概率60%+，中等40-60%，低概率<40%）
2. 可能导致交易失败的情形
3. 备选方案（如果当前设置失效）

补充信息：
- 当前时间：[{time}]
- 风险管理：单笔风险不超过2%
    """
    )


def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="从CSV数据生成交易计划")
    parser.add_argument(
        "--symbol", 
        type=str, 
        help=(
            "交易对参数，格式: SYMBOL[,TIMEFRAME[,ROWS]] "
            "例如: XAUUSDm 或 XAUUSDm,M5 或 XAUUSDm,M5,300"
        )
    )
    args = parser.parse_args()

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("plan_from_csv")

    # 加载 .env（若存在），便于本地开发与部署环境变量管理
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        # 使用 override=True 以覆盖系统中已存在但为空/旧值的环境变量
        load_dotenv(dotenv_path=env_path, override=True)
        logging.info("已加载环境变量文件: %s", env_path)
    else:
        logging.info("未检测到 .env 文件，使用系统环境变量")

    # 配置 - 从命令行参数或默认值获取
    if args.symbol:
        try:
            symbol, timeframe, rows = parse_symbol_args(args.symbol)
            symbols: List[str] = [symbol]
            timeframes: List[str] = [timeframe]
        except ValueError as e:
            logger.error("参数解析错误: %s", e)
            return
    else:
        # 默认配置
        symbols: List[str] = ["BTCUSDm"]
        timeframes: List[str] = ["M5"]
        rows = 300

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    plans_dir = output_dir / "plans"
    conv_dir = output_dir / "conversations"
    plans_dir.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)

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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        content = client.chat(messages)
        # 将助手回复加入对话
        if isinstance(content, str):
            messages.append({"role": "assistant", "content": content})
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

        # 保存对话 JSON，便于复用上下文
        conv_payload = {
            "symbol": symbol,
            "timeframes": timeframes,
            "rows": rows,
            "generated_at": ts,
            "messages": messages,
        }
        conv_path = conv_dir / f"conv_{symbol}_{ts}.json"
        conv_path.write_text(json.dumps(conv_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("已保存对话: %s", conv_path)


if __name__ == "__main__":
    main()


