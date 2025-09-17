#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

from src.MT5DataProvider import get_positions, MT5Config, MT5DataProvider
from src.OpenAIClient import OpenAIClient, OpenAIConfig


def load_conversation(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def positions_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "当前无持仓。"
    # 仅展示关键字段，避免过长
    show_cols = [
        c for c in [
            "ticket", "symbol", "type", "volume", "price_open", "price_current",
            "sl", "tp", "profit", "time_update"
        ] if c in df.columns
    ]
    df_show = df[show_cols].copy()
    # type 映射：0=BUY, 1=SELL（MT5定义）
    if "type" in df_show.columns:
        type_map = {0: "BUY", 1: "SELL"}
        df_show["type"] = df_show["type"].map(type_map).fillna(df_show["type"])  # type: ignore[index]
    # 转字符串，便于插入
    return df_show.to_markdown(index=False)


def build_messages(original_messages: List[Dict[str, str]], positions_md: str) -> List[Dict[str, str]]:
    # 将持仓信息追加为新的 user 消息，提示模型给出仓位管理建议
    appended = list(original_messages)
    appended.append({
        "role": "user",
        "content": (
            "以下是当前账户的持仓（来自MT5）：\n\n" + positions_md +
            "\n\n请结合先前的分析与计划，给出仓位管理与风险控制建议：\n"
            "- 是否应加减仓、在何处；\n"
            "- 止损/止盈的优化与移动；\n"
            "- 若无持仓，应说明是否建议建立新仓，以及触发条件。\n"
        )
    })
    return appended


def derive_conversation_from_plan(plan_path: Path) -> Path:
    """
    根据计划文件名反推会话文件名：
    计划: plan_{SYMBOL}_{YYYYMMDD_HHMMSS}.md
    会话: conv_{SYMBOL}_{YYYYMMDD_HHMMSS}.json
    优先在 plan 所在目录的同级 conversations 目录中查找。
    """
    name = plan_path.name
    # 基于简单模式解析
    if not name.startswith("plan_") or not name.endswith(".md"):
        raise ValueError(f"无法从计划文件名解析：{name}")
    stem = name[len("plan_"):-len(".md")]  # SYMBOL_YYYYMMDD_HHMMSS
    parts = stem.rsplit("_", 2)
    if len(parts) < 2:
        raise ValueError(f"计划文件名格式不正确：{name}")
    # 兼容 SYMBOL 中含有下划线的情况，将最后两个片段合并成时间戳
    ts = parts[-2] + "_" + parts[-1]
    symbol = stem[: -(len(ts) + 1)]
    conv_name = f"conv_{symbol}_{ts}.json"
    # 优先同级 conversations
    if plan_path.parent.name == "plans":
        base_dir = plan_path.parent.parent
        conv_dir = base_dir / "conversations"
        candidate = conv_dir / conv_name
        if candidate.exists():
            return candidate
        # 退回到同目录
        candidate2 = plan_path.parent / conv_name
        if candidate2.exists():
            return candidate2
        return candidate  # 返回默认期望路径
    # 通用回退：同目录
    return plan_path.parent / conv_name


def derive_symbol_from_conversation_path(conv_path: Path) -> Optional[str]:
    """从会话文件名 conv_{SYMBOL}_{YYYYMMDD_HHMMSS}.json 推导 symbol。"""
    name = conv_path.name
    if not name.startswith("conv_") or not name.endswith(".json"):
        return None
    stem = name[len("conv_"):-len(".json")]  # SYMBOL_YYYYMMDD_HHMMSS
    parts = stem.rsplit("_", 2)
    if len(parts) < 2:
        return None
    ts = parts[-2] + "_" + parts[-1]
    symbol = stem[: -(len(ts) + 1)]
    return symbol or None


def main() -> None:
    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("positions_advisor")

    # 加载 .env（若存在）
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info("已加载环境变量文件: %s", env_path)
    else:
        logger.info("未检测到 .env 文件，使用系统环境变量")

    parser = argparse.ArgumentParser(description="附加MT5持仓并请求OpenAI给出仓位管理建议")
    parser.add_argument("--conversation", type=str, nargs="?", default=None, help="会话JSON路径，如 output/conversations/conv_xxx.json")
    parser.add_argument("--plan", type=str, default=None, help="可选：从计划文件名推导会话文件，如 output/plans/plan_xxx.md")
    parser.add_argument("--symbol", type=str, default=None, help="可选：仅提取该品种持仓；若缺省将尝试自动推导")
    parser.add_argument("--terminal", type=str, default=None, help="可选：MT5终端路径")
    parser.add_argument("--output", type=str, default=None, help="可选：保存包含建议与持仓的JSON到该路径（不含会话）")
    parser.add_argument("--md_output", type=str, default=True, help="可选：将建议与持仓另存为Markdown文件（不含会话）")
    args = parser.parse_args()

    conv_path: Optional[Path] = None
    if args.conversation:
        conv_path = Path(args.conversation)
        logger.info("会话来源：命令行 conversation，路径=%s", conv_path)
    elif args.plan:
        conv_path = derive_conversation_from_plan(Path(args.plan))
        logger.info("会话来源：由计划文件推导，plan=%s -> conv=%s", args.plan, conv_path)
    else:
        raise SystemExit("请提供会话文件路径或 --plan 计划文件路径。")

    if not conv_path.exists():
        raise FileNotFoundError(f"会话文件不存在: {conv_path}")

    conv = load_conversation(conv_path)
    messages: List[Dict[str, str]] = conv.get("messages", [])  # type: ignore[assignment]

    # 决定 symbol：优先 --symbol，其次会话JSON中的 symbol，再其次从文件名推导
    effective_symbol: Optional[str] = args.symbol or conv.get("symbol") or derive_symbol_from_conversation_path(conv_path)
    if effective_symbol:
        effective_symbol = effective_symbol.lower()
    logger.info("解析到 symbol=%s (统一为小写)", effective_symbol or "<未指定>")

    # 读取持仓
    if args.terminal:
        logger.info("使用指定 MT5 终端路径：%s", args.terminal)
    positions_df = get_positions(symbol=effective_symbol, terminal_path=args.terminal)
    positions_count = 0 if positions_df is None else len(positions_df.index)
    logger.info("已读取持仓 %d 条%s", positions_count, f"（过滤 symbol={effective_symbol}）" if effective_symbol else "")
    positions_md = positions_to_markdown(positions_df)

    # 组装消息并调用OpenAI
    final_messages = build_messages(messages, positions_md)

    # 从环境变量读取 OpenAI 兼容接口配置（避免打印敏感信息）
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("OPENAI_MODEL", OpenAIConfig().model)
    logger.info("OpenAI 配置：base_url=%s, api_key_present=%s, model=%s", base_url, api_key_present, model_name)

    oa_conf = OpenAIConfig(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
    )
    client = OpenAIClient(oa_conf)
    logger.info("开始请求 OpenAI 建议……")
    advice = client.chat(final_messages, max_tokens=1200)
    logger.info("OpenAI 建议获取完成，长度=%d 字符", len(advice) if isinstance(advice, str) else -1)

    # 输出到控制台
    print("==== 仓位管理建议 ====")
    print(advice)

    # 可选保存为 JSON（不包含会话全文，仅保存 positions 与 advice）
    if args.output:
        out_path = Path(args.output)
        out_payload = {
            "symbol": effective_symbol,
            "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "positions": positions_df.to_dict(orient="records"),
            "advice": advice,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        logger.info("已保存JSON至：%s", out_path)
        print(f"已保存建议至: {out_path}")

    # 可选保存为 Markdown（不包含会话全文）
    if args.md_output:
        md_path = Path(args.md_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 组装 Markdown 内容
        lines = []
        lines.append(f"# 仓位管理建议 ({effective_symbol or 'n/a'})")
        lines.append("")
        lines.append(f"- 生成时间: {ts}")
        if effective_symbol:
            lines.append(f"- 品种: `{effective_symbol}`")
        lines.append("")
        lines.append("## 当前持仓")
        lines.append("")
        lines.append(positions_md)
        lines.append("")
        lines.append("## 建议")
        lines.append("")
        lines.append(advice if isinstance(advice, str) else str(advice))
        content = "\n".join(lines)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(content)
        logger.info("已保存Markdown至：%s", md_path)
        print(f"已保存Markdown至: {md_path}")


if __name__ == "__main__":
    main()
