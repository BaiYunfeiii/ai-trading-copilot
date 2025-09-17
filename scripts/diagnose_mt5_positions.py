#!/usr/bin/env python3
"""
MT5 持仓诊断脚本
用于诊断 get_positions 方法无法返回持仓数据的问题
"""

import logging
import sys
from typing import Optional
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, '.')

from src.MT5DataProvider import MT5DataProvider, MT5Config
import MetaTrader5 as mt5

def diagnose_mt5_connection(terminal_path: Optional[str] = None) -> bool:
    """诊断MT5连接状态"""
    print("=== MT5 连接诊断 ===")
    
    # 1. 检查MT5模块导入
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 模块导入成功")
    except ImportError as e:
        print(f"✗ MetaTrader5 模块导入失败: {e}")
        return False
    
    # 2. 尝试初始化MT5
    print(f"尝试初始化MT5 (终端路径: {terminal_path or '默认'})")
    kwargs = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    
    success = mt5.initialize(timeout=30, **kwargs)
    if not success:
        last_error = mt5.last_error()
        print(f"✗ MT5初始化失败: {last_error}")
        print("可能的原因:")
        print("  - MT5终端未安装或未运行")
        print("  - 终端路径不正确")
        print("  - 账户未登录")
        print("  - 权限不足")
        return False
    else:
        print("✓ MT5初始化成功")
    
    # 3. 检查账户信息
    account_info = mt5.account_info()
    if account_info is None:
        print("✗ 无法获取账户信息")
        mt5.shutdown()
        return False
    else:
        print(f"✓ 账户信息获取成功:")
        print(f"  - 账户: {account_info.login}")
        print(f"  - 服务器: {account_info.server}")
        print(f"  - 余额: {account_info.balance}")
        print(f"  - 权益: {account_info.equity}")
        print(f"  - 保证金: {account_info.margin}")
    
    # 4. 检查交易状态
    trade_allowed = mt5.terminal_info().trade_allowed
    print(f"✓ 交易允许状态: {trade_allowed}")
    
    mt5.shutdown()
    return True

def diagnose_positions_data(terminal_path: Optional[str] = None, symbol: Optional[str] = None) -> None:
    """诊断持仓数据获取"""
    print("\n=== 持仓数据诊断 ===")
    
    # 初始化MT5
    kwargs = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    
    success = mt5.initialize(timeout=30, **kwargs)
    if not success:
        print("✗ MT5初始化失败，无法继续诊断")
        return
    
    try:
        # 1. 直接调用MT5 API
        print(f"直接调用 mt5.positions_get(symbol={symbol})")
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        
        print(f"MT5返回结果类型: {type(positions)}")
        print(f"MT5返回结果: {positions}")
        
        if positions is None:
            last_error = mt5.last_error()
            print(f"✗ positions_get 返回 None，错误代码: {last_error}")
            print("错误代码含义:")
            if last_error[0] == -10004:
                print("  -10004: 内部IPC连接失败")
            elif last_error[0] == -10005:
                print("  -10005: 内部超时")
            elif last_error[0] == -8:
                print("  -8: 自动交易被禁用")
            elif last_error[0] == -6:
                print("  -6: 授权失败")
            else:
                print(f"  {last_error[0]}: {last_error[1]}")
        elif len(positions) == 0:
            print("✓ positions_get 返回空列表 - 当前无持仓")
        else:
            print(f"✓ positions_get 返回 {len(positions)} 个持仓")
            for i, pos in enumerate(positions[:3]):  # 只显示前3个
                print(f"  持仓 {i+1}: {pos}")
        
        # 2. 测试我们的封装方法
        print(f"\n测试 MT5DataProvider.get_positions(symbol={symbol})")
        provider = MT5DataProvider(MT5Config(terminal_path=terminal_path))
        try:
            df = provider.get_positions(symbol=symbol)
            print(f"✓ 封装方法返回 DataFrame: {df.shape}")
            if len(df) > 0:
                print("前几行数据:")
                print(df.head().to_string())
            else:
                print("DataFrame为空")
        except Exception as e:
            print(f"✗ 封装方法失败: {e}")
        
    finally:
        mt5.shutdown()

def check_symbol_info(terminal_path: Optional[str] = None, symbol: Optional[str] = None) -> None:
    """检查品种信息"""
    if not symbol:
        return
        
    print(f"\n=== 品种信息检查 ({symbol}) ===")
    
    kwargs = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    
    success = mt5.initialize(timeout=30, **kwargs)
    if not success:
        print("✗ MT5初始化失败")
        return
    
    try:
        # 获取品种信息
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"✗ 无法获取品种 {symbol} 的信息")
            print("可能的原因:")
            print("  - 品种名称不正确")
            print("  - 品种不在当前账户的交易列表中")
            print("  - 品种已停用")
        else:
            print(f"✓ 品种 {symbol} 信息:")
            print(f"  - 名称: {symbol_info.name}")
            print(f"  - 描述: {symbol_info.description}")
            print(f"  - 可见: {symbol_info.visible}")
            print(f"  - 交易模式: {symbol_info.trade_mode}")
            print(f"  - 交易停止: {symbol_info.trade_stops_level}")
        
        # 获取所有可用品种
        symbols = mt5.symbols_get()
        if symbols:
            symbol_names = [s.name for s in symbols]
            print(f"\n可用品种数量: {len(symbol_names)}")
            if symbol and symbol in symbol_names:
                print(f"✓ {symbol} 在可用品种列表中")
            elif symbol:
                print(f"✗ {symbol} 不在可用品种列表中")
                # 查找相似的品种名
                similar = [s for s in symbol_names if symbol.upper() in s.upper()]
                if similar:
                    print(f"相似的品种: {similar[:5]}")
    finally:
        mt5.shutdown()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MT5持仓诊断工具")
    parser.add_argument("--terminal", help="MT5终端路径")
    parser.add_argument("--symbol", help="要检查的品种名称")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print("MT5 持仓诊断工具")
    print("=" * 50)
    
    # 1. 连接诊断
    connection_ok = diagnose_mt5_connection(args.terminal)
    if not connection_ok:
        print("\n连接诊断失败，请检查MT5安装和配置")
        return
    
    # 2. 品种信息检查
    if args.symbol:
        check_symbol_info(args.terminal, args.symbol)
    
    # 3. 持仓数据诊断
    diagnose_positions_data(args.terminal, args.symbol)
    
    print("\n诊断完成")

if __name__ == "__main__":
    main()
