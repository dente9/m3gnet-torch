#!/usr/bin/env python3
"""
M3GNet 项目测试脚本
自动执行所有核心模块的单元测试
"""
import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    # 获取当前脚本所在目录 (m3gnet 目录)
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent  # 项目根目录
    
    print(f"🔍 当前工作目录: {current_dir}")
    print(f"📁 项目根目录: {project_root}")
    
    # 定义要运行的测试列表
    tests = [
        "m3gnet.graph.tests.test_graph_and_batch",
        "m3gnet.layers.tests.test_layers",
        "m3gnet.models.tests.test_models",
        "m3gnet.train.tests.test_trainer"
    ]
    
    # 切换到项目根目录
    os.chdir(project_root)
    print(f"\n🚀 切换到项目根目录: {project_root}")
    
    # 运行所有测试
    results = {}
    for test in tests:
        print(f"\n🔧 开始测试: {test}")
        try:
            # 使用 -v 参数获取详细输出
            result = subprocess.run(
                [sys.executable, "-m", test, "-v"],
                check=True,
                capture_output=True,
                text=True
            )
            results[test] = "✅ 通过"
            print(f"🟢 测试通过: {test}")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            results[test] = "❌ 失败"
            print(f"🔴 测试失败: {test}")
            print(e.stdout)
            print(e.stderr)
    
    # 打印汇总结果
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    for test, status in results.items():
        print(f"{test}: {status}")
    print("="*50)
    
    # 检查是否有失败的测试
    if any(status == "❌ 失败" for status in results.values()):
        print("\n❌ 部分测试失败，请检查日志")
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过！")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()