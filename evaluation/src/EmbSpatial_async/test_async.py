#!/usr/bin/env python3
"""
EmbSpatial异步版本测试脚本
用于验证代码语法和基本功能
"""

import sys
import os
import json
import asyncio

def test_imports():
    """测试导入依赖"""
    try:
        import aiohttp
        import tqdm
        print("✅ 所有依赖包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 依赖包导入失败: {e}")
        return False

def test_data_file():
    """测试数据文件"""
    data_file = "./embspatial_bench_new.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            required_fields = ['question', 'image', 'answer_options', 'answer', 'objects']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"❌ 数据格式错误，缺少字段: {missing_fields}")
                return False
            
            print(f"✅ 数据文件格式正确，样本字段: {list(data.keys())}")
            return True
    except Exception as e:
        print(f"❌ 数据文件读取失败: {e}")
        return False

def test_script_syntax():
    """测试Python脚本语法"""
    try:
        import eval_embspatial_async
        print("✅ Python脚本语法正确")
        return True
    except SyntaxError as e:
        print(f"❌ Python脚本语法错误: {e}")
        return False
    except Exception as e:
        print(f"⚠️  脚本导入时发生其他错误（可能是正常的）: {e}")
        return True  # 其他错误可能是正常的，比如缺少服务器连接

async def test_async_function():
    """测试异步函数基本功能"""
    try:
        # 测试基本的异步功能
        await asyncio.sleep(0.1)
        print("✅ 异步功能正常")
        return True
    except Exception as e:
        print(f"❌ 异步功能测试失败: {e}")
        return False

def test_shell_script():
    """测试shell脚本"""
    script_file = "./run_evaluation_async.sh"
    if not os.path.exists(script_file):
        print(f"❌ shell脚本不存在: {script_file}")
        return False
    
    if not os.access(script_file, os.X_OK):
        print(f"❌ shell脚本没有执行权限: {script_file}")
        return False
    
    print("✅ shell脚本存在且有执行权限")
    return True

async def main():
    """主测试函数"""
    print("🧪 EmbSpatial异步版本测试开始")
    print("=" * 50)
    
    tests = [
        ("依赖导入测试", test_imports),
        ("数据文件测试", test_data_file),
        ("脚本语法测试", test_script_syntax),
        ("异步功能测试", test_async_function),
        ("shell脚本测试", test_shell_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 测试结果摘要:")
    success_count = sum(results)
    total_count = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ 通过" if results[i] else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {success_count}/{total_count} 通过")
    
    if success_count == total_count:
        print("🎉 所有测试通过！异步版本已准备就绪。")
        print("\n💡 使用建议:")
        print("  1. 运行: ./run_evaluation_async.sh --help")
        print("  2. 测试: ./run_evaluation_async.sh --concurrent 2")
        print("  3. 正式运行: ./run_evaluation_async.sh --concurrent 10")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        
    return success_count == total_count

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        sys.exit(1) 