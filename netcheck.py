# redis_debug.py - 详细诊断Redis连接和数据获取问题
import os
import sys
import traceback
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 设置Python路径
sys.path.insert(0, '/Node/src')
import src.set_pypath

def diagnose_redis_issue():
    print("=" * 60)
    print("🔍 Redis连接和数据获取诊断")
    print("=" * 60)
    
    # 1. 环境变量检查
    print("\n1. 环境变量检查:")
    env_vars = ['REDIS_HOST', 'REDIS_PORT', 'REDIS_DB', 'REDIS_PWD']
    for var in env_vars:
        value = os.getenv(var, 'NOT_SET')
        print(f"   {var}: {value}")
    
    # 2. 导入检查
    print("\n2. 模块导入检查:")
    try:
        from src.worker.data.redis import REDIS_CONNECTOR
        from src.worker.data.redis import REDIS_PREFIX_MANAGER
        print("   ✅ 模块导入成功")
    except Exception as e:
        print(f"   ❌ 模块导入失败: {e}")
        return
    
    # 3. 连接器状态检查
    print("\n3. Redis连接器状态:")
    print(f"   连接器对象: {REDIS_CONNECTOR}")
    print(f"   连接器类: {type(REDIS_CONNECTOR)}")
    
    # 4. 键构建检查
    print("\n4. 键构建检查:")
    try:
        key = REDIS_PREFIX_MANAGER.build_train_slice_key(1997, 1, '000001')
        print(f"   生成的键: {key}")
        print(f"   键类型: {type(key)}")
    except Exception as e:
        print(f"   ❌ 键构建失败: {e}")
        return
    
    # 5. 连接建立检查
    print("\n5. Redis连接建立:")
    try:
        redis_client = REDIS_CONNECTOR.get_client()
        print(f"   客户端对象: {redis_client}")
        print(f"   客户端类型: {type(redis_client)}")
        print("   ✅ 连接建立成功")
    except Exception as e:
        print(f"   ❌ 连接建立失败: {e}")
        traceback.print_exc()
        return
    
    # 6. 连接测试
    print("\n6. Redis连接测试:")
    try:
        ping_result = redis_client.ping()
        print(f"   PING结果: {ping_result}")
        print("   ✅ PING测试成功")
    except Exception as e:
        print(f"   ❌ PING测试失败: {e}")
        traceback.print_exc()
        return
    
    # 7. 数据获取测试
    print("\n7. 数据获取测试:")
    try:
        print(f"   尝试获取键: {key}")
        data = redis_client.get(key)
        print(f"   数据类型: {type(data)}")
        print(f"   数据长度: {len(data) if data else 'None'}")
        
        if data is None:
            print("   ℹ️  键不存在或为空")
            
            # 尝试设置一个测试值
            print("   尝试设置测试数据...")
            test_key = "test:debug_console"
            test_value = "hello from direct execution"
            redis_client.set(test_key, test_value)
            
            retrieved = redis_client.get(test_key)
            print(f"   测试设置结果: {retrieved}")
            
            # 清理测试数据
            redis_client.delete(test_key)
            
        else:
            print("   ✅ 数据获取成功")
            # 只显示前100个字符
            data_str = str(data)
            if len(data_str) > 100:
                print(f"   数据内容(前100字符): {data_str[:100]}...")
            else:
                print(f"   数据内容: {data_str}")
                
    except Exception as e:
        print(f"   ❌ 数据获取失败: {e}")
        traceback.print_exc()
    
    # 8. 连接信息
    print("\n8. 连接详细信息:")
    try:
        connection_info = redis_client.connection
        print(f"   连接对象: {connection_info}")
        print(f"   连接池: {redis_client.connection_pool}")
        print(f"   连接池大小: {len(redis_client.connection_pool._available_connections) if hasattr(redis_client.connection_pool, '_available_connections') else 'Unknown'}")
    except Exception as e:
        print(f"   获取连接信息失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 诊断完成")

if __name__ == "__main__":
    diagnose_redis_issue()