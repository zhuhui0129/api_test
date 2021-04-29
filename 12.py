import os
import socket
import json
import logging



"""
# 所有配置，通过完整key名获取配置
GLOBAL_CONFIGS = {
                    'mysql.asi.mix.feature_read': 'xxxx',
                }
"""
GLOBAL_CONFIGS = {}
HOST = socket.gethostbyname(socket.gethostname())


def get_mysql_configs():
    """
    获取当前域的和mix域的所有配置
    {
        'feature_read': {'host': 'xxx', 'port': 'xxx',..},
        'risk_read': {'host': 'xxx', 'port': 'xxx',..},
        'dana_read': {'host': 'xxx', 'port': 'xxx',..},
    }
    """
    mysql_configs = {}
    for full_key, value in GLOBAL_CONFIGS.items():
        key_slice = tuple(full_key.split('.'))
        if key_slice[0] != 'mysql':
            continue
        if len(key_slice) != 3:
            raise Exception(f'mysql 配置key命名不规范: {full_key}')
        group, business, db_type = key_slice
        # 取本域和公共配置
        if business in [GLOBAL_CONFIGS['env.business'].lower(), 'mix']:
            mysql_configs[db_type] = json.loads(value)
    return mysql_configs

res=get_mysql_configs
print(res())
