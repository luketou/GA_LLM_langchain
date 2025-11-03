#!/usr/bin/env python3
"""
統一的日誌配置模組，用於整個分子優化項目。
提供可以在所有模組間共享的日誌記錄功能。
"""

import os
import logging
import logging.handlers
from datetime import datetime
import sys

# 定義日誌目錄
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 建立日誌格式
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# 建立時間格式
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logger(name, log_file=None, level=logging.INFO, console=True, detailed_format=False):
    """
    設置並返回自定義的 logger
    
    Args:
        name: logger 的名稱 
        log_file: 日誌文件路徑（可選）
        level: 日誌級別
        console: 是否也記錄到控制台
        detailed_format: 是否使用詳細格式（包含文件名和行號）
        
    Returns:
        Logger: 配置好的 logger 實例
    """
    # 選擇格式
    log_format = DETAILED_LOG_FORMAT if detailed_format else DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(log_format, TIME_FORMAT)
    
    # 創建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 確保 handler 不會重複添加
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 如果提供了日誌文件，添加文件處理器
    if log_file:
        # 如果沒有提供絕對路徑，則使用預設日誌目錄
        if not os.path.isabs(log_file):
            log_file = os.path.join(LOG_DIR, log_file)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 如果需要控制台輸出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# 創建一個預設的應用程式日誌記錄器
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')



# 為各個主要模組創建專用日誌記錄器
main_logger = setup_logger('main', os.path.join(LOG_DIR, f'main_{timestamp}.log'))
agent_logger = setup_logger('agent', os.path.join(LOG_DIR, f'agent_{timestamp}.log'))
oracle_logger = setup_logger('oracle', os.path.join(LOG_DIR, f'oracle_{timestamp}.log'))
generator_logger = setup_logger('generator', os.path.join(LOG_DIR, f'generator_{timestamp}.log'))
ga_logger = setup_logger('graph_ga', os.path.join(LOG_DIR, f'graph_ga_{timestamp}.log'))

def get_logger(name=None):
    """
    獲取指定名稱的 logger 或預設 logger
    
    Args:
        name: logger 的名稱（可選）
        
    Returns:
        Logger: 已存在的或新建的 logger
    """
    
    # 檢查是否已有此名稱的 logger
    existing_logger = logging.getLogger(name)
    if existing_logger.hasHandlers():
        return existing_logger
    
    # 否則創建新的 logger
    log_file = os.path.join(LOG_DIR, f'{name}_{timestamp}.log')
    return setup_logger(name, log_file)