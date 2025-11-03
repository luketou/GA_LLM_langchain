# oracle_scoring.py

from guacamol.standard_benchmarks import (
    hard_osimertinib, hard_fexofenadine, ranolazine_mpo,
    amlodipine_rings, sitagliptin_replacement, zaleplon_with_other_formula,
    median_camphor_menthol, median_tadalafil_sildenafil, similarity,
    perindopril_rings, hard_cobimetinib, qed_benchmark, logP_benchmark,
    tpsa_benchmark, cns_mpo, scaffold_hop, decoration_hop, weird_physchem,
    isomers_c11h24, isomers_c9h10n2o2pf2cl, valsartan_smarts
)
import sys
import os
from guacamol.scoring_function import ScoringFunction 
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark 

# 添加根目錄到 Python 路徑以導入 logger_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import oracle_logger as logger

# Celecoxib SMILES 字串
CELECOXIB_SMILES = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
TROGLITAZONE_SMILES = 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O'
THIOTHIZENE_SMILES = 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1'

# 建立 task 名稱和對應 benchmark 函數的映射
TASK_BENCHMARK_MAPPING = {
    #MPO tasks
    'osimertinib': hard_osimertinib,
    'fexofenadine': hard_fexofenadine,
    'ranolazine': ranolazine_mpo,
    'amlodipine': amlodipine_rings,
    'perindopril': perindopril_rings,
    'sitagliptin': sitagliptin_replacement,
    'zaleplon': zaleplon_with_other_formula,
    'cobimetinib': hard_cobimetinib,
    'qed': qed_benchmark,
    'cns_mpo': cns_mpo,
    'scaffold_hop': scaffold_hop,
    'decoration_hop': decoration_hop,
    'weird_physchem': weird_physchem,
    'valsartan_smarts': valsartan_smarts,
    
    # median tasks
    'median1': median_camphor_menthol,
    'median2': median_tadalafil_sildenafil,

    # isomer tasks
    'isomer_c11h24': isomers_c11h24,
    'isomer_c9h10n2o2pf2cl': isomers_c9h10n2o2pf2cl,
    
    'logp_2.5': lambda: logP_benchmark(target_value=2.5),
    'tpsa_100': lambda: tpsa_benchmark(target_value=100),

    # rediscovery tasks
    'celecoxib': lambda: similarity(CELECOXIB_SMILES, 'Celecoxib', fp_type='ECFP4', threshold=1.0),
    'troglitazone': lambda: similarity(TROGLITAZONE_SMILES, 'Troglitazone', fp_type='ECFP4', threshold=1.0),
    'thiothixene': lambda: similarity(THIOTHIZENE_SMILES, 'Thiothixene', fp_type='ECFP4', threshold=1.0)

}

logger.info(f"已初始化 {len(TASK_BENCHMARK_MAPPING)} 個任務的 benchmark 映射")

# 預設使用的 benchmark，可依需求更換
DEFAULT_BENCHMARK = hard_osimertinib()
current_task = 'osimertinib'  # 預設任務
current_benchmark = DEFAULT_BENCHMARK  # 預設 benchmark
logger.info(f"預設任務設為: {current_task}")

def get_benchmark(task_name):
    """
    根據任務名稱取得相應的 benchmark
    
    Args:
        task_name (str): 任務名稱
        
    Returns:
        評分 benchmark 實例
    """
    # 轉為小寫以處理大小寫不敏感情況
    task_lower = task_name.lower()
    logger.debug(f"尋找任務 '{task_lower}' 的 benchmark")
    
    # 提取基本任務名稱 (對於 parametric 任務，如 logp_X)
    base_task = task_lower
    parametric_benchmark = None

    if '_' in task_lower:
        parts = task_lower.split('_')
        # 檢查是否為已知的參數化任務
        if parts[0] == 'logp' and len(parts) > 1:
            try:
                target = float(parts[1])
                logger.info(f"識別到參數化 logP benchmark，目標值: {target}")
                parametric_benchmark = logP_benchmark(target_value=target)() # 直接實例化
            except (ValueError, IndexError):
                logger.warning(f"解析 '{task_name}' 中的 logP 參數失敗")
        elif parts[0] == 'tpsa' and len(parts) > 1:
            try:
                target = float(parts[1])
                logger.info(f"識別到參數化 TPSA benchmark，目標值: {target}")
                parametric_benchmark = tpsa_benchmark(target_value=target)() # 直接實例化
            except (ValueError, IndexError):
                logger.warning(f"解析 '{task_name}' 中的 TPSA 參數失敗")
        else: # 如果不是已知的參數化任務，則將 base_task 設為 task_lower
            base_task = task_lower

    if parametric_benchmark:
        logger.info(f"為參數化任務 '{task_name}' 返回 benchmark 實例。")
        return parametric_benchmark

    # 從映射中取得 benchmark 函數
    benchmark_func_or_instance = TASK_BENCHMARK_MAPPING.get(task_lower)
    if not benchmark_func_or_instance: # 如果完整任務名找不到，嘗試基本任務名
        base_task_name_for_map = task_lower.split('_')[0] if '_' in task_lower else task_lower
        benchmark_func_or_instance = TASK_BENCHMARK_MAPPING.get(base_task_name_for_map)

    if benchmark_func_or_instance:
        logger.info(f"為任務 '{task_name}' 找到對應的 benchmark 項目。")
        # 檢查它是否已經是實例 (例如，來自 lambda: similarity(...)())
        # 或者它是一個需要被呼叫的函數 (例如，hard_osimertinib)
        if callable(benchmark_func_or_instance) and not isinstance(benchmark_func_or_instance, (ScoringFunction, GoalDirectedBenchmark)):
            try:
                instance = benchmark_func_or_instance()
                logger.info(f"已呼叫 benchmark 函數並獲得實例: {type(instance)}")
                return instance
            except Exception as e:
                logger.error(f"呼叫 benchmark 函數 '{benchmark_func_or_instance.__name__ if hasattr(benchmark_func_or_instance, '__name__') else str(benchmark_func_or_instance)}' 時發生錯誤: {e}")
                logger.warning(f"無法為任務 '{task_name}' 創建 benchmark 實例，將使用默認 benchmark。")
                return DEFAULT_BENCHMARK # 返回實例
        elif isinstance(benchmark_func_or_instance, (ScoringFunction, GoalDirectedBenchmark)):
             logger.info(f"直接返回已實例化的 benchmark: {type(benchmark_func_or_instance)}")
             return benchmark_func_or_instance
        else:
            logger.warning(f"任務 '{task_name}' 的 benchmark 項目類型未知: {type(benchmark_func_or_instance)}，將使用默認 benchmark。")
            return DEFAULT_BENCHMARK # 返回實例
    
    logger.warning(f"警告: 無法找到任務 '{task_name}' 的 benchmark，使用默認任務 'osimertinib'")
    return DEFAULT_BENCHMARK # 返回實例

def set_task(task_name):
    """
    設定當前使用的任務和 benchmark
    
    Args:
        task_name (str): 任務名稱
    
    Returns:
        str: 設定的任務名稱
    """
    global current_task, current_benchmark
    current_task = task_name
    # get_benchmark 現在總是返回一個實例
    benchmark_instance = get_benchmark(task_name)
    
    # 從 GoalDirectedBenchmark 實例中提取 objective
    if isinstance(benchmark_instance, GoalDirectedBenchmark):
        logger.info(f"從 GoalDirectedBenchmark 實例中提取 objective 作為 current_benchmark。")
        current_benchmark = benchmark_instance.objective
    elif isinstance(benchmark_instance, ScoringFunction):
        logger.info(f"直接使用 ScoringFunction 實例作為 current_benchmark。")
        current_benchmark = benchmark_instance
    else:
        logger.warning(f"get_benchmark 返回了未預期的類型 {type(benchmark_instance)}，current_benchmark 可能不正確。")
        current_benchmark = benchmark_instance # 作為備案

    logger.info(f"已設定任務為: {current_task}, current_benchmark 類型: {type(current_benchmark)}")
    print(f"已設定任務為: {current_task}")
    return current_task

def score_smiles(smiles: list, benchmark_or_task_name=None) -> list:
    """
    Oracle scoring: 
    對每個 SMILES 呼叫 GuacaMol benchmark 的 score_molecule 方法，
    回傳一個與 smiles 等長的分數列表。
    
    Args:
        smiles: SMILES 字串列表
        benchmark_or_task_name: 指定的 benchmark 實例或任務名稱字串 (可選，若不指定則使用當前全域 benchmark)
    
    Returns:
        分數列表
    """
    scoring_function_to_use = None

    if benchmark_or_task_name is None:
        logger.info(f"未提供 benchmark_or_task_name，使用 current_benchmark (類型: {type(current_benchmark)})")
        scoring_function_to_use = current_benchmark
    elif isinstance(benchmark_or_task_name, str):
        logger.info(f"接收到字符串形式的 benchmark_or_task_name: '{benchmark_or_task_name}'，嘗試從任務名稱獲取 benchmark。")
        benchmark_instance = get_benchmark(benchmark_or_task_name)
        if isinstance(benchmark_instance, GoalDirectedBenchmark):
            logger.info(f"從 get_benchmark 返回的 GoalDirectedBenchmark 中提取 objective。")
            scoring_function_to_use = benchmark_instance.objective
        elif isinstance(benchmark_instance, ScoringFunction):
            logger.info(f"get_benchmark 直接返回了 ScoringFunction。")
            scoring_function_to_use = benchmark_instance
        else:
            logger.error(f"get_benchmark 返回了未預期類型 {type(benchmark_instance)} for task '{benchmark_or_task_name}'。將使用默認零分。")
            scoring_function_to_use = lambda s: 0.0
    elif isinstance(benchmark_or_task_name, GoalDirectedBenchmark):
        logger.info(f"接收到 GoalDirectedBenchmark 實例，提取其 objective。")
        scoring_function_to_use = benchmark_or_task_name.objective
    elif isinstance(benchmark_or_task_name, ScoringFunction):
        logger.info(f"接收到 ScoringFunction 實例，直接使用。")
        scoring_function_to_use = benchmark_or_task_name
    else:
        logger.warning(f"接收到未預期的 benchmark_or_task_name 類型: {type(benchmark_or_task_name)}。嘗試直接使用。")
        scoring_function_to_use = benchmark_or_task_name # 可能仍然是 callable

    if not callable(scoring_function_to_use) and not hasattr(scoring_function_to_use, 'score'):
        logger.error(f"最終的 scoring_function_to_use (類型: {type(scoring_function_to_use)}) 不是可呼叫的，也沒有 score 方法。評分將失敗。")
        return [0.0] * len(smiles)
        
    logger.debug(f"準備評分 {len(smiles)} 個分子，使用評分函數類型: {type(scoring_function_to_use)}")
    
    scores = []
    for smi in smiles:
        try:
            # 現在 scoring_function_to_use 應該是 ScoringFunction 實例或類似的 callable
            if hasattr(scoring_function_to_use, 'score') and callable(scoring_function_to_use.score):
                score = scoring_function_to_use.score(smi)
            elif callable(scoring_function_to_use): # 作為備案，如果它本身就是一個可呼叫的評分函數
                score = scoring_function_to_use(smi)
            else:
                logger.error(f"無法對 SMILES '{smi}' 進行評分，評分函數 (類型: {type(scoring_function_to_use)}) 既沒有 'score' 方法也不可呼叫。")
                score = 0.0
            scores.append(score)
        except Exception as e:
            logger.error(f"評分分子 '{smi}' 時發生錯誤: {str(e)}", exc_info=True)
            scores.append(0.0)
            
    if scores:
        avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        logger.info(f"評分完成: 平均={avg_score:.4f}, 最高={max_score:.4f}, 最低={min_score:.4f} (共 {len(scores)} 個分子)")
    
    return scores

def guacamol_score_llm(args) -> list:
    """
    Score SMILES strings using GuacaMol benchmark
    
    Args:
        args: Can be:
            - Dictionary with 'smiles' or 'smiles_list' key
            - JSON string representing such a dictionary
            - A direct list of SMILES strings
            - Dictionary of form {'smiles': smiles, 'task': task_name} to specify a task
    
    Returns:
        list: List of scores corresponding to input SMILES
    """
    # Extract SMILES list from various input formats
    smiles_list = []
    specified_task = None
    
    # Handle string input (convert JSON string to dict)
    if isinstance(args, str):
        try:
            import json
            args = json.loads(args)
        except Exception as e:
            logger.error(f"Failed to parse input as JSON: {e}")
            return []
    
    # Handle dictionary input
    if isinstance(args, dict):
        # Check for optional task specification
        if 'task' in args:
            specified_task = args.get('task')
            logger.info(f"Custom task specified for scoring: {specified_task}")
        
        # Extract SMILES from various possible keys
        if 'smiles' in args:
            smiles_list = args['smiles']
        elif 'smiles_list' in args:
            smiles_list = args['smiles_list']
        elif 'molecules' in args:
            smiles_list = args['molecules']
    
    # Handle direct list input (from llm_generate or other sources)
    elif isinstance(args, list):
        smiles_list = args
        logger.info(f"Received direct list of {len(args)} SMILES")
    
    # Handle unexpected input types
    else:
        logger.warning(f"Unexpected input type: {type(args)}, attempting to convert to list")
        try:
            smiles_list = list(args)
        except Exception as e:
            logger.error(f"Could not convert {type(args)} to list: {e}")
            return []
    
    # Ensure smiles_list is actually a list
    if not isinstance(smiles_list, list):
        smiles_list = [smiles_list]
    
    # Additional processing for dict of molecules with scores
    processed_smiles = []
    for item in smiles_list:
        if isinstance(item, dict) and 'smiles' in item:
            processed_smiles.append(item['smiles'])
        elif isinstance(item, str):
            processed_smiles.append(item)
    
    if processed_smiles:
        smiles_list = processed_smiles
    
    # Log information about the scoring request
    logger.info(f"收到評分請求, 分子數量: {len(smiles_list)}")
    if smiles_list:
        logger.debug(f"First few SMILES to score: {smiles_list[:min(3, len(smiles_list))]}")
    else:
        logger.warning("No SMILES to score after processing input")
        return []
    
    # Use specified task or current global task
    task_to_use = specified_task if specified_task else current_task
    
    # Get the appropriate benchmark for specified task
    if specified_task:
        benchmark = get_benchmark(specified_task)
        logger.info(f"Using custom benchmark for task: {specified_task}")
        return score_smiles(smiles_list, benchmark)
    else:
        # Use the current global benchmark
        logger.info(f"Using current global benchmark for task: {current_task}")
        return score_smiles(smiles_list)

def get_current_task():
    """
    取得當前任務名稱
    
    Returns:
        str: 當前任務名稱
    """
    logger.debug(f"取得當前任務: {current_task}")
    return current_task