# tools/generator.py
from tools.graph_ga.goal_directed_generation import GB_GA_Generator
from prompt import llm_generate_template,get_scoring_criteria
from utils.cerebras_llm import CerebrasChatLLM
from tools.oracle_scoring import get_current_task, get_benchmark, guacamol_score_llm, score_smiles
from guacamol.scoring_function import ScoringFunction 
import sys
import os
import os.path
import re # Add import for regular expressions
import numpy as np
import csv
import json
from datetime import datetime
import heapq
from utils.file_config import setup_result_dir
from utils.record_generation import record_generation_stats, record_generation_details_json, save_elite_pool, load_generation_history_from_json, get_run_timestamp
from rdkit import Chem
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# 添加根目錄到 Python 路徑以導入 logger_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_config import generator_logger as logger
from agent import get_remaining_oracle_budget, update_remaining_oracle_budget, get_total_oracle_budget

# 初始化 LLM 用於生成分子
generation_llm = None
GLOBAL_MOLECULE_POOL_SIZE = 100
global_elite_pool = [] # size = 100
logger.info("Generation LLM will be initialized with API key")

def graph_ga_generate(args: dict):
    """
    使用 Graph GA 生成分子
    """
    # 如果收到的是字串，嘗試當成 JSON 解析
    if isinstance(args, str):
        try:
            import json
            args = json.loads(args)
        except Exception:
            logger.warning("graph_ga_generate: 無法解析字串參數為 JSON，返回空結果")
            return []
    
    # 型別檢查
    if not isinstance(args, dict):
        logger.warning(f"graph_ga_generate: 預期 dict 類型，卻收到 {type(args)}，返回空結果")
        return []

    params = args.get('params', {})
    population_size = args.get('population', 100)
    
    # center oracle budget
    oracle_budget = get_remaining_oracle_budget()
    logger.info(f"啟動 Graph GA 生成，種群大小: {population_size}，剩餘預算: {oracle_budget}")
    logger.debug(f"Graph GA 參數: {params}")
    
    # ----------------get current task for scoring function----------------
    current_task = get_current_task()
    logger.info(f"當前任務: {current_task}")
    
    actual_scoring_function_for_ga = None

    # 碮保 params 有正確的 scoring_function
    if 'scoring_function' not in params or params['scoring_function'] is None:
        from tools.oracle_scoring import get_benchmark
        benchmark_object = get_benchmark(current_task) # benchmark_object 是 GoalDirectedBenchmark 實例
        
        # GoalDirectedBenchmark 物件的 .objective 屬性是 ScoringFunction 實例
        if hasattr(benchmark_object, 'objective') and isinstance(benchmark_object.objective, ScoringFunction):
            actual_scoring_function_for_ga = benchmark_object.objective
            logger.info(f"已從任務 '{current_task}' 的 benchmark 中提取 'objective' 作為評分函數。")
        else:
            logger.error(f"無法從任務 '{current_task}' 的 benchmark 物件 (類型: {type(benchmark_object)}) 中獲取有效的 'objective' (ScoringFunction)。將嘗試使用 create_scoring_wrapper 作為備案。")
            # 保留 create_scoring_wrapper 作為備案，儘管它可能有問題
            actual_scoring_function_for_ga = create_scoring_wrapper(benchmark_object)
            if not callable(actual_scoring_function_for_ga): # 確保 wrapper 至少是可呼叫的
                 logger.error(f"備案 create_scoring_wrapper 也未能提供可用的評分函數。")
                 return []
            logger.info(f"已使用 create_scoring_wrapper 為任務 '{current_task}' 創建 benchmark 包裝函數作為備案。")
    else:
        # 如果 params 中提供了 scoring_function，假設它已經是正確的類型
        actual_scoring_function_for_ga = params['scoring_function']
        # 進行更嚴格的檢查，確保它是 ScoringFunction 的實例或至少有 score 方法
        if not (isinstance(actual_scoring_function_for_ga, ScoringFunction) or \
                (hasattr(actual_scoring_function_for_ga, 'score') and callable(actual_scoring_function_for_ga.score))):
            logger.error(f"params 中提供的 'scoring_function' 不是一個有效的 ScoringFunction 物件或缺少 score 方法。類型: {type(actual_scoring_function_for_ga)}")
            return [] 

    # 檢查 actual_scoring_function_for_ga 是否成功設定
    if actual_scoring_function_for_ga is None:
        logger.error("評分函數 (scoring_function) 未能成功初始化。")
        return []

    try:
        # 初始化 GA 生成器
        ga_results_dir = setup_result_dir(current_task) # 從 params 獲取 results_dir

        # 準備精英池中的分子作為可能的起始種群
        starting_population = None
        elite_molecules = get_elite_molecules()
        if (elite_molecules and len(elite_molecules) >= 10):  # 至少需要一定數量的精英分子
            logger.info(f"使用精英池中的 {len(elite_molecules)} 個高分子作為起始種群")
            # 從精英池中提取 SMILES 字符串
            starting_population = [smiles for _, smiles in elite_molecules]
        else:
            logger.info("精英池為空或分子不足，將使用預設的起始種群")

        # Make add_to_elite_pool, save_elite_pool, and get_current_task available to GB_GA_Generator
        # This will allow the GA class to update the elite pool directly
        import sys
        from sys import modules
        
        # Ensure these functions are in globals() for the GA class to find
        modules['add_to_elite_pool'] = add_to_elite_pool
        modules['save_elite_pool'] = save_elite_pool
        modules['get_current_task'] = get_current_task
        
        ga = GB_GA_Generator(
            smi_file=params.get('smi_file', '/home/luketou/LLM_AI_agent/Agent_predictor/data/guacamol_v1_all.txt'),
            population_size=population_size,
            offspring_size=params.get('offspring_size', 100),
            generations=params.get('generations', 3), 
            mutation_rate=params.get('mutation_rate', 0.5),
            crossover_rate=params.get('crossover_rate', 1), 
            n_jobs=params.get('n_jobs', 1),
            random_start=params.get('random_start', False),
            top_from_random=params.get('top_from_random', False),
            random_sample_size=params.get('random_sample_size', 1000),
            use_lowest_scoring=params.get('use_lowest_scoring', True), 
            patience=params.get('patience', 5),
            llm_agent=None, 
            llm_frequency=params.get('llm_frequency', 5),
            return_initial_population=False,
            oracle_budget=oracle_budget,  # 使用中央化的預算值
            dynamic_offspring_size=params.get('dynamic_offspring_size', False),
            batch_size=params.get('batch_size', 1000),
            early_stopping=params.get('early_stopping', False),
            results_dir=ga_results_dir, # 傳遞 results_dir 給 GA
            task_name=current_task  # 使用當前任務名稱
        )
        logger.info("GB_GA_Generator 初始化成功")
        ga.generate_optimized_molecules(
            scoring_function=actual_scoring_function_for_ga,
            number_molecules=population_size,
            starting_population=starting_population,  # 使用精英池中的分子作為起始種群
            results_dir=ga_results_dir,  # 傳遞 results_dir
        )
        
        raw_history = ga.generation_history
        logger.info(f"Graph GA raw history received with {len(raw_history)} generations.")

        processed_history = []
        for i, gen_data in enumerate(raw_history):
            # Create base generation data structure
            processed_gen_data = {
                "generation": gen_data.get('generation', i),
                "source": "GA",
                "max_score": gen_data.get('max_score'),
                "avg_score": gen_data.get('avg_score'),
                "min_score": gen_data.get('min_score', 0),
                "std_score": gen_data.get('std_score', 0),
                "best_smiles": gen_data.get('best_smiles'),
                "remaining_budget": gen_data.get('remaining_budget')
            }

            # 優先使用 evaluated_offspring 數據記錄每一代所有評估過的分子
            molecules_with_scores = []
            
            # 檢查是否有 evaluated_offspring 數據（記錄了所有評估的後代分子）
            if 'evaluated_offspring' in gen_data and gen_data['evaluated_offspring']:
                logger.info(f"Generation {processed_gen_data['generation']}: Using evaluated_offspring data with {len(gen_data['evaluated_offspring'])} molecules")
                molecules_with_scores = gen_data['evaluated_offspring']
                
                # 使用所有評估過的分子來計算和排序，找出 top 分子
                sorted_molecules = sorted(molecules_with_scores, key=lambda x: x['score'], reverse=True)
                top_10_molecules = sorted_molecules[:10]
                
                # 排序找出最低分的分子
                sorted_molecules_asc = sorted(molecules_with_scores, key=lambda x: x['score'])
                # bottom_5_molecules = sorted_molecules_asc[:5]
                
                # 記錄詳細的分子資料到 JSON 使用中央化函數
                record_generation_details_json(
                    task_name=current_task,
                    generation_type="GA",
                    generation_num=processed_gen_data['generation'],
                    molecules_data=molecules_with_scores,
                    oracle_budget=gen_data.get('remaining_budget', oracle_budget)
                )
                
                # 記錄統計資料到 CSV 使用中央化函數
                # 最多取前 100 個分子用於記錄統計資料
                top_100_molecules = sorted_molecules[:min(100, len(sorted_molecules))]
                top_100_scores = [mol["score"] for mol in top_100_molecules]
                
                record_generation_stats(
                    task_name=current_task, 
                    generation_type="GA",
                    generation_num=processed_gen_data['generation'],
                    scores=top_100_scores, 
                    source="GA",
                    oracle_calls=len(molecules_with_scores),
                    mutation_rate=gen_data.get('mutation_rate'),
                    offspring_size=gen_data.get('offspring_size'),
                    crossover_rate=gen_data.get('crossover_rate')
                )
                
                # 將高分分子添加到精英池
                for mol in top_100_molecules:
                    add_to_elite_pool(mol['SMILES'], mol['score'])
                
            # 如果沒有 evaluated_offspring 數據，則回退到使用原有的 population_smiles/scores 
            elif 'population_smiles' in gen_data and 'population_scores' in gen_data:
                population_smiles = gen_data.get('population_smiles', [])
                population_scores = gen_data.get('population_scores', [])
                
                # 如果長度匹配則建立分子-分數對
                if len(population_smiles) == len(population_scores):
                    for smiles, score in zip(population_smiles, population_scores):
                        if smiles is not None and score is not None:
                            try:
                                molecules_with_scores.append({
                                    "SMILES": smiles, 
                                    "score": float(score)
                                })
                            except (ValueError, TypeError):
                                # Skip entries that can't be converted to proper format
                                continue
                    
                    if molecules_with_scores:
                        # Sort by score for top molecules (highest first)
                        sorted_molecules = sorted(molecules_with_scores, key=lambda x: x['score'], reverse=True)
                        top_10_molecules = sorted_molecules[:10]
                        
                        # Sort by score for bottom molecules (lowest first)
                        sorted_molecules_asc = sorted(molecules_with_scores, key=lambda x: x['score'])
                        # bottom_5_molecules = sorted_molecules_asc[:5]
                        
                        # Get top 100 molecules (or all if less than 100)
                        top_100_molecules = sorted_molecules[:min(100, len(sorted_molecules))]
                        top_100_scores = [mol["score"] for mol in top_100_molecules]
                        
                        # Record statistics using centralized functions
                        record_generation_stats(
                            task_name=current_task, 
                            generation_type="GA",
                            generation_num=processed_gen_data['generation'],
                            scores=top_100_scores, 
                            source="GA",
                            oracle_calls=len(molecules_with_scores),
                            mutation_rate=gen_data.get('mutation_rate'),
                            offspring_size=gen_data.get('offspring_size'),
                            crossover_rate=gen_data.get('crossover_rate')
                        )
                        
                        # Record detailed molecule data using centralized function
                        record_generation_details_json(
                            task_name=current_task,
                            generation_type="GA",
                            generation_num=processed_gen_data['generation'],
                            molecules_data=molecules_with_scores,
                            oracle_budget=gen_data.get('remaining_budget', oracle_budget)
                        )
                        
                        # Add the top molecules to the global elite pool
                        for mol in top_100_molecules:  # Add all top 100 molecules to elite pool
                            add_to_elite_pool(mol['SMILES'], mol['score'])
                    else:
                        logger.warning(f"GA Generation {processed_gen_data['generation']}: No valid molecule-score pairs found.")
                        top_10_molecules = []
                        bottom_5_molecules = []
                else:
                    # Check if there's a single "best molecule" we can use
                    best_smiles = gen_data.get('best_smiles')
                    max_score = gen_data.get('max_score')
                    
                    if best_smiles and max_score is not None:
                        top_10_molecules = [{"SMILES": best_smiles, "score": float(max_score)}]
                        # bottom_5_molecules = top_10_molecules
                        logger.info(f"GA Generation {processed_gen_data['generation']}: Using best molecule as fallback.")
                    else:
                        logger.warning(f"GA Generation {processed_gen_data['generation']}: Missing or mismatched population data.")
                        top_10_molecules = []
                        # bottom_5_molecules = []
            else:
                # 沒有任何分子數據
                logger.warning(f"GA Generation {processed_gen_data['generation']}: No molecule data found.")
                top_10_molecules = []
                # bottom_5_molecules = []
            
            # Add the results to the processed data
            processed_gen_data['top_10_molecules'] = top_10_molecules
            # processed_gen_data['bottom_5_molecules'] = bottom_5_molecules
            processed_history.append(processed_gen_data)

        # Save the elite pool after all generations using centralized function
        save_elite_pool(
            elite_pool=global_elite_pool,
            results_dir=ga_results_dir,
            task_name=current_task,
            generation_type="GA"
        )
        
        logger.info(f"Graph GA processed history with {len(processed_history)} generations.")
        return processed_history
        

    except Exception as e:
        logger.error(f"Graph GA 生成過程發生錯誤: {str(e)}", exc_info=True)
        return []  # 發生錯誤時返回空列表

def llm_generate(args) -> dict:
    """
    Generate molecules using LLM with RDKit validation.
    Retry generation for invalid molecules up to two times.
    Records statistics for the generated batch and updates the elite pool.
    """
    if isinstance(args, str):
        try:
            import json
            args = json.loads(args)
        except Exception:
            logger.warning("llm_generate: Invalid string input.")
            return {"error": "Invalid input format"}

    count = args.get('count', 10)
    current_task = get_current_task()

    # Load historical molecules for examples
    all_historical_molecules_from_json = []
    try:
        json_history_data = load_generation_history_from_json(current_task)
        if json_history_data and 'generations' in json_history_data:
            for gen_entry in json_history_data['generations']:
                if 'molecules' in gen_entry and isinstance(gen_entry['molecules'], list):
                    all_historical_molecules_from_json.extend(gen_entry['molecules'])
    except Exception as e:
        logger.error(f"Error loading historical molecules: {str(e)}", exc_info=True)

    examples_text = "No historical examples available."
    if all_historical_molecules_from_json:
        valid_molecules_from_history = [mol for mol in all_historical_molecules_from_json if isinstance(mol, dict) and 'SMILES' in mol and 'score' in mol]
        if valid_molecules_from_history:
            sorted_molecules_history = sorted(valid_molecules_from_history, key=lambda x: x['score'], reverse=True)
            top_10 = sorted_molecules_history[:10]
            # bottom_5 = sorted(valid_molecules_from_history, key=lambda x: x['score'])[:10]
            examples_text = "\n".join([
                "Top 10 Molecules from history:",
                *[f"  {i+1}. SMILES: {mol['SMILES']}, Score: {mol['score']:.4f}" for i, mol in enumerate(top_10)],
                # "\nBottom 5 Molecules from history:",
                # *[f"  {i+1}. SMILES: {mol['SMILES']}, Score: {mol['score']:.4f}" for i, mol in enumerate(bottom_5)]
            ])

    generated_smiles_validated = []
    max_retries = 2
    retries = 0

    while retries <= max_retries and len(generated_smiles_validated) < count:
        remaining_to_generate = count - len(generated_smiles_validated)
        template_params = {
            "example_block": examples_text,
            "count": remaining_to_generate,
            "task_description": get_scoring_criteria(current_task)
        }

        try:
            prompt = llm_generate_template.substitute(template_params)
            # Ensure generation_llm is initialized
            if generation_llm is None:
                logger.error("Generation LLM is not initialized. Cannot generate molecules.")
                return {"examples": examples_text, "generated_molecules": [], "error": "Generation LLM not initialized"}

            raw_llm_output_obj = generation_llm.invoke(prompt, max_tokens=remaining_to_generate * 80 + 200)
            
            # Handle different types of LLM output (e.g. AIMessage content)
            if hasattr(raw_llm_output_obj, 'content'):
                raw_llm_output = raw_llm_output_obj.content
            elif isinstance(raw_llm_output_obj, str):
                raw_llm_output = raw_llm_output_obj
            else:
                raw_llm_output = str(raw_llm_output_obj)


            potential_smiles = re.findall(r'[OCSNFClBrIcnosp\(\)=#\[\]@H1-9\.\/\\]+', raw_llm_output)
            current_batch_valid_smiles = [s for s in potential_smiles if Chem.MolFromSmiles(s)]

            generated_smiles_validated.extend(current_batch_valid_smiles[:remaining_to_generate])
        except Exception as e:
            logger.error(f"Error during LLM generation (attempt {retries + 1}): {str(e)}", exc_info=True)

        retries += 1

    if len(generated_smiles_validated) < count:
        logger.warning(f"Only {len(generated_smiles_validated)} valid molecules generated after {max_retries +1} attempts.")

    scored_molecules = []
    if generated_smiles_validated:
        logger.info(f"Scoring {len(generated_smiles_validated)} LLM-generated molecules.")
        scores = score_smiles(generated_smiles_validated, current_task)
        
        oracle_calls_this_batch = len(generated_smiles_validated)
        current_budget = get_remaining_oracle_budget()
        new_budget = current_budget - oracle_calls_this_batch
        update_remaining_oracle_budget(new_budget)
        logger.info(f"Oracle budget updated: {current_budget} -> {new_budget} after {oracle_calls_this_batch} calls by LLM.")

        scored_molecules = [{"SMILES": smi, "score": score} for smi, score in zip(generated_smiles_validated, scores) if score is not None]
        
        if scored_molecules:
            logger.info(f"Successfully scored {len(scored_molecules)} molecules by LLM.")
            
            # 1. Record statistics for this specific LLM batch
            llm_batch_scores = [mol['score'] for mol in scored_molecules]
            record_generation_stats(
                task_name=current_task,
                generation_type="LLM",
                generation_num="LLM_Batch", # Identifier for LLM batches
                scores=llm_batch_scores,
                source="LLM",
                oracle_calls=oracle_calls_this_batch,
                notes=f"LLM generated batch of {len(scored_molecules)} molecules"
            )
            
            # 2. Add scored molecules to the global elite pool
            for mol in scored_molecules:
                add_to_elite_pool(mol['SMILES'], mol['score'])
            
            # 3. Save the updated elite pool (this also records elite pool stats)
            results_dir = setup_result_dir(current_task)
            save_elite_pool(
                elite_pool=global_elite_pool,
                results_dir=results_dir,
                task_name=current_task,
                generation_type="LLM_Elite_Update" # Indicates elite pool snapshot after LLM update
            )
        else:
            logger.warning("No molecules were successfully scored from the LLM batch.")
    else:
        logger.info("No valid SMILES were generated by LLM to score.")

    return {"examples": examples_text, "generated_molecules": scored_molecules}


##==================================================================================
# small function 

def initialize_llm_with_key(api_key):
    """使用提供的 API 密鑰初始化 LLM"""
    global generation_llm
    try:
        generation_llm = CerebrasChatLLM(
            model="gpt-oss-120b",
            temperature=0.7,
            api_key=api_key
        )
        logger.info("Generation LLM initialized with provided API key")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize generation LLM: {str(e)}")
        return False

def add_to_elite_pool(smiles, score):
    """
    將高分子添加到全局精英池中
    
    Args:
        smiles: SMILES 字符串
        score: 分子評分
    """
    global global_elite_pool

    # 檢查分子是否已在池中
    existing_smiles = [s for _, s in global_elite_pool]
    if smiles in existing_smiles:
        return  # 如果分子已存在，則不添加
    
    # 如果池未滿，直接添加
    if len(global_elite_pool) < GLOBAL_MOLECULE_POOL_SIZE:
        heapq.heappush(global_elite_pool, (score, smiles))
    else:
        # 如果池已滿且新分子得分更高，則替換最低分的分子
        lowest_score = global_elite_pool[0][0] if global_elite_pool else 0
        if score > lowest_score:
            heapq.heappushpop(global_elite_pool, (score, smiles))

def get_elite_molecules():
    """
    從精英池中獲取分子，按得分從高到低排序
    
    Returns:
        List of (score, smiles) tuples sorted by score in descending order
    """
    # sort the elite pool by score in descending order
    return sorted(global_elite_pool, key=lambda x: x[0], reverse=True)

# 添加這個包裝函數來處理 GoalDirectedBenchmark 類型的評分函數
def create_scoring_wrapper(benchmark):
    """
    為各種類型的評分函數創建統一的包裝器
    
    Args:
        benchmark: 評分對象，可能是 GuacaMol benchmark 或其他類型的評分函數
    
    Returns:
        callable: 能夠接受 SMILES 字符串並返回分數的函數
    """
    # 針對 GoalDirectedBenchmark 類型創建包裝
    if hasattr(benchmark, 'score_molecule'):
        logger.info("創建 score_molecule 方法的包裝函數")
        return lambda smiles: benchmark.score_molecule(smiles)
    # 針對具有 score 方法的對象
    elif hasattr(benchmark, 'score'):
        logger.info("創建 score 方法的包裝函數")
        return lambda smiles: benchmark.score(smiles)
    # 針對已經可調用的函數
    elif callable(benchmark):
        logger.info("使用直接可調用的評分函數")
        return benchmark
    # 針對其他特殊情況
    elif hasattr(benchmark, 'raw_scoring_function'):
        logger.info("創建 raw_scoring_function 方法的包裝函數")
        return lambda smiles: benchmark.raw_scoring_function(smiles)
    else:
        logger.error(f"無法創建評分函數包裝器，未知的評分函數類型: {type(benchmark)}")
        # 返回默認的零分函數
        return lambda smiles: 0.0
