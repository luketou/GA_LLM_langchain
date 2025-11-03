import sys
import os
import json

# 將專案根目錄添加到 Python 路徑，以便導入模組
# 假設此腳本與您的專案資料夾結構一致
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 如果您的 tools 和 agent 模組在 GA_LLM_langchain 目錄下，
# 並且此腳本在 GA_LLM_langchain 的父目錄，您可能需要調整路徑
# 例如: sys.path.append(os.path.join(project_root, "GA_LLM_langchain"))
# 根據您的實際結構調整下面的導入
try:
    from tools.generator import graph_ga_generate, initialize_llm_with_key
    from tools.oracle_scoring import set_task, get_current_task
    from agent import initialize_parameter_llm, get_remaining_oracle_budget, update_remaining_oracle_budget, TOTAL_ORACLE_BUDGET
    from utils.logger_config import main_logger as logger
except ImportError as e:
    print(f"導入模組時發生錯誤，請檢查您的 PYTHONPATH 和檔案結構: {e}")
    sys.exit(1)

def run_ga_test(task_name="amlodipine", generations=3, population_size=20, mutation_rate=0.1, crossover_rate=0.8, xai_api_key=None):
    """
    執行 graph_ga_generate 的測試函數。

    Args:
        task_name (str): 要測試的 GuacaMol 任務名稱。
        generations (int): GA 運行的代數。
        population_size (int): GA 的種群大小。
        mutation_rate (float): 突變率。
        crossover_rate (float): 交叉率。
        xai_api_key (str, optional): XAI API 金鑰。
    """
    logger.info(f"--- 開始測試 graph_ga_generate for task: {task_name} ---")

    # 1. 初始化設定 (模擬 main.py 的部分行為)
    set_task(task_name)
    logger.info(f"任務已設定為: {get_current_task()}")

    # 初始化 Oracle 預算
    update_remaining_oracle_budget(TOTAL_ORACLE_BUDGET)
    logger.info(f"初始剩餘 Oracle 預算: {get_remaining_oracle_budget()}")

    # 嘗試初始化 LLMs (如果您的 GA 或評分函數間接需要)
    # 即使 graph_ga_generate 本身不直接用，其依賴可能需要
    if xai_api_key:
        try:
            initialize_llm_with_key(xai_api_key) # For generator's LLM
            initialize_parameter_llm(xai_api_key) # For agent's parameter tuner LLM
            logger.info("LLMs 初始化嘗試完成 (使用 API 金鑰)。")
        except Exception as e:
            logger.warning(f"LLM 初始化期間發生錯誤: {e}。測試將繼續，但依賴 LLM 的功能可能受影響。")
    else:
        logger.warning("未提供 XAI API 金鑰，依賴 LLM 的功能可能無法正常運作。")


    # 2. 準備 graph_ga_generate 的參數
    # 確保參數結構與 graph_ga_generate 函數期望的一致
    # 特別是 'params' 字典用於 GA 內部參數
    ga_args = {
        "params": {
            "generations": generations, # 控制 GA 運行的代數
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            # 您可以在這裡添加更多 GB_GA_Generator 支援的參數
            # 例如: 'smi_file', 'offspring_size', 'n_jobs', etc.
            # 'results_dir': './ga_test_results' # 選擇性：指定 GA 結果目錄
        },
        "population": population_size # 這個會傳遞給 GB_GA_Generator 的 population_size
    }
    logger.info(f"傳遞給 graph_ga_generate 的參數: \n{json.dumps(ga_args, indent=2)}")

    # 3. 呼叫 graph_ga_generate
    logger.info("呼叫 graph_ga_generate...")
    try:
        history = graph_ga_generate(ga_args)
    except Exception as e:
        logger.error(f"執行 graph_ga_generate 時發生未預期的錯誤: {e}", exc_info=True)
        history = []

    # 4. 分析結果
    logger.info(f"graph_ga_generate 返回的歷史記錄長度: {len(history)}")

    if not history:
        logger.warning("graph_ga_generate 未返回任何歷史記錄。請檢查日誌以獲取詳細錯誤。")
        logger.info("可能的原因包括：評分函數初始化失敗、GA 內部錯誤、或未能生成任何有效分子。")
    else:
        logger.info("--- GA 演化歷史記錄 ---")
        for i, gen_data in enumerate(history):
            print(f"\n--- 第 {gen_data.get('generation', i + 1)} 代 ---")
            print(f"  最大分數: {gen_data.get('max_score', 'N/A')}")
            print(f"  平均分數: {gen_data.get('avg_score', 'N/A')}")
            print(f"  最小分數: {gen_data.get('min_score', 'N/A')}")
            print(f"  標準差: {gen_data.get('std_score', 'N/A')}")
            print(f"  Oracle 呼叫次數: {gen_data.get('oracle_call_count', 'N/A')}")
            print(f"  剩餘預算: {gen_data.get('remaining_budget', 'N/A')}")
            top_molecules = gen_data.get('top_molecules', [])
            if top_molecules:
                print("  該代部分高分分子:")
                for score, smiles in top_molecules[:3]: # 只顯示前3個
                    print(f"    SMILES: {smiles}, 分數: {score:.4f}")
            # 檢查 population_scores 是否存在且非空
            population_scores = gen_data.get('population_scores')
            if population_scores and isinstance(population_scores, list) and len(population_scores) > 0 :
                print(f"  該代種群分數數量: {len(population_scores)}")
            else:
                print(f"  該代種群分數: 未提供或為空")


        # 檢查分數是否有演化趨勢 (簡單檢查)
        if len(history) > 1:
            initial_max_score = history[0].get('max_score', 0)
            final_max_score = history[-1].get('max_score', 0)
            initial_avg_score = history[0].get('avg_score', 0)
            final_avg_score = history[-1].get('avg_score', 0)

            print("\n--- 演化趨勢總結 ---")
            print(f"初始最大分數: {initial_max_score}, 最終最大分數: {final_max_score}")
            print(f"初始平均分數: {initial_avg_score}, 最終平均分數: {final_avg_score}")

            if final_max_score > initial_max_score:
                logger.info("觀察到最大分數有所提升。")
            else:
                logger.info("最大分數未觀察到明顯提升。")
            if final_avg_score > initial_avg_score:
                logger.info("觀察到平均分數有所提升。")
            else:
                logger.info("平均分數未觀察到明顯提升。")

    logger.info(f"--- graph_ga_generate 測試完成 (任務: {task_name}) ---")
    return history

if __name__ == "__main__":
    # --- 設定測試參數 ---
    # 您可以修改這些參數來進行不同的測試
    TASK = "celecoxib"  # GuacaMol 任務名稱
    # TASK = "qed" # 另一個可以快速測試的任務
    # TASK = "median1" # 另一個 GuacaMol 任務
    GENERATIONS = 5      # GA 運行的代數
    POPULATION_SIZE = 100 # 種群大小
    MUTATION_RATE = 0.5
    CROSSOVER_RATE = 1
    # 在這裡填寫您的 XAI API 金鑰，如果需要的話
    XAI_API_KEY = os.getenv("CEREBRAS_API_KEY", None)

    if not XAI_API_KEY:
        print("警告: 未在環境變數 XAI_API_KEY 中找到 API 金鑰。")
        print("如果您的評分函數或 GA 的某些部分依賴 LLM，可能會出現問題。")
        # response = input("是否繼續執行 (y/n)? ").lower()
        # if response != 'y':
        #     sys.exit("測試已中止。")

    # 建立一個 GA 測試結果的目錄 (如果 GA 內部會寫入檔案)
    ga_test_output_dir = os.path.join("/home/luketou/LLM_AI_agent/Agent_predictor/GA_LLM_langchain/tests", "ga_test_results", TASK)
    os.makedirs(ga_test_output_dir, exist_ok=True)
    print(f"GA 測試的潛在輸出目錄 (如果 GA 內部使用): {ga_test_output_dir}")


    # 執行測試
    test_history = run_ga_test(
        task_name=TASK,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        xai_api_key=XAI_API_KEY
    )

    # 您可以在這裡添加更多斷言或檢查 test_history 的內容
    if test_history:
        print(f"\n測試成功完成，GA 運行了 {len(test_history)} 代。")
    else:
        print(f"\n測試完成，但 GA 未返回歷史記錄。請檢查日誌。")
