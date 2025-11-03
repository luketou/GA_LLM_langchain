from __future__ import print_function

import argparse
import heapq
import json
import os
import random
import csv
from time import time
from datetime import datetime
from typing import List, Optional, Callable, Tuple, Union, Dict
import threading

import joblib
import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from guacamol.utils.helpers import setup_default_logger
from joblib import delayed      
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from agent import get_remaining_oracle_budget, get_total_oracle_budget
import sys
import os
# 添加父目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from graph_ga.crossover import crossover
from graph_ga.mutate import mutate as mu
import logging

logger = logging.getLogger(__name__)

# Define the total oracle call budget
TOTAL_ORACLE_BUDGET = get_total_oracle_budget() # Remains as a default

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # Handle edge cases where scores might be invalid
    # Clean up any NaN or negative values in scores
    population_scores = np.array(population_scores)
    # Replace NaN or negative values with zero
    population_scores = np.nan_to_num(population_scores, nan=0.0, neginf=0.0)
    # Ensure all scores are non-negative
    population_scores = np.maximum(population_scores, 0.0)
    
    # If all scores are zero or very small, use uniform distribution
    sum_scores = population_scores.sum()
    if (sum_scores <= 1e-10):
        print("Warning: All scores are zero or very small. Using uniform distribution.")
        population_probs = np.ones(len(population_mol)) / len(population_mol)
    else:
        # scores -> probs
        population_probs = population_scores / sum_scores
    
    # Sanity check for valid probabilities
    if not np.all(np.isfinite(population_probs)):
        print("Warning: Invalid probabilities detected. Using uniform distribution.")
        population_probs = np.ones(len(population_mol)) / len(population_mol)
    
    # Make sure probabilities sum to 1
    population_probs = population_probs / population_probs.sum()
    
    # 確保 offspring_size 為正整數，若不合法則回退到種群大小
    try:
        size = int(offspring_size)
    except Exception:
        size = len(population_mol)
    if size <= 0:
        size = len(population_mol)
    
    try:
        mating_pool = np.random.choice(population_mol,
                                       p=population_probs,
                                       size=size,
                                       replace=True)
        return mating_pool
    except ValueError as e:
        print(f"Error in mating pool selection: {e}")
        # Fallback: just select randomly
        return np.random.choice(population_mol,
                                size=size,
                                replace=True)

def reproduce(mating_pool, mutation_rate, crossover_rate=0.5):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
        crossover_rate: rate of crossover
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = crossover(parent_a, parent_b, crossover_rate)
    if new_child is not None:
        new_child = mu(new_child, mutation_rate)
    return new_child

def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))

def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population

# Function to score molecules without incrementing oracle call counter
def score_without_counting(smiles, scoring_function):
    """不計入預算的評分函數"""
    try:
        # 針對 GoalDirectedBenchmark 物件使用 .score_molecule() 方法
        if hasattr(scoring_function, 'score_molecule'):
            return scoring_function.score_molecule(smiles)
        # 針對 ScoringFunction 物件使用 .score() 方法
        elif hasattr(scoring_function, 'score'):
            return scoring_function.score(smiles)
        # 針對可直接調用的函數
        elif callable(scoring_function):
            return scoring_function(smiles)
        # 針對其他特殊情況
        elif hasattr(scoring_function, 'raw_scoring_function'):
            return scoring_function.raw_scoring_function(smiles)
        else:
            logger.warning(f"無法評分分子，未知的評分函數類型: {type(scoring_function)}")
            return 0.0
    except Exception as e:
        logger.error(f"評分錯誤: {str(e)}")
        return 0.0

# Add a thread-safe counter class
class ThreadSafeCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = threading.Lock()
        
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value
            
    def get(self):
        with self.lock:
            return self.value
    
    def remaining(self, budget):
        with self.lock:
            return max(0, budget - self.value)
    
    def reset(self, new_value=0):
        """重置計數器到指定值"""
        with self.lock:
            old_value = self.value
            self.value = new_value
            return old_value
            
    def check_and_increment(self, budget):
        """原子性地檢查是否超出預算並增加計數（如果未超出）"""
        with self.lock:
            if self.value >= budget:
                # 已經達到預算，不增加計數
                return False, self.value
            else:
                # 未達到預算，增加計數
                self.value += 1
                return True, self.value

class GB_GA_Generator(GoalDirectedGenerator):
    """
    Graph-based GA Generator for molecular optimization
    """
    def __init__(self, smi_file, population_size, offspring_size, generations, mutation_rate, crossover_rate=0.5, 
                 n_jobs=1, random_start=False, top_from_random=False, random_sample_size=10000, 
                 use_lowest_scoring=True, patience=5, llm_agent=None, llm_frequency=5, return_initial_population=False,
                 oracle_budget=None, dynamic_offspring_size=False, batch_size=1000, early_stopping=False,
                 oracle_counter=None, thread_safe=False, results_dir=None, task_name=None):
        """
        Initialize GB_GA_Generator
        
        Args:
            smi_file: Path to SMILES file with starting molecules
            population_size: Size of population (default 100)
            offspring_size: Size of offspring (default 200)
            generations: Number of generations
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate (default 0.5)
            n_jobs: Number of parallel jobs (default 1, use 1 to avoid pickling issues)
            random_start: Whether to randomly select initial population
            top_from_random: Whether to select molecules from a random subset based on scoring
            random_sample_size: Size of random sample if top_from_random is True (default 10000)
            use_lowest_scoring: Whether to select lowest scoring molecules instead of highest scoring
            patience: Early stopping patience
            llm_agent: LLM agent for parameter optimization
            llm_frequency: How often LLM agent intervenes
            return_initial_population: Whether to return initial population info
            oracle_budget: Total budget for oracle calls (default 5000)
            dynamic_offspring_size: Whether to allow dynamic control of offspring size
            batch_size: Size of batches for processing molecules during initialization
            early_stopping: Whether to enable early stopping during initial population selection
            oracle_counter: 外部提供的共享計數器 (用於線程安全)
            thread_safe: 是否啟用線程安全模式
            results_dir: 結果目錄，用於保存 CSV 記錄
            task_name: 任務名稱，用於命名 CSV 文件
        """
        # Force n_jobs to 1 to avoid pickling issues with the scoring function
        self.n_jobs = n_jobs
        self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_start = random_start
        self.top_from_random = top_from_random  # New parameter
        self.random_sample_size = random_sample_size  # New parameter
        self.use_lowest_scoring = use_lowest_scoring  # New parameter
        self.patience = patience
        # LLM相關參數
        self.llm_agent = llm_agent
        self.llm_frequency = llm_frequency  # 每幾個generation呼叫一次LLM
        self.generation_history = []  # 保存每個generation的結果歷史
        self.return_initial_population = return_initial_population  # Whether to return initial population info
        self.initial_population = []  # Store initial population SMILES
        self.initial_scores = []  # Store initial population scores
        self.dynamic_offspring_size = dynamic_offspring_size  # Whether to allow dynamic control of offspring size
        self.batch_size = batch_size  # Batch size for processing molecules during initialization
        self.early_stopping = early_stopping  # Enable early stopping during initial population selection
        
        # Add oracle call tracking
        self.oracle_tracking = []  # List to track all oracle calls
        # self.oracle_call_count is for local tracking of calls made by this GA instance during evolution
        # self.initial_oracle_calls tracks calls for initial population (should be 0 if score_without_counting is used)
        self.initial_oracle_calls = 0  # Assuming initial population scoring does not use budgeted calls

        # Add oracle budget tracking
        if oracle_budget is None:
            self.instance_oracle_budget_reference = get_total_oracle_budget() # Get current total budget from agent as a reference for this instance
        else:
            self.instance_oracle_budget_reference = oracle_budget # Use provided budget as a reference
        
        # Variable to track if we reached the oracle budget (based on agent's report)
        self.oracle_budget_reached = False
        
        # 支持外部提供計數器（用於線程安全）
        if oracle_counter is not None:
            self.oracle_call_count = oracle_counter
        else:
            self.oracle_call_count = ThreadSafeCounter(0)
            
        self.thread_safe = thread_safe
        
        # GPU 相關優化設置
        if n_jobs > 1:
            # 配置 joblib 使用 threading 後端以避免 pickling 問題
            self.pool = joblib.Parallel(n_jobs=n_jobs, backend="threading")
            print(f"Using threading backend with {n_jobs} workers")
        else:
            self.pool = joblib.Parallel(n_jobs=1)
        
        # 新增結果目錄和任務名稱的屬性
        self.results_dir = results_dir
        self.task_name = task_name
        
        # 初始化代數計數
        self.generation_count = 0
        
        # 初始化 CSV 記錄文件
        self.csv_file_path = self._initialize_generation_csv() if results_dir else None
        
        self.population_mol: List[Optional[Mol]] = []
        self.population_scores: List[float] = []
        
    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            # Use a simple list comprehension instead of parallel processing
            return [canonicalize(s.strip()) for s in f]
            
    def top_k(self, smiles, scoring_function, k):
        # Score molecules sequentially to avoid pickling issues
        scores = [scoring_function.score(s) for s in smiles]
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]
        
    def save_initial_population(self, smiles_list, scores_list, results_dir=None):
        """
        Save the initial population to a CSV file with their oracle scores
        
        Args:
            smiles_list: List of SMILES strings in the initial population
            scores_list: List of corresponding scores
            results_dir: Directory to save the file (will use cwd if None)
        """
        if results_dir is None:
            results_dir = os.path.join(parent_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the CSV file path
        csv_path = os.path.join(results_dir, f'initial_population_{timestamp}.csv')
        
        # Write the data to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles', 'oracle_score'])
            for smiles, score in zip(smiles_list, scores_list):
                writer.writerow([smiles, score])
                
        print(f"Saved initial population with {len(smiles_list)} molecules to {csv_path}")
        return csv_path
        
    def _initialize_generation_csv(self):
        """初始化 CSV 文件用於實時記錄每代的統計數據"""
        if not self.results_dir:
            print("No results directory provided, CSV logging disabled")
            return None
        
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_suffix = f"{self.task_name}_" if self.task_name else ""
        csv_path = os.path.join(self.results_dir, f'{task_suffix}generation_stats_{timestamp}.csv')
        
        # CSV 欄位定義
        fieldnames = [
            'generation', 'timestamp', 'source',
            'max_score', 'avg_score', 'min_score', 'std_score',
            'oracle_calls', 'cumulative_oracle_calls',
            'mutation_rate', 'offspring_size', 'crossover_rate',
            'remaining_budget', 'notes'
        ]
        
        # 創建 CSV 文件並寫入標題行
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
        print(f"Initialized generation statistics CSV: {csv_path}")
        return csv_path
        
    def log_generation_stats(self, population_scores, oracle_calls, source="GA", 
                           mutation_rate=None, offspring_size=None, crossover_rate=None,
                           remaining_budget=None, notes=""):
        """記錄一代的統計數據到 CSV"""
        if not self.csv_file_path or not population_scores:
            return
        
        # 使用實例成員變數作為默認值，如果沒有提供參數值
        mutation_rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        offspring_size = offspring_size if offspring_size is not None else self.offspring_size
        crossover_rate = crossover_rate if crossover_rate is not None else self.crossover_rate
        
        # 計算統計數據
        max_score = np.max(population_scores) if population_scores else 0.0
        avg_score = np.mean(population_scores) if population_scores else 0.0
        min_score = np.min(population_scores) if population_scores else 0.0
        std_score = np.std(population_scores) if population_scores and len(population_scores) > 1 else 0.0
        
        # local_cumulative_calls: cumulative calls made by THIS GA instance's evolution phase
        local_cumulative_calls_evolution = self.oracle_call_count.get() 
        
        # For "Initial" source, oracle_calls is self.initial_oracle_calls (expected to be 0)
        # and cumulative can also be self.initial_oracle_calls.
        log_cumulative_calls = local_cumulative_calls_evolution
        if source == "Initial":
            log_cumulative_calls = self.initial_oracle_calls # Should be 0

        # 如果沒有提供剩餘預算，則從 agent 獲取最新的剩餘預算
        if remaining_budget is None:
            current_agent_remaining_budget = get_remaining_oracle_budget()
        else:
            current_agent_remaining_budget = remaining_budget # Use provided if available
            
        # 建立記錄項
        record = {
            'generation': self.generation_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': source,
            'max_score': f"{max_score:.6f}",
            'avg_score': f"{avg_score:.6f}",
            'min_score': f"{min_score:.6f}",
            'std_score': f"{std_score:.6f}",
            'oracle_calls': oracle_calls, # Oracle calls in this specific generation/step by this source
            'cumulative_oracle_calls': log_cumulative_calls, # Cumulative calls by this GA instance for evolution/initial
            'mutation_rate': f"{mutation_rate:.4f}",
            'offspring_size': offspring_size,
            'crossover_rate': f"{crossover_rate:.4f}",
            'remaining_budget': current_agent_remaining_budget, # Agent's current remaining budget
            'notes': notes
        }
        
        # 使用線程安全的文件寫入（如果啟用）
        if self.thread_safe:
            import threading
            file_lock = threading.Lock()
            with file_lock, open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                writer.writerow(record)
        else:
            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                writer.writerow(record)
                    
        # 增加生成計數
        self.generation_count += 1
        
        # 控制台輸出
        print(f"Gen {self.generation_count-1} ({source}): "
             f"max={max_score:.4f}, avg={avg_score:.4f}, min={min_score:.4f}, "
             f"calls={oracle_calls}, total={log_cumulative_calls}, "
             f"mut_rate={mutation_rate:.4f}, remain={record['remaining_budget']}")
                 
    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None,
                                     results_dir: Optional[str] = None) -> Union[List[str], Tuple[List[str], List[str], List[float]]]:
        """
        Generate optimized molecules using genetic algorithm
            
        Args:
            scoring_function: Function to score molecules
            number_molecules: Number of molecules to return
            starting_population: Optional starting population
            results_dir: Directory to save initial population (default None)
        Returns:
            If self.return_initial_population is True:
                Tuple of (optimized_molecules, initial_population, initial_scores)
            Otherwise:
                List of optimized molecules
        """
        requested_total = number_molecules  # Fix: ensure requested_total is defined
        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')
            
        # Setup oracle tracking directory
        oracle_tracking_dir = os.path.join(parent_dir, 'oracle_calls_tracking')
        os.makedirs(oracle_tracking_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        oracle_tracking_file = os.path.join(oracle_tracking_dir, f'oracle_tracking_{timestamp}.csv')
        
        # 在線程安全模式下使用文件鎖避免競態條件
        if self.thread_safe:
            import threading
            file_lock = threading.Lock()
            
            # 創建 CSV 文件頭
            with file_lock, open(oracle_tracking_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['call_number', 'timestamp', 'smiles', 'score', 'source', 'population_phase', 'remaining_budget'])
        else:
            # 標準單線程模式
            with open(oracle_tracking_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['call_number', 'timestamp', 'smiles', 'score', 'source', 'population_phase', 'remaining_budget'])
                
        # 更新結果目錄（如果提供）
        if results_dir:
            self.results_dir = results_dir
            # 重新初始化 CSV 文件（如果需要）
            if self.csv_file_path is None:
                self.csv_file_path = self._initialize_generation_csv()

        # Reset budget reached flag
        self.oracle_budget_reached = False
        
        # 添加 debug 日誌，記錄初始預算和計數器狀態
        print(f"Starting optimization. Agent total budget: {get_total_oracle_budget()}, Agent remaining: {get_remaining_oracle_budget()}. Local GA counter: {self.oracle_call_count.get()}")
        
        # Create wrapped scoring function to track oracle calls with thread safety
        def tracked_scoring_function(smiles, source="graph_ga", phase="initialization"):
            """
            Wraps the scoring function to track oracle calls.
            Budget checks are now primarily against get_remaining_oracle_budget().
            """
            current_call_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
            score = 0.0
            local_call_number_for_log = self.oracle_call_count.get() + 1 # Tentative local call number for logging

            if phase == "evolution":
                agent_remaining_before_score = get_remaining_oracle_budget()
                if agent_remaining_before_score <= 0:
                    if not self.oracle_budget_reached:
                        print(f"Warning: Oracle call budget (from agent) is zero or less ({agent_remaining_before_score}) before scoring in phase '{phase}'. Skipping scoring for {smiles}.")
                        self.oracle_budget_reached = True
                    # Log this attempt with score 0 and current agent budget
                    # Do not increment local evolution counter if actual scoring is skipped.
                    # However, the CSV log entry structure expects a call number.
                    # For skipped calls, we might log with a special score or note.
                    # For now, return 0.0, local counter not incremented here for this path.
                    
                    # Log the denied attempt
                    denied_call_record = {
                        'call_number': f"denied_{local_call_number_for_log}",
                        'timestamp': current_call_timestamp, 'smiles': smiles, 'score': 0.0, 'source': source, 'phase': phase,
                        'remaining_budget': agent_remaining_before_score
                    }
                    self.oracle_tracking.append(denied_call_record)
                    # Log to file
                    log_row_values = [denied_call_record['call_number'], current_call_timestamp, smiles, 0.0, source, phase, agent_remaining_before_score]
                    if self.thread_safe and hasattr(self, 'oracle_file_lock') and self.oracle_file_lock:
                        with self.oracle_file_lock:
                            with open(oracle_tracking_file, 'a', newline='') as f_lock:
                                csv.writer(f_lock).writerow(log_row_values)
                    else:
                        with open(oracle_tracking_file, 'a', newline='') as f_no_lock:
                            csv.writer(f_no_lock).writerow(log_row_values)
                    return 0.0 # Return 0 as scoring was not performed due to budget

            # If phase is "initialization", or "evolution" and agent has budget (checked above for evolution)
            
            # Increment local counter (ThreadSafeCounter)
            # For "initialization", this is informational and reset before evolution.
            # For "evolution", this tracks calls made by this GA instance.
            actual_local_call_number = self.oracle_call_count.increment()

            try:
                if hasattr(scoring_function, 'score_molecule'):
                    score = scoring_function.score_molecule(smiles)
                elif hasattr(scoring_function, 'score'):
                    score = scoring_function.score(smiles)
                else:
                    score = scoring_function(smiles)
                # Assumes successful execution of scoring_function implies agent's budget was consumed by it.
            except Exception as e:
                print(f"Error scoring molecule {smiles}: {e}")
                score = 0.0 
                # If scoring fails, agent budget consumption depends on agent's internal logic.
                # We assume an attempt was made and budget might be consumed.

            # Get the LATEST remaining budget from agent AFTER the scoring attempt
            final_agent_remaining_budget_after_score = get_remaining_oracle_budget()

            if phase == "evolution":
                if final_agent_remaining_budget_after_score <= 0:
                    if not self.oracle_budget_reached: # Avoid redundant logging
                         print(f"Oracle budget (from agent) became zero or less ({final_agent_remaining_budget_after_score}) after scoring {smiles} in phase '{phase}'.")
                    self.oracle_budget_reached = True

            call_record = {
                'call_number': actual_local_call_number, # Actual incremented local counter
                'timestamp': current_call_timestamp,
                'smiles': smiles,
                'score': score,
                'source': source,
                'phase': phase,
                'remaining_budget': final_agent_remaining_budget_after_score
            }
            
            self.oracle_tracking.append(call_record)
            
            log_row_values_success = [actual_local_call_number, current_call_timestamp, smiles, score, source, phase, final_agent_remaining_budget_after_score]
            if self.thread_safe and hasattr(self, 'oracle_file_lock') and self.oracle_file_lock:
                with self.oracle_file_lock:
                    with open(oracle_tracking_file, 'a', newline='') as f_lock:
                        csv.writer(f_lock).writerow(log_row_values_success)
            else:
                 with open(oracle_tracking_file, 'a', newline='') as f_no_lock:
                    csv.writer(f_no_lock).writerow(log_row_values_success)
            
            return score

        # Clear initial population data
        self.initial_population = []
        self.initial_scores = []
        self.oracle_tracking = []  # Reset oracle tracking
        self.oracle_call_count = ThreadSafeCounter(0)  # 重置計數器            

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            # Randomly sample molecules without replacement
            # New option: Select molecules from a random subset
            if self.top_from_random:
                print(f'Randomly sampling {self.random_sample_size} molecules and selecting {"lowest" if self.use_lowest_scoring else "highest"} scoring molecules...')
                
                # Randomly sample molecules without replacement
                if len(self.all_smiles) > self.random_sample_size:
                    random_sample = np.random.choice(self.all_smiles, self.random_sample_size, replace=False)
                else:
                    random_sample = self.all_smiles
                    print(f'Warning: Requested sample size ({self.random_sample_size}) is larger than available molecules ({len(self.all_smiles)})')
                
                # Score molecules with batch processing and early stopping
                print(f'Using efficient batch processing to find {"lowest" if self.use_lowest_scoring else "highest"} scoring molecules...')
                starting_population = self._select_molecules_with_batching(
                    scoring_function,
                    random_sample, 
                    use_lowest_scoring=self.use_lowest_scoring,
                    count=self.population_size,
                    batch_size=min(1000, len(random_sample))
                )
                
                if starting_population:
                    print(f'Initial population selected with efficient batching')
                else:
                    print(f'Warning: Failed to select initial population with batching, falling back to standard method')
                    # If batching failed, fall back to standard method
                    scores = [score_without_counting(s, scoring_function) for s in random_sample]
                    scored_smiles = sorted(zip(scores, random_sample), key=lambda x: x[0], reverse=not self.use_lowest_scoring)
                    starting_population = [s for _, s in scored_smiles[:self.population_size]]
                    
            # Standard random selection
            elif self.random_start:
                starting_population = np.random.choice(self.all_smiles, self.population_size)
                print('Random initial population selected')
                
            # Standard evaluation of all molecules
            else:
                print('Evaluating all molecules WITHOUT tracking oracle calls...')
                starting_population = self._select_molecules_with_batching(
                    scoring_function,
                    self.all_smiles, 
                    use_lowest_scoring=self.use_lowest_scoring,
                    count=self.population_size,
                    batch_size=min(1000, len(self.all_smiles))
                )
                if not starting_population:
                    print(f'Warning: Failed to select initial population with batching, falling back to standard method')
                    # Score molecules without batching as fallback            
                    if self.n_jobs > 1:
                        scores = self.pool(delayed(score_without_counting)(s, scoring_function) for s in self.all_smiles)
                    else:
                        scores = [score_without_counting(s, scoring_function) for s in self.all_smiles]
                    # Create (score, SMILES) pairs, sort, and take top or bottom population_size
                    scored_smiles = sorted(zip(scores, self.all_smiles), key=lambda x: x[0], reverse=not self.use_lowest_scoring)
                    starting_population = [s for _, s in scored_smiles[:self.population_size]]

        # Select initial population and score them for storage using the tracked function
        # This scores don't count against the budget
        initial_population_scores = []
        scored_population = []
        
        for s in starting_population:
            # Score without counting against the budget
            score = score_without_counting(s, scoring_function)
            initial_population_scores.append(score)
            scored_population.append((score, s))
        # Sort by score (lowest or highest)
        scored_population.sort(key=lambda x: x[0], reverse=not self.use_lowest_scoring)
        
        # Store the final initial population
        self.initial_population = [s for _, s in scored_population[:self.population_size]]
        self.initial_scores = [score for score, _ in scored_population[:self.population_size]]
        
        # Save the initial population to CSV
        population_csv = self.save_initial_population(self.initial_population, self.initial_scores, results_dir)
        print(f"Initial population statistics:")
        print(f"  - Lowest score: {min(self.initial_scores):.4f}")
        print(f"  - Highest score: {max(self.initial_scores):.4f}")
        print(f"  - Average score: {sum(self.initial_scores)/len(self.initial_scores):.4f}")
        
        # Store the number of oracle calls used for initialization
        self.initial_oracle_calls = 0 # Explicitly set to 0, as score_without_counting is used.
        print(f"Oracle calls used for initial population selection (tracked locally, should be 0): {self.initial_oracle_calls}")
        
        # print(f"Starting evolution with oracle budget of {self.oracle_budget}") # self.oracle_budget is instance reference
        print(f"Starting evolution. Agent's current total budget: {get_total_oracle_budget()}, Agent's remaining budget: {get_remaining_oracle_budget()}")
        
        # Reset the oracle call counter for the evolution phase
        # self.oracle_call_count is ThreadSafeCounter for local calls in evolution
        old_local_count_before_reset = self.oracle_call_count.get()
        self.oracle_call_count.reset(0) 
        print(f"Resetting local oracle call counter from {old_local_count_before_reset} to {self.oracle_call_count.get()} for evolution phase.")
        
        # Convert SMILES to molecules
        population_mol = [Chem.MolFromSmiles(s) for s in self.initial_population]
        
        # Use tracked scoring function for evolution phase
        def score_mol_tracked(mol, source="GA"):
            """追蹤分子評分的函數，計入預算消耗"""
            smiles = Chem.MolToSmiles(mol)
            return tracked_scoring_function(smiles, source=source, phase="evolution")
        
        # Batch processing and GPU optimization
        def batch_score_mol_tracked(molecules, source="graph_ga"):
            """批量評分分子以提高 GPU 利用率"""
            results = []
            for mol in molecules:
                results.append(score_mol_tracked(mol, source))
            return results
            
        # 定義並行交叉和變異的函數
        def parallel_reproduce_batch(batch_size, mating_pool, mutation_rate, crossover_rate):
            """並行處理一批交叉和變異操作"""
            results = []
            for _ in range(batch_size):
                offspring = reproduce(mating_pool, mutation_rate, crossover_rate)
                if offspring is not None:
                    results.append(offspring)
            return results
        
        # Score population with more efficient batching
        if self.n_jobs > 1:
            # 將分子分成批次以便更有效地利用 GPU
            batch_size = min(100, len(population_mol))  # 較小的批次以平衡 GPU 利用率和 Oracle 計數
            batches = [population_mol[i:i+batch_size] for i in range(0, len(population_mol), batch_size)]
            # 並行處理批次
            population_scores_batches = self.pool(
                delayed(batch_score_mol_tracked)(batch) for batch in batches
            )
            population_scores = [score for batch in population_scores_batches for score in batch]
        else:
            population_scores = [score_mol_tracked(m) for m in population_mol]
        
        # evolution: go go go!!
        t0 = time()
        patience = 0
        # 清空generation歷史記錄
        self.generation_history = []
        
        # Track current offspring size, which can be dynamically adjusted by LLM
        current_offspring_size = self.offspring_size
        
        # 當評估初始種群時，記錄統計數據
        # 更改這行代碼後的 population_scores 計算邏輯：
        # 評估種群時
        if self.csv_file_path:
            # 記錄初始種群統計數據
            self.log_generation_stats(
                population_scores=self.initial_scores,
                oracle_calls=self.initial_oracle_calls,
                source="Initial",
                notes="Initial population evaluation"
            )
        
        self.generation_history = [] # clear

        for generation in range(self.generations):
            # Check agent's budget at the start of each generation
            current_agent_remaining_budget = get_remaining_oracle_budget()
            if current_agent_remaining_budget <= 0:
                print(f"Oracle budget (from agent) is zero or less ({current_agent_remaining_budget}) at the start of generation {generation}. Stopping evolution.")
                self.oracle_budget_reached = True 
                break 

            if self.oracle_budget_reached: # This flag is set by tracked_scoring_function or here
                print(f"Oracle budget_reached flag is true at generation {generation} (agent remaining: {current_agent_remaining_budget}). Stopping evolution.")
                break
            
            # 保存當前世代信息
            current_gen_info = {
                'generation': generation,
                'population_scores': population_scores.copy(),
                'max_score': np.max(population_scores),
                'avg_score': np.mean(population_scores),
                'min_score': np.min(population_scores),
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'oracle_call_count': self.oracle_call_count.get(),  # Only evolution phase calls
                'remaining_budget': current_agent_remaining_budget,
                'offspring_size': current_offspring_size  # Track current offspring size
            }
            self.generation_history.append(current_gen_info)

            if self.llm_agent is not None and generation > 0 and generation % self.llm_frequency == 0:
                # LLM intervention might consume oracle calls if it generates and scores molecules.
                # The tracked_scoring_function (if used by LLM agent internally) will handle budget.
                print(f"Generation {generation}: LLM agent intervention. Agent remaining budget before LLM: {get_remaining_oracle_budget()}")
                try:
                    params = self.llm_agent.suggest_parameters(self.generation_history)
                    if self.dynamic_offspring_size:
                        total_req = params.get('num_molecules', self.offspring_size)
                        llm_req = params.get('llm_molecules', min(50, total_req//2))
                        llm_count = max(10, min(200, llm_req))  # enforce LLM gen count range
                        ga_req = max(0, total_req - llm_count)
                        ga_req = max(10, min(200, ga_req))      # enforce GA gen count range
                        current_offspring_size = ga_req
                    else:
                        temp_llm = params.get('llm_molecules', self.offspring_size)
                        llm_count = max(10, min(200, temp_llm))
                        current_offspring_size = max(10, min(200, self.offspring_size))
                except Exception as e:
                    print(f"Error during LLM intervention: {e}")

            # Check again if budget is exceeded after LLM intervention
            if self.oracle_call_count.get() >= self.instance_oracle_budget_reference:
                print(f"Oracle call budget exceeded during generation {generation}. Stopping early.")
                break
            
            # Check agent's budget again after potential LLM actions
            agent_remaining_after_llm = get_remaining_oracle_budget()
            if agent_remaining_after_llm <= 0:
                print(f"Oracle call budget (from agent) is zero or less ({agent_remaining_after_llm}) after LLM intervention in generation {generation}. Stopping early.")
                self.oracle_budget_reached = True
                break
            
            # Calculate new remaining budget (redundant if using get_remaining_oracle_budget() directly)
            # remaining_budget = self.oracle_budget - self.oracle_call_count.get() # Old local logic
            
            # Only proceed with GA reproduction/scoring if agent has budget remaining
            if get_remaining_oracle_budget() <= 0:
                print("No remaining oracle call budget (from agent) before GA reproduction/scoring. Stopping GA evolution.")
                self.oracle_budget_reached = True
                break

            # Ensure we generate exactly the requested number of offsprings
            offspring_count = min(current_offspring_size, self.population_size)
            mating_pool = make_mating_pool(population_mol, population_scores, offspring_count)
            
            offspring_mol_list = []

            # Process offspring
            if self.n_jobs > 1:
                batch_size = min(self.batch_size, offspring_count)
                n_batches = (offspring_count + batch_size - 1) // batch_size
                offspring_batches = self.pool(
                    delayed(parallel_reproduce_batch)(
                        (batch_size, offspring_count - i*batch_size),
                        mating_pool, self.mutation_rate, self.crossover_rate
                    ) for i in range(n_batches)
                )
                # 合併結果
                offspring_mol = []
                for batch in offspring_batches:
                    offspring_mol.extend(batch)
            else:
                # Use current_offspring_size instead of fixed population_size
                offspring_mol = []
                for _ in range(offspring_count):
                    offspring = reproduce(mating_pool, self.mutation_rate, self.crossover_rate)
                    if offspring is not None:
                        offspring_mol.append(offspring)

            # add new_population
            offspring_mol.append(offspring)

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)
            # stats
            gen_time = time() - t0
            mol_sec = offspring_count / gen_time if gen_time > 0 else 0
            t0 = time()
            old_scores = population_scores

            # Check if agent has enough budget for scoring the current population_mol
            # This is a general check; tracked_scoring_function will do per-molecule check.
            num_molecules_to_score_this_gen = len(population_mol)
            agent_budget_before_scoring_this_gen_batch = get_remaining_oracle_budget()

            if agent_budget_before_scoring_this_gen_batch <= 0:
                 print(f"No agent budget ({agent_budget_before_scoring_this_gen_batch}) to score any molecules in generation {generation} batch. Stopping.")
                 self.oracle_budget_reached = True
                 break 
            elif agent_budget_before_scoring_this_gen_batch < num_molecules_to_score_this_gen:
                print(f"Warning: Agent budget ({agent_budget_before_scoring_this_gen_batch}) is less than molecules to score ({num_molecules_to_score_this_gen}). "
                      f"Scoring will proceed, but `tracked_scoring_function` will deny calls beyond budget.")
            
            # Score the population
            # tracked_scoring_function internally checks agent budget for each call during "evolution" phase
            if self.n_jobs > 1:
                population_scores = self.pool(delayed(score_mol_tracked)(m) for m in population_mol)
            else:
                population_scores = [score_mol_tracked(m) for m in population_mol]

            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            # Show evolution oracle calls and budget information
            evolution_calls_local_this_instance = self.oracle_call_count.get() # Local calls by this GA instance in evolution
            current_agent_remaining_budget_for_log = get_remaining_oracle_budget() # Get latest from agent
            print(f'{generation} | '
                  f'max: {np.max(population_scores) if population_scores else 0:.3f} | '
                  f'avg: {np.mean(population_scores) if population_scores else 0:.3f} | '
                  f'min: {np.min(population_scores) if population_scores else 0:.3f} | '
                  f'std: {np.std(population_scores) if population_scores and len(population_scores) > 1 else 0:.3f} | '
                  f'local GA evol. calls: {evolution_calls_local_this_instance} | '
                  f'agent rem. budget: {current_agent_remaining_budget_for_log} | '
                  f'offspring size: {current_offspring_size} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')
            logger.info(f'{generation} | '
                        f'max: {np.max(population_scores) if population_scores else 0:.3f} | '
                        f'avg: {np.mean(population_scores) if population_scores else 0:.3f} | '
                        f'min: {np.min(population_scores) if population_scores else 0:.3f} | '
                        f'std: {np.std(population_scores) if population_scores and len(population_scores) > 1 else 0:.3f} | '
                        f'local GA evol. calls: {evolution_calls_local_this_instance} | '
                        f'agent rem. budget: {current_agent_remaining_budget_for_log} | '
                        f'offspring size: {current_offspring_size} | '
                        f'{gen_time:.2f} sec/gen | '
                        f'{mol_sec:.2f} mol/sec')
            
            
            # 記錄 GA 代數的統計數據
            oracle_calls_in_generation = self.oracle_call_count.get() - previous_call_count if 'previous_call_count' in locals() else 0
            previous_call_count = self.oracle_call_count.get()
            
            self.log_generation_stats(
                population_scores=population_scores if population_scores else [0.0],
                oracle_calls=oracle_calls_in_generation, # Calls in this specific generation by GA
                source="GA",
                mutation_rate=self.mutation_rate,
                offspring_size=current_offspring_size,
                crossover_rate=self.crossover_rate,
                remaining_budget=current_agent_remaining_budget_for_log, # Pass current agent budget
                notes=f"Generation {generation}"
            )

            # Combine parents and offspring
            current_population_mol = population_mol + offspring_mol
            current_population_mol = sanitize(current_population_mol)

            # Score the combined population
            current_population_scores = [score_mol_tracked(m) for m in current_population_mol]

            # Sort by score and select the best molecule
            all_scored_tuples = sorted(zip(current_population_scores, current_population_mol), key=lambda x: x[0], reverse=True)
            best_mol = all_scored_tuples[0][1] if all_scored_tuples else None
            best_smiles = Chem.MolToSmiles(best_mol) if best_mol else None
            max_score = all_scored_tuples[0][0] if all_scored_tuples else 0.0


            # 收集所有後代分子及其分數
            evaluated_offspring = []
            for mol, score in zip(offspring_mol, population_scores):
                try:
                    smiles = Chem.MolToSmiles(mol)
                    evaluated_offspring.append({"SMILES": smiles, "score": score})
                except Exception as e:
                    logger.warning(f"Failed to convert molecule to SMILES: {e}")

            # Log generation stats
            self.generation_history.append({
                'generation': generation,
                'population_scores': [s for s, _ in all_scored_tuples],
                'max_score': max_score,
                'best_smiles': best_smiles,
                'remaining_budget': self.instance_oracle_budget_reference - self.oracle_call_count.get(),
                'population_smiles': [Chem.MolToSmiles(mol) for mol in population_mol],
                'evaluated_offspring': evaluated_offspring  # 新增：記錄當代所有評估過的後代
            })

            # Select the next generation
            population_mol = [t[1] for t in all_scored_tuples[:self.population_size]]
            population_scores = [t[0] for t in all_scored_tuples[:self.population_size]]

            # Update generation_history with agent's remaining budget
            self.generation_history[-1]['remaining_budget'] = get_remaining_oracle_budget() # Update with latest
            self.generation_history[-1]['oracle_call_count'] = self.oracle_call_count.get() # local evolution calls

        # Get optimized molecules
        optimized_molecules = [Chem.MolToSmiles(m) for m in population_mol][:requested_total]
        if len(optimized_molecules) != requested_total:
            print(f"Requested {requested_total} molecules but generated {len(optimized_molecules)}")
            
        # Print final oracle call count - separately for initialization and evolution
        print(f"Initial population oracle calls (local GA tracking, should be 0): {self.initial_oracle_calls}")
        evolution_calls_local_total = self.oracle_call_count.get() # Total local calls by this GA in evolution phase
        print(f"Evolution phase oracle calls (tracked locally by this GA instance): {evolution_calls_local_total}")
        
        final_agent_total_budget = get_total_oracle_budget()
        final_agent_remaining_budget = get_remaining_oracle_budget()
        print(f"Agent's reported total budget at end: {final_agent_total_budget}")
        print(f"Agent's reported final remaining budget: {final_agent_remaining_budget}")
        # print(f"Budget limit: {self.oracle_budget}, Unused budget: {max(0, self.oracle_budget - evolution_calls)}") # Old logic
        print(f"Local GA instance budget_reached flag: {self.oracle_budget_reached}")
        print(f"Oracle call tracking saved to {oracle_tracking_file}")
        print(f"Initial population saved to {population_csv}")
        
        # Save generation history to CSV
        if self.generation_history:
            # Extract task name from the scoring function if available
            task_name = getattr(scoring_function, 'descriptor', 'unknown_task')
            if task_name == 'unknown_task' and hasattr(scoring_function, '__class__'):
                task_name = scoring_function.__class__.__name__
            history_csv = self.save_generation_history(task_name, results_dir)
            print(f"Generation history saved to {history_csv}")
        
        # Return based on whether initial population info is requested
        if self.return_initial_population:
            return optimized_molecules, self.initial_population, self.initial_scores
        else:
            return optimized_molecules

    def save_generation_history(self, task_name, results_dir=None):
        """
        Save generation history to a CSV file with generation-level metrics
        
        Args:
            task_name: Name of the task/experiment
            results_dir: Directory to save the file (optional)
            
        Returns:
            Path to the saved CSV file
        """
        if not self.generation_history:
            print("No generation history to save")
            return None
        
        # Determine directory path
        if results_dir is None:
            # Determine parent directory (up one level from script location)
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(parent_dir, 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create CSV filename with task name and timestamp
        csv_path = os.path.join(results_dir, f'{task_name}_generation_history_{timestamp}.csv')
        
        # Get headers from the first item in generation history
        sample_generation = self.generation_history[0]
        headers = list(sample_generation.keys())
        
        # Save data to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for generation_data in self.generation_history:
                writer.writerow([generation_data.get(header, '') for header in headers])
                
        print(f"Generation history saved to {csv_path}")
        return csv_path

    def _select_molecules_with_batching(self, scoring_function, molecule_list, use_lowest_scoring=True, count=100, batch_size=1000):
        """
        Efficiently select molecules using batch processing with early stopping
        
        Args:
            scoring_function: Function to score molecules
            molecule_list: List of SMILES strings to choose from
            use_lowest_scoring: If True, select lowest scoring molecules, otherwise highest
            count: Number of molecules to select
            batch_size: Size of each batch to process
        Returns:
            List of selected SMILES strings
        """
        print(f"Using batch processing with early stopping to select {count} {'lowest' if use_lowest_scoring else 'highest'} scoring molecules")
        print(f"Processing {len(molecule_list)} molecules in batches of {batch_size}")
        
        # If we need most of the molecules, just score them all
        if count > len(molecule_list) * 0.5:
            print(f"Selection size ({count}) is more than 50% of total molecules, scoring all at once")
            # Score all molecules at once
            try:
                scores = [score_without_counting(s, scoring_function) for s in molecule_list]
                scored_smiles = sorted(zip(scores, molecule_list), key=lambda x: x[0], reverse=not use_lowest_scoring)
                return [s for _, s in scored_smiles[:count]]
            except Exception as e:
                print(f"Error during standard scoring: {e}")
                return []
        try:
            # Priority queue (heap) to keep track of best molecules
            # For min heap, store scores as is; for max heap, negate scores
            if use_lowest_scoring:
                # Min heap for lowest scoring
                best_molecules = []
            else:
                # Max heap for highest scoring (use negated scores for max heap)
                best_molecules = []
            threshold = float('inf') if use_lowest_scoring else float('-inf')
            total_processed = 0
            
            # Process molecules in batches
            for batch_start in range(0, len(molecule_list), batch_size):
                batch_end = min(batch_start + batch_size, len(molecule_list))
                batch = molecule_list[batch_start:batch_end]
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(molecule_list) + batch_size - 1)//batch_size} with {len(batch)} molecules")
                
                # Score the batch
                batch_scores = []
                for i, s in enumerate(batch):
                    try:
                        score = score_without_counting(s, scoring_function)
                        # For lowest scoring, we want to keep scores less than threshold
                        # For highest scoring, we want to keep scores greater than threshold
                        if (use_lowest_scoring and score < threshold) or (not use_lowest_scoring and score > threshold):
                            if len(best_molecules) >= count:
                                # Replace the worst molecule in our current selection
                                if use_lowest_scoring:
                                    # For min heap, heappushpop removes the largest element
                                    heapq.heappushpop(best_molecules, (-score, s))
                                else:
                                    # For max heap, heappushpop removes the smallest element
                                    heapq.heappushpop(best_molecules, (score, s))
                            else:
                                # Just add the molecule to our selection
                                if use_lowest_scoring:
                                    heapq.heappush(best_molecules, (-score, s))
                                else:
                                    heapq.heappush(best_molecules, (score, s))
                            
                            # Update the threshold to be the worst score in our current selection
                            if len(best_molecules) >= count:
                                if use_lowest_scoring:
                                    # For min heap, the smallest element has the highest negative score
                                    threshold = -best_molecules[0][0]
                                else:
                                    # For max heap, the smallest element has the lowest scores greater than threshold
                                    threshold = best_molecules[0][0]
                    except Exception as e:
                        print(f"Error scoring molecule {i} in batch: {e}")
                
                total_processed += len(batch)
                print(f"Processed {total_processed}/{len(molecule_list)} molecules, current threshold: {threshold:.4f}")
                
                # Check if we have enough molecules and can stop early
                if len(best_molecules) >= count:
                    # If our threshold is very good, we can stop early
                    if (use_lowest_scoring and threshold < 0.1) or (not use_lowest_scoring and threshold > 0.9):
                        print(f"Early stopping at {total_processed}/{len(molecule_list)} molecules with threshold {threshold:.4f}")
                        break
            
            # Extract the molecules from the heap
            if use_lowest_scoring:
                # For min heap, sort by negative score (ascending)
                selected_molecules = [s for _, s in sorted(best_molecules, key=lambda x: -x[0])]
            else:
                # For max heap, sort by score (descending)
                selected_molecules = [s for _, s in sorted(best_molecules, key=lambda x: x[0], reverse=True)]
                
            print(f"Selected {len(selected_molecules)} molecules with {'lowest' if use_lowest_scoring else 'highest'} scores")
            
            return selected_molecules[:count]
        except Exception as e:
            print(f"Error during batch processing: {e}")
            return []
     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.txt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--offspring_size', type=int, default=200)
    parser.add_argument('--mutation_rate', type=float, default=0.01)
    parser.add_argument('--crossover_rate', type=float, default=0.5)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=1)  # Default to 1 to avoid pickling issues
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--top_from_random', action='store_true',
                        help="Select top molecules from a random sample")
    parser.add_argument('--random_sample_size', type=int, default=1000,
                        help="Size of random sample if top_from_random is True")
    parser.add_argument('--use_highest_scoring', action='store_true', 
                        help="Select highest scoring molecules instead of lowest scoring")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--suite', default='v2')
    parser.add_argument('--return_initial_population', action='store_true',
                        help="Whether to return initial population info")
    parser.add_argument('--oracle_budget', type=int, default=TOTAL_ORACLE_BUDGET,
                        help="Budget for oracle calls during evolution")
    parser.add_argument('--dynamic_offspring_size', action='store_true',
                        help="Allow dynamic control of offspring size based on LLM recommendations")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="Batch size for processing molecules during initialization")
    parser.add_argument('--early_stopping', action='store_true',
                        help="Enable early stopping during initial population selection")
    args = parser.parse_args()

    np.random.seed(args.seed)
    setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))
        
    # save command line args
    with open(os.path.join(args.output_dir, 'goal_directed_params.json'), 'w') as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    # Default to using lowest scoring molecules (opposite of args.use_highest_scoring)
    use_lowest_scoring = not args.use_highest_scoring

    optimiser = GB_GA_Generator(smi_file=args.smiles_file,
                                population_size=args.population_size,
                                offspring_size=args.offspring_size,
                                generations=args.generations,
                                mutation_rate=args.mutation_rate,
                                crossover_rate=args.crossover_rate,
                                n_jobs=args.n_jobs,  # Use user specified n_jobs value
                                random_start=args.random_start,
                                top_from_random=args.top_from_random,
                                random_sample_size=args.random_sample_size,
                                use_lowest_scoring=use_lowest_scoring,
                                patience=args.patience,
                                return_initial_population=args.return_initial_population,
                                oracle_budget=args.oracle_budget,
                                dynamic_offspring_size=args.dynamic_offspring_size,
                                batch_size=args.batch_size,
                                early_stopping=args.early_stopping,
                                results_dir=args.output_dir,
                                task_name=args.suite)

    json_file_path = os.path.join(args.output_dir, 'goal_directed_results.json')
    assess_goal_directed_generation(optimiser, json_output_file=json_file_path, benchmark_version=args.suite)

if __name__ == "__main__":
    main()
