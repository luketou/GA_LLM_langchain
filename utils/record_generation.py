import os
import csv
import json
import numpy as np
from datetime import datetime
from utils.file_config import setup_result_dir
from utils.logger_config import main_logger as logger
import sys

# Add parent directory to path to import from agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import get_remaining_oracle_budget, get_total_oracle_budget

# Set up global run timestamp
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_run_timestamp():
    """Get the current run timestamp"""
    global run_timestamp
    return run_timestamp

def record_generation_stats(task_name, generation_type, generation_num, scores, 
                           source="GA", mutation_rate=None, offspring_size=None, 
                           crossover_rate=None, oracle_calls=None, notes=""):
    """
    Record generation statistics to a unified CSV file
    
    Args:
        task_name: Current task name
        generation_type: Type of generation ('GA' or 'LLM')
        generation_num: Generation number
        scores: List of molecule scores
        source: Source of the molecules (GA, LLM, Initial)
        mutation_rate: Mutation rate used for GA (optional)
        offspring_size: Offspring size used for GA (optional)
        crossover_rate: Crossover rate used for GA (optional)
        oracle_calls: Number of oracle calls made in this generation
        notes: Additional notes
    """
    if not scores:
        logger.warning(f"Cannot record {generation_type} statistics: Score list is empty")
        return

    # Ensure results directory exists
    results_dir = setup_result_dir(task_name)
    
    # Generation record CSV path
    global run_timestamp
    csv_path = os.path.join(results_dir, f'each_generation_record_{task_name}_{run_timestamp}.csv')
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Calculate statistics
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    std_score = np.std(scores) if len(scores) > 1 else 0.0
    
    # Calculate cumulative oracle calls
    remaining_budget = get_remaining_oracle_budget()
    total_budget = get_total_oracle_budget()
    if oracle_calls is None:
        oracle_calls = 0
    cumulative_oracle_calls = total_budget - remaining_budget
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write statistics to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow([
                'timestamp', 'task_name', 'source', 'generation', 
                'max_score', 'avg_score', 'min_score', 'std_score', 
                'count', 'oracle_calls', 'cumulative_oracle_calls',
                'mutation_rate', 'offspring_size', 'crossover_rate', 'notes'
            ])
        
        # Write data
        writer.writerow([
            timestamp,
            task_name,
            source,
            generation_num,
            f"{max_score:.6f}",
            f"{avg_score:.6f}",
            f"{min_score:.6f}",
            f"{std_score:.6f}",
            len(scores),
            oracle_calls,
            cumulative_oracle_calls,
            f"{mutation_rate:.4f}" if mutation_rate is not None else "N/A",
            offspring_size if offspring_size is not None else "N/A",
            f"{crossover_rate:.4f}" if crossover_rate is not None else "N/A",
            notes
        ])
    
    logger.info(f"Recorded {generation_type} generation {generation_num} statistics to {csv_path}")
    logger.info(f"Statistics: max={max_score:.4f}, avg={avg_score:.4f}, min={min_score:.4f}, std={std_score:.4f}, sample size={len(scores)}")
    
    return csv_path

def record_generation_details_json(task_name, generation_type, generation_num, 
                                  molecules_data, oracle_budget=None):
    """
    Record detailed molecule data for each generation to a JSON file
    
    Args:
        task_name: Current task name
        generation_type: Type of generation ('GA' or 'LLM')
        generation_num: Generation number
        molecules_data: List of molecule data [{"SMILES": str, "score": float}, ...]
        oracle_budget: Remaining oracle budget
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure results directory exists
    results_dir = setup_result_dir(task_name)
    
    # JSON file path
    global run_timestamp
    json_path = os.path.join(results_dir, f'generation_details_{task_name}_{run_timestamp}.json')
    
    # Load existing data or create new data structure
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {
            "task_name": task_name,
            "run_timestamp": run_timestamp,
            "generations": []
        }
    
    # Calculate statistics
    if molecules_data:
        scores = [mol["score"] for mol in molecules_data if "score" in mol]
        if scores:
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
        else:
            max_score = avg_score = min_score = std_score = 0.0
    else:
        max_score = avg_score = min_score = std_score = 0.0
        scores = []
    
    # If oracle_budget is not provided, get it from agent
    if oracle_budget is None:
        oracle_budget = get_remaining_oracle_budget()
    
    # Create generation data structure
    generation_data = {
        "generation_number": generation_num,
        "generation_type": generation_type,
        "timestamp": timestamp,
        "oracle_budget_remaining": oracle_budget,
        "cumulative_oracle_calls": get_total_oracle_budget() - oracle_budget,
        "statistics": {
            "max_score": max_score,
            "avg_score": avg_score,
            "min_score": min_score,
            "std_score": std_score,
            "molecule_count": len(molecules_data)
        },
        "molecules": molecules_data
    }
    
    # Add top and bottom molecules if data is available
    if molecules_data:
        generation_data["top_10_molecules"] = sorted(
            molecules_data, key=lambda x: x.get("score", 0), reverse=True
        )[:10] if molecules_data else []
        
        generation_data["bottom_10_molecules"] = sorted(
            molecules_data, key=lambda x: x.get("score", 0)
        )[:10] if molecules_data else []
    
    # Add to generations list
    data["generations"].append(generation_data)
    
    # Save to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Recorded generation {generation_num} details to {json_path}")
    logger.info(f"Recorded {len(molecules_data)} molecules, max={max_score:.4f}, avg={avg_score:.4f}")
    
    return json_path

def save_elite_pool(elite_pool, results_dir=None, task_name=None, generation_type=None):
    """
    Save the elite pool to a CSV file
    
    Args:
        elite_pool: List of (score, smiles) tuples
        results_dir: Results directory
        task_name: Current task name
        generation_type: Type of generation ('GA' or 'LLM')
        
    Returns:
        Path to the saved CSV file
    """
    if not elite_pool:
        logger.warning("Elite pool is empty, nothing to save")
        return None
    
    if task_name is None:
        logger.warning("No task name provided, cannot save elite pool")
        return None
    
    if results_dir is None:
        results_dir = setup_result_dir(task_name)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Get current timestamp
    global run_timestamp
    csv_path = os.path.join(results_dir, f'elite_pool_{task_name}_{run_timestamp}.csv')
    
    # Sort elite pool by score (highest first)
    sorted_elite = sorted(elite_pool, key=lambda x: x[0], reverse=True)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'smiles'])
        for score, smiles in sorted_elite:
            writer.writerow([f"{score:.6f}", smiles])
    
    logger.info(f"Saved elite pool with {len(elite_pool)} molecules to {csv_path}")
    
    # Also update the generation record CSV with elite pool statistics
    elite_scores = [score for score, _ in elite_pool]
    if elite_scores:
        record_generation_stats(
            task_name=task_name,
            generation_type=generation_type if generation_type else "Elite",
            generation_num="Elite",
            scores=elite_scores,
            source="Elite Pool",
            notes=f"Elite pool snapshot with {len(elite_pool)} molecules"
        )
    
    return csv_path

def load_generation_history_from_json(task_name, run_timestamp=None):
    """
    Load generation history from JSON file
    
    Args:
        task_name: Task name
        run_timestamp: Specific run timestamp, if None load the latest
        
    Returns:
        dict: Complete generation history data
    """
    results_dir = setup_result_dir(task_name)
    
    if run_timestamp:
        json_path = os.path.join(results_dir, f'generation_details_{task_name}_{run_timestamp}.json')
    else:
        # Find the latest JSON file
        json_files = []
        for filename in os.listdir(results_dir):
            if filename.startswith(f'generation_details_{task_name}_') and filename.endswith('.json'):
                json_files.append(os.path.join(results_dir, filename))
        
        if not json_files:
            logger.warning(f"No JSON history files found for task {task_name}")
            return None
            
        json_path = max(json_files, key=os.path.getmtime)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data.get('generations', []))} generations of history from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON history file: {str(e)}")
        return None