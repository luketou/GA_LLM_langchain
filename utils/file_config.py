import os 
import json
from datetime import datetime
from utils.logger_config import main_logger as logger, get_logger

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create results directory
def setup_result_dir(task_name):
    """Set up results directory"""
    global timestamp
    result_dir = f"results/{task_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Created results directory: {result_dir}")
    return result_dir

# Save intermediate results
def save_intermediate_results(result_dir, gen_number, results):
    """Save intermediate results for each generation"""
    filename = f"{result_dir}/gen_{gen_number:03d}_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved intermediate results to: {filename}")
    return filename
