# agent.py

import re
from prompt import GAParameters, get_scoring_criteria, parameter_optimization_template
from tools.oracle_scoring import get_current_task
from utils.logger_config import agent_logger as logger
from utils.cerebras_llm import CerebrasChatLLM

# Initialize LLM for parameter analysis, lower temperature ensures more stable parameter recommendations
parameter_llm = None  # Initialize as None first, will be set later
logger.info("Parameter adjustment LLM will be initialized with API key when provided")

# Global oracle budget tracking
TOTAL_ORACLE_BUDGET = 5000
_remaining_oracle_budget = TOTAL_ORACLE_BUDGET  # Private variable for budget tracking

def get_total_oracle_budget():
    """Get the total oracle budget"""
    return TOTAL_ORACLE_BUDGET

def get_remaining_oracle_budget():
    """Get the remaining oracle budget"""
    return _remaining_oracle_budget

def update_remaining_oracle_budget(new_remaining):
    """Update the remaining oracle budget"""
    global _remaining_oracle_budget
    old_value = _remaining_oracle_budget
    _remaining_oracle_budget = max(0, new_remaining)
    logger.info(f"Updated remaining oracle budget: {old_value} -> {_remaining_oracle_budget}")
    return _remaining_oracle_budget

def initialize_parameter_llm(api_key=None):
    """Initialize the parameter tuning LLM with the given API key"""
    global parameter_llm
    try:
        # Only initialize if API key is provided
        if api_key:
            parameter_llm = CerebrasChatLLM(model="gpt-oss-120b", temperature=0.1, api_key=api_key)
            logger.info("Parameter adjustment LLM initialized with provided API key")
            return True
        else:
            logger.warning("No API key provided for parameter_llm initialization. LLM will not be initialized.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize parameter adjustment LLM: {str(e)}")
        return False

# Do NOT initialize with default settings as it will fail without an API key
# The LLM will be initialized in main.py when the API key is available

def tune_parameters(history: list) -> dict:
    """
    Generate better GA parameters based on historical data
    
    Args:
        history: Historical optimization records
        
    Returns:
        dict: Dictionary containing optimized parameters
    """
    # Determine new parameters based on history
    # Return default parameters if history is empty or too limited
    if not history or len(history) < 2:
        logger.info("Insufficient history, returning default parameters")
        return {
            "mutation_rate": 0.5,
            "crossover_rate": 1.0,
            "population_size": 100,
            "num_molecules": 100,
            "llm_molecules": 50,
            "generations": 3, # Default generations
            "reasoning": "Insufficient history, using default parameters."
        }
    
    # Check if parameter_llm is properly initialized
    if parameter_llm is None:
        logger.error("Parameter LLM is not initialized. Returning default parameters.")
        return {
            "mutation_rate": 0.5,
            "crossover_rate": 1.0,
            "population_size": 100,
            "num_molecules": 100,
            "llm_molecules": 50,
            "generations": 3, # Default generations
            "reasoning": "Parameter LLM not initialized, using default parameters."
        }
    
    # Use LLM to analyze history and suggest parameters
    try:
        logger.info(f"Using LLM to adjust parameters based on {len(history)} historical records")
        
        # Generate prompt for LLM
        prompt = get_parameter_optimization_prompt(history)
        
        # Use LLM to generate parameter recommendations
        # Convert string prompt to proper chat message
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
        llm_response = parameter_llm.invoke(messages).content
        
        logger.debug(f"LLM parameter suggestion response: {llm_response}")
        
        # Parse LLM response
        params = parse_llm_parameters_response(llm_response)
        
        # Log parameter adjustment results
        logger.info(f"LLM suggested parameters: mutation_rate={params['mutation_rate']}, "
                    f"num_molecules={params['num_molecules']}, llm_molecules={params['llm_molecules']}, "
                    f"generations={params['generations']}") # Added generations to log
        logger.info(f"Adjustment reason: {params['reasoning']}")
        
        return params
        
    except Exception as e:
        logger.error(f"LLM parameter adjustment error: {e}", exc_info=True)
        # Return default values when error occurs
        return {
            "mutation_rate": 0.5,
            "crossover_rate": 1.0,
            "population_size": 100,
            "num_molecules": 100,
            "llm_molecules": 50,
            "generations": 3, # Default generations
            "reasoning": f"LLM parameter adjustment error: {e}, using default parameters."
        }

def format_history_for_llm(history: list) -> str:
    formatted_lines = []
    if not history:
        return "No history available."

    for i, gen_data in enumerate(history):
        if not isinstance(gen_data, dict):
            formatted_lines.append(f"Generation/Event {i} (Unknown format): Invalid data")
            continue

        source = gen_data.get('source', 'Unknown')
        generation_id_str = f"Gen {gen_data['generation']}" if 'generation' in gen_data else f"Event {i}"
        line = f"{generation_id_str} (Source: {source}):"

        scores_info = []
        for key in ['max_score', 'avg_score', 'min_score', 'std_score']:
            value = gen_data.get(key)  
            if isinstance(value, (int, float)):
                scores_info.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
        if scores_info:
            line += " " + ", ".join(scores_info) + "."
        formatted_lines.append(line)

        if 'top_5_molecules' in gen_data and gen_data['top_5_molecules']:
            formatted_lines.append("  Top 5 Molecules:")
            for mol_idx, mol in enumerate(gen_data['top_5_molecules']):
                formatted_lines.append(f"    {mol_idx+1}. SMILES: {mol['SMILES']}, Score: {mol['score']:.4f}")

        if 'bottom_5_molecules' in gen_data and gen_data['bottom_5_molecules']:
            formatted_lines.append("  Bottom 5 Molecules:")
            for mol_idx, mol in enumerate(gen_data['bottom_5_molecules']):
                formatted_lines.append(f"    {mol_idx+1}. SMILES: {mol['SMILES']}, Score: {mol['score']:.4f}")

        formatted_lines.append("-" * 30)

    return "\n".join(formatted_lines)

def parse_llm_parameters_response(llm_response: str) -> dict:
    """
    Parse LLM parameter recommendation response
    
    Args:
        llm_response: LLM response text
        
    Returns:
        dict: Dictionary of parsed parameters
    """
    logger.info("Parsing LLM parameter response")
    try:
        # Use GAParameters class to parse LLM output
        params = GAParameters.parse_text(llm_response)
        logger.info(f"Successfully parsed parameters: mutation_rate={params.mutation_rate}, "
                    f"num_molecules={params.num_molecules}, llm_molecules={params.llm_molecules}, "
                    f"generations={params.generations}") 
        return {
            "mutation_rate": params.mutation_rate,
            "crossover_rate": 1.0,  # Fixed value
            "population_size": params.num_molecules, 
            "num_molecules": params.num_molecules, 
            "llm_molecules": params.llm_molecules,
            "generations": params.generations, 
            "reasoning": params.reasoning
        }
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}", exc_info=True)
        # Return default values
        logger.warning("Using default parameters due to parsing error")
        return {
            "mutation_rate": 0.5,
            "crossover_rate": 1.0,
            "population_size": 100,
            "num_molecules": 100,
            "llm_molecules": 50,
            "generations": 3, # Default generations
            "reasoning": "Parsing error, using default values"
        }

def parameter_tuner_tool(args) -> dict:
    """
    Tool for adjusting GA parameters based on history and LLM response
    
    Args:
        args: Dictionary containing historical data and optional LLM response,
             or a string that should be parsed as a JSON
        
    Returns:
        dict: Dictionary of adjusted parameters
    """
    logger.info(f"Parameter adjustment tool called - Args type: {type(args)}")
    
    # Initialize history and llm_response with default values
    history_data = []
    llm_response_data = None

    if isinstance(args, str):
        logger.info(f"Received string input to parameter_tuner_tool: {args}")
        try:
            import json
            # Try to parse it as a JSON string
            try:
                args_dict = json.loads(args)
                history_data = args_dict.get('history', [])
                llm_response_data = args_dict.get('llm_response', None)
            except json.JSONDecodeError:
                # If not valid JSON, treat the entire string as llm_response
                history_data = []
                llm_response_data = args
        except Exception as e:
            logger.error(f"Error parsing parameter_tuner_tool string input: {e}")
            return tune_parameters([])  # Use default parameters
            
    elif isinstance(args, dict):
        # Handle dictionary input
        history_data = args.get('history', [])
        llm_response_data = args.get('llm_response', None)
        
    else:
        # Handle unexpected types
        logger.warning(
            f"parameter_tuner_tool received an unexpected type for args: {type(args)}. "
            f"Value (first 200 chars): {str(args)[:200]}. Proceeding with default history and no llm_response."
        )
        # history_data remains []
        # llm_response_data remains None

    logger.info(f"Parameter adjustment tool processing - History length: {len(history_data)}, LLM response: {'provided' if llm_response_data else 'none'}")
    
    # If LLM response is provided, try to parse parameters from it
    if llm_response_data:
        # Ensure llm_response_data is a string before passing to parse_llm_parameters_response
        if not isinstance(llm_response_data, str):
            logger.warning(f"llm_response_data is not a string (type: {type(llm_response_data)}), converting to string.")
            llm_response_data = str(llm_response_data)
        return parse_llm_parameters_response(llm_response_data)
    
    # Otherwise call tune_parameters to use LLM to analyze history and generate parameters
    return tune_parameters(history=history_data)

def get_parameter_optimization_prompt(history: list) -> str:
    current_task = get_current_task()
    remaining_budget = get_remaining_oracle_budget()
    history_text = format_history_for_llm(history)
    scoring_criteria = get_scoring_criteria(current_task)
    prompt = parameter_optimization_template.substitute(
        history_text=history_text,  # 將 examples_text 修正為 history_text 以匹配模板中的變數名
        remaining_budget=remaining_budget,
        scoring_criteria=scoring_criteria
    )
    return prompt

def analyze_trend(history: list) -> dict:
    """
    Analyze trends in historical records
    
    Args:
        history: Historical optimization records
        
    Returns:
        dict: Dictionary containing trend analysis results
    """
    if not history or len(history) < 2:
        return {"trend": "insufficient_data"}
        
    try:
        # Calculate change rates of maximum scores and average scores
        max_scores = [gen.get('max_score', 0) for gen in history]
        avg_scores = [gen.get('avg_score', 0) for gen in history]
        
        # Calculate recent generation change trend
        recent_max_change = (max_scores[-1] - max_scores[-2]) / max(0.0001, max_scores[-2])
        recent_avg_change = (avg_scores[-1] - avg_scores[-2]) / max(0.0001, avg_scores[-2])
        
        # Calculate overall trend
        overall_max_change = (max_scores[-1] - max_scores[0]) / max(0.0001, max_scores[0])
        overall_avg_change = (avg_scores[-1] - avg_scores[0]) / max(0.0001, avg_scores[0])
        
        # Determine if plateau has been reached
        plateau_threshold = 0.001  # 0.1% change rate as plateau threshold
        plateau_count = 0
        for i in range(len(max_scores)-1, 0, -1):
            change = (max_scores[i] - max_scores[i-1]) / max(0.0001, max_scores[i-1])
            if abs(change) < plateau_threshold:
                plateau_count += 1
            else:
                break
                
        is_plateau = plateau_count >= 3  # Three consecutive generations with change below threshold indicates plateau
        
        # Determine current strategy effectiveness
        strategy = "explore" if recent_max_change > 0.01 else "exploit"
        if is_plateau:
            strategy = "diversify"  # If plateau is reached, recommend increasing diversity
            
        return {
            "recent_max_change": recent_max_change,
            "recent_avg_change": recent_avg_change,
            "overall_max_change": overall_max_change,
            "overall_avg_change": overall_avg_change,
            "is_plateau": is_plateau,
            "plateau_count": plateau_count,
            "strategy": strategy
        }
        
    except Exception as e:
        logger.error(f"Trend analysis error: {e}", exc_info=True)
        return {"trend": "error", "error": str(e)}
