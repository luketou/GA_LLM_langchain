from langchain.agents import initialize_agent, Tool, AgentType, ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from tools.generator import graph_ga_generate, llm_generate, initialize_llm_with_key
from tools.oracle_scoring import set_task, get_current_task
from agent import parameter_tuner_tool, get_parameter_optimization_prompt, initialize_parameter_llm
from prompt import system_message, initial_prompt, get_task_prompt, get_system_message_with_task
import argparse
import json
import os
from utils.logger_config import main_logger as logger, get_logger
from utils.file_config import setup_result_dir, save_intermediate_results
from utils.cerebras_llm import CerebrasChatLLM
from dotenv import load_dotenv

# Load .env configuration so API keys are available before initialization
load_dotenv()

# Configure logger
logger.info("Initializing molecular optimization Agent")

# --- 1. Initialize LLM ---
# Will be initialized with API key when available
agent_llm = None
logger.info("Agent LLM will be initialized with API key")

# --- 2. Tools ---
tools = [
    Tool(
        name="graph_ga_generate", 
        func=graph_ga_generate,
        description="Execute Graph GA for a specified number of generations. Provide parameters as a dictionary (including 'generations' which can be between 3 and 10) and population size."
    ),
    Tool(
        name="llm_generate",  
        func=llm_generate,
        description="Generate molecules using LLM and automatically score them. Provide count of molecules to generate and history list."
    ),
    Tool(
        name="parameter_tuner", 
        func=parameter_tuner_tool,
        description="Dynamically adjust GA parameters based on optimization history."
    ),
]
logger.info(f"Tools configuration complete: {[tool.name for tool in tools]}")

# --- 3. Create custom ZeroShotAgent prompt ---
def create_custom_zero_shot_prompt(task_name):
    """Create custom ZeroShotAgent prompt"""
    # Get task-specific system message
    task_system_message = get_system_message_with_task(task_name)
    
    # Custom prefix (includes system message and task description)
    prefix = f"""
    {task_system_message}

    You are asked to maximize molecule performance on the {task_name} task.
    You should analyze historical data and decide whether to use GA or LLM to generate molecules.

    You have access to the following tools:
    """

    # Custom format instructions that match exactly what ZeroShotAgent expects
    format_instructions = """Use the following format:

Thought: I need to reason about what to do.
Action: The action to take, should be one of [{tool_names}]. (Do NOT include the input here. The input for the action must be specified on the next line with 'Action Input:')
Action Input: The input to the action. This MUST be on a new line immediately after the 'Action:' line. If the action takes a dictionary as input, provide it as a JSON string.
Observation: The result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: The final answer to the original question.
"""

    # Custom suffix (message ending)
    suffix = """
Begin thinking about how to optimize molecular structures. For complex parameter optimization decisions, use the parameter_tuner tool instead of guessing the best parameters yourself.

Question: {input}
{agent_scratchpad}
"""

    # Use ZeroShotAgent.create_prompt to create prompt template
    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        format_instructions=format_instructions,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )
    
    logger.info(f"Created custom prompt for task '{task_name}'")
    return prompt

# Function to initialize all LLMs with API key
def initialize_all_llms_with_key(api_key):
    """Initialize all LLMs with the provided API key"""
    global agent_llm

    resolved_key = api_key or os.environ.get("CEREBRAS_API_KEY")
    if not resolved_key:
        logger.error("No Cerebras API key provided. Please set --cerebras_api_key or CEREBRAS_API_KEY.")
        return False
    
    # Initialize agent LLM
    agent_llm = CerebrasChatLLM(
        model="gpt-oss-120b",
        temperature=0.1,
        api_key=resolved_key,
    )
    logger.info("Agent LLM initialized with API key")
    
    # Initialize generator LLM
    initialize_llm_with_key(resolved_key)
    logger.info("Generator LLM initialized with API key")
    
    # Initialize parameter tuner LLM with explicit API key
    success = initialize_parameter_llm(resolved_key)
    if success:
        logger.info("Parameter tuner LLM initialized with API key")
    else:
        logger.warning("Failed to initialize parameter tuner LLM with API key")
    return success

# --- 4. Main execution flow ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, 
        choices=[
            'celecoxib', 'troglitazone', 'thiothixene',
            'aripiprazole', 'albuterol', 'mestranol',
            'isomer_c11h24', 'isomer_c9h10n2o2pf2cl',
            'median1', 'median2',
            'osimertinib', 'fexofenadine', 'ranolazine', 
            'perindopril', 'amlodipine', 'sitagliptin', 'zaleplon',
            'valsartan_smarts', 'decoration_hop', 'scaffold_hop',
            'cobimetinib', 'qed', 'cns_mpo', 'weird_physchem',
            'logp_2.5', 'tpsa_100',
            'pharmacophore'
        ],
        default='osimertinib',
        help='Benchmark task to optimize (default: osimertinib)')
    parser.add_argument('--cerebras_api_key', type=str, default=None,
        help='API key for Cerebras (if not set in environment variables)')
    args = parser.parse_args()
    
    selected_task = args.task
    cerebras_api_key = args.cerebras_api_key
    logger.info(f"Selected task: {selected_task}")
    print(f"Selected task: {selected_task}")
    
    # Try to get API key from environment if not provided as argument
    if not cerebras_api_key:
        cerebras_api_key = os.environ.get('CEREBRAS_API_KEY')
        if cerebras_api_key:
            logger.info("Using Cerebras API key from environment variables")
    
    # Pass the selected task to the oracle_scoring module
    set_task(selected_task)
    logger.info(f"Task set in oracle_scoring module: {selected_task}")
    
    # Set up results directory
    result_dir = setup_result_dir(selected_task)
    print(f"Results will be saved in: {result_dir}")
    
    # Initialize all LLMs with API key if provided
    if cerebras_api_key or os.environ.get('CEREBRAS_API_KEY'):
        initialize_all_llms_with_key(cerebras_api_key)
    else:
        logger.error("No Cerebras API key provided; agent cannot initialize Cerebras-backed LLMs.")
        raise SystemExit("Cerebras API key is required.")
    
    try:
        # Create conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")
        
        # Create custom prompt
        custom_prompt = create_custom_zero_shot_prompt(selected_task)
        
        # Create LLM chain
        llm_chain = LLMChain(
            llm=agent_llm,
            prompt=custom_prompt,
            verbose=True,
            memory=memory
        )
        
        # Use ZeroShotAgent with updated format
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        logger.info("ZeroShotAgent created")
        
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=50
        )
        logger.info("Agent executor created")
        
        # Get task-specific initial prompt
        prompt = get_task_prompt(selected_task)
        logger.info("Task-specific prompt retrieved")
        
        # Execute Agent
        logger.info(f"Starting Agent execution for task: {selected_task}")
        result = agent_executor({"input": prompt})
        logger.info("Agent execution completed")
        
        # Save final results
        raw_steps = result["intermediate_steps"]
        # 轉成 list of list of str
        serial_steps = [[str(item) for item in step] for step in raw_steps]
        final_result = {
            "task": selected_task,
            "intermediate_steps": serial_steps,
            "output": result["output"],
            "chat_history": str(memory.chat_memory.messages)
        }
        save_intermediate_results(result_dir, 999, final_result)
        
        logger.info("Optimization completed")
        print("Optimization completed, Agent final response:")
        print(result["output"])
    except Exception as e:
        logger.error(f"Error occurred during execution: {str(e)}")
        raise
