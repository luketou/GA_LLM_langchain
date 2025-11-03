from string import Template
import re
from pydantic import BaseModel, Field

# ======== Pydantic Models for Output Parsing ========
class GAParameters(BaseModel):
    """Parameters for the genetic algorithm"""
    mutation_rate: float = Field(..., ge=0.001, le=0.5, description="The mutation rate to use in the genetic algorithm")
    num_molecules: int = Field(100, ge=10, le=200, description="Number of molecules to generate in total (population size for GA, between 10-200)") # Clarified meaning
    llm_molecules: int = Field(50, ge=5, le=100, description="Number of molecules for LLM to generate (between 5-100)")
    generations: int = Field(..., ge=3, le=10, description="Number of generations for Graph GA to run (between 3 and 10)") # Added generations
    reasoning: str = Field(..., description="Reasoning for the parameters")
    
    @classmethod
    def parse_text(cls, text: str) -> "GAParameters":
        """Parse parameters from LLM output text"""
        # Use regex to find the values
        mutation_rate_match = re.search(r'["\']*mutation_rate["\']*\s*[:=]\s*(\d+\.\d+|\d+)', text)
        num_molecules_match = re.search(r'["\']*num_molecules["\']*\s*[:=]\s*(\d+)', text)
        llm_molecules_match = re.search(r'["\']*llm_molecules["\']*\s*[:=]\s*(\d+)', text)
        generations_match = re.search(r'["\']*generations["\']*\s*[:=]\s*(\d+)', text) # Added generations regex
        reasoning_match = re.search(r'["\']*reasoning["\']*\s*[:=]\s*["\']([^"\']+)["\']', text)
        
        # Extract values or use defaults
        mutation_rate = float(mutation_rate_match.group(1)) if mutation_rate_match else 0.05
        num_molecules = int(num_molecules_match.group(1)) if num_molecules_match else 100
        llm_molecules = int(llm_molecules_match.group(1)) if llm_molecules_match else min(50, num_molecules // 2)
        generations = int(generations_match.group(1)) if generations_match else 5 # Added generations extraction, default 5
        reasoning = reasoning_match.group(1) if reasoning_match else "Default reasoning"
        
        # Ensure values are within acceptable ranges
        mutation_rate = max(0.001, min(0.5, mutation_rate))
        num_molecules = max(10, min(200, num_molecules))
        llm_molecules = max(5, min(100, llm_molecules))
        generations = max(3, min(10, generations)) # Added generations range check
        
        # Make sure llm_molecules doesn't exceed num_molecules
        llm_molecules = min(llm_molecules, num_molecules)
        
        return cls(
            mutation_rate=mutation_rate,
            num_molecules=num_molecules,
            llm_molecules=llm_molecules,
            generations=generations, # Added generations
            reasoning=reasoning
        )

# ======== Scoring Function Descriptions ========
# Dictionary of scoring criteria descriptions for different tasks
TASK_SCORING_CRITERIA = {
    "osimertinib": """
Design molecules that:
1. Achieve high structural similarity to the reference (Tanimoto ≥ 0.8).
2. Have TPSA centered around 100 (Gaussian modifier).
3. Have logP centered around 1 (Gaussian modifier).
4. Are evaluated by the geometric mean of Top‑1, Top‑10 and Top‑100 scores.
""",
    "celecoxib": """
Generate molecules that:
1. Maximize Tanimoto similarity to the reference (no modifier).
2. Are scored by Top‑1 similarity only.
""",
    "fexofenadine": """
Generate molecules that:
1. Exceed 0.8 Tanimoto similarity to the reference (thresholded).
2. Have TPSA centered around 90 (Gaussian modifier).
3. Have logP centered around 4 (Gaussian modifier).
4. Are evaluated by the geometric mean of Top‑1, Top‑10 and Top‑100 scores.
""",
    "ranolazine": """
Generate molecules that:
1. Exceed 0.7 Tanimoto similarity to the reference (thresholded).
2. Have TPSA centered around 95 (Gaussian modifier).
3. Have logP centered around 7 (Gaussian modifier).
4. Have approximately one fluorine atom (Gaussian modifier).
5. Are evaluated by the geometric mean of Top‑1, Top‑10 and Top‑100 scores.
""",
    "amlodipine": """
Generate molecules that:
1. Maximize Tanimoto similarity to the reference.
2. Have around three ring systems (Gaussian modifier).
3. Are evaluated by the geometric mean of Top‑1, Top‑10 and Top‑100 scores.
""",
    "sitagliptin": """
Generate molecules that:
1. Achieve high structural similarity to the reference (Gaussian modifier).
2. Have TPSA centered around 77 (Gaussian modifier).
3. Have logP centered around 2.0 (Gaussian modifier).
4. Maintain the specified isomeric count (no modifier).
5. Are evaluated by the geometric mean of Top‑1, Top‑10 and Top‑100 scores.
""",
    "zaleplon": """
Generate molecules that:
1. Maximize Tanimoto similarity to the reference.
2. Match the target isomer count (no modifier).
3. Are evaluated by the geometric mean of Top‑1, Top‑10 and Top‑100 scores.
""",
    "median1": """
Generate molecules that:
1. Balance similarity to two reference scaffolds (geometric mean of two Tanimoto scores).
2. Are evaluated by Top‑1, Top‑10 and Top‑100 geometric mean.
""",
    "median2": """
Generate molecules that:
1. Balance similarity to two reference scaffolds (geometric mean of two Tanimoto scores).
2. Are evaluated by Top‑1, Top‑10 and Top‑100 geometric mean.
""",
    "troglitazone": """
Generate molecules that:
1. Maximize Tanimoto similarity to the reference (no modifier).
2. Are scored by Top‑1 similarity only.
""",
    "default": """
Design drug-like molecules that:
1. Have QED > 0.5.
2. TPSA in 70–140 range.
3. 0 to 7 rotatable bonds.
4. Fewer than 4 aromatic rings.
5. Exclude known problematic substructures.
"""
}

# Function to get scoring criteria based on task
def get_scoring_criteria(task):
    """
    Get the scoring criteria description for a specific molecule optimization task.
    
    Args:
        task (str): The task name (e.g., "osimertinib", "celecoxib")
        
    Returns:
        str: Formatted scoring criteria description
    """
    # Convert task to lowercase to handle case-insensitive matching
    task_lower = task.lower()
    
    # Extract base task name without parameters (for parametric tasks like logp_X)
    base_task = task_lower
    if '_' in task_lower:
        base_task = task_lower.split('_')[0]
    
    # Helper function to format criteria
    def format_criteria(criteria_text):
        """Format the criteria text by removing extra whitespace"""
        return '\n'.join([line.strip() for line in criteria_text.strip().split('\n')])
    
    # Find the matching criteria or use default
    for key in TASK_SCORING_CRITERIA:
        if key.lower() == task_lower or key.lower() == base_task:
            return format_criteria(TASK_SCORING_CRITERIA[key])
    
    # Return default criteria if no match found
    return format_criteria(TASK_SCORING_CRITERIA["default"])

# ======== Prompt Templates ========
# System message
system_message = """
You are a molecular optimization Agent:
- When performing internal reasoning, start your line with "Thought: <your reasoning>".
- When calling tools, use the format "Action: <tool_name>[<kwargs>]".
Available tools: graph_ga_generate, llm_generate, guacamol_score, parameter_tuner.
Your objective is to maximize performance within a limit of 5000 oracle calls,
combining Graph GA and LLM generation strategies.
LLM should dynamically decide the number of molecules to generate per generation (between 10 and 200).
"""

# Main process initial prompt
initial_prompt = """
Start optimization with initial GA parameters:
mutation_rate=0.5, crossover_rate=1.0, population_size=100.
Ensure that each generation's total generated molecule count is between 10 and 200.
"""

# LLM molecule generation prompt template
llm_generate_template = Template("""
You are a seasoned computational chemist and de novo design expert. Your objective is to propose novel, high-scoring drug-like molecules for the current task.

1. Insight Extraction
   • Carefully examine the following top 10 high-scoring SMILES examples from previous generations:
$example_block

   • Identify and summarize in your mind the key structural motifs, scaffolds, and substituent patterns that correlate with high scores (e.g., ring systems, H-bond donors/acceptors, lipophilicity handles).

2. Task and Constraints
   • Task scoring criteria:
$task_description

3. Generation Instructions
   • Produce exactly $count chemically valid, unique SMILES strings.
   • Strive to balance two goals:
     1. **Exploit** the beneficial features seen in the examples (e.g., bioisosteric replacements, key H-bond patterns).
     2. **Explore** new scaffolds or substitution patterns that fit the scoring drivers.
   • Ensure diversity by varying core scaffolds and side-chain functionalities.

4. Output Format
   • **Only** output the SMILES strings, separated by commas.
   • Do **not** include numbering, bullet points, commentary, or any text beyond the comma-separated SMILES list.

Example output: CC(OCc1ccccc1)c1c(C#N)cncc1-c1cnc2c(c1)CCCN2C(=N)O,CN(Cc1ccc(C(=O)Nc2cc(-c3ccccc3)ccc2O)cc1)Cc1cnn(C)c1,Cc1ccc2c(c1)NC(=O)C(CC(=O)NCCCN1CCCC1)O2
""")

# Parameter optimization template
parameter_optimization_template = Template("""
As a GA optimization expert specialized in molecular design, suggest parameters for a molecule generation genetic algorithm.
Here's the recent performance history:
$history_text

IMPORTANT BUDGET INFORMATION:
There is a total oracle call budget of 5000 calls, with $remaining_budget calls remaining.
Each molecule you generate will require at least one oracle call to evaluate.

The scoring function evaluates molecules based on:
$scoring_criteria

Analyze the trend in max_score and avg_score, considering oracle call efficiency.
Suggest the following parameters for the next generation:
1. mutation_rate (0.1-1.0): Controls the rate of mutation. Consider starting with a higher mutation rate (e.g. 0.5 to 1.0) in the initial stages to encourage exploration and diversity. If the optimization seems to converge or the diversity of high-scoring molecules decreases, then consider gradually reducing the mutation rate to refine promising candidates.
2. num_molecules to generate in total (population size for GA, 50-200)
3. llm_molecules for LLM to generate directly (5-100, must be less than num_molecules)
4. generations for Graph GA to run (3-10)

Note: The crossover_rate is fixed at 1.0 and cannot be changed.

GUIDANCE FOR MOLECULE COUNTS AND GENERATIONS:
- num_molecules is the population size for the GA.
- llm_molecules is the subset that will be generated directly by the LLM, the rest will be generated by graph GA.
- generations determines how many evolution cycles the GA will run. More generations can lead to better molecules but consume more budget per GA call (population_size * generations oracle calls).
- If there's plenty of budget remaining (>2000 calls), you can suggest a moderate number of generations (e.g., 5-7) and a larger population (closer to 200).
- If budget is getting low (<1000 calls), be more conservative with both population size (closer to 50) and generations (e.g., 3-5).
- Consider efficiency: if historical data shows significant improvement per oracle call, you may want to generate more LLM molecules or run GA for more generations if the budget allows.

IMPORTANT: You must format your response EXACTLY as follows (do not include any other text):
{
  "mutation_rate": 0.XX,
  "num_molecules": YYY,
  "llm_molecules": ZZ,
  "generations": G,
  "reasoning": "Brief reason for these parameter selections, considering both performance and budget constraints"
}
Where 0.XX is a numerical value, YYY is an integer (50-200), ZZ is an integer (5-100, < YYY), and G is an integer (3-10).
""")

# Basic molecule generation template (no examples)
basic_molecule_generation_template = Template("""
Generate $n_molecules valid SMILES strings for drug-like molecules optimized for $task task.

IMPORTANT BUDGET INFORMATION:
There is a total oracle call budget of 5000 oracle calls, with $remaining_budget calls remaining.
Each molecule you generate will consume one oracle call when evaluated.

Target properties:
$scoring_criteria

CRITICAL INSTRUCTION: You must enclose each SMILES string between <SMILES> and </SMILES> tokens.
EACH MOLECULE MUST BE ON ITS OWN LINE.
DO NOT include ANY additional text, explanations, introductions, or conclusions.
DO NOT number the SMILES strings.
DO NOT explain your process or reasoning.

Example of CORRECT response format (ONLY SMILES strings with tags):
<SMILES>COCCCN(C(S)=Nc1cc(Cl)cc(Cl)c1)C(C)c1ccccn1</SMILES>
<SMILES>CCC(C)Sc1ccc(C#Cc2csc(C)n2)cn1</SMILES>
...
""")

# Enhanced molecule generation template with examples
enhanced_molecule_generation_template = Template(
    """
    Generate $n_molecules molecules optimized for the task "$task".
    Remaining budget: $remaining_budget.

    Scoring criteria:
    $scoring_criteria

    Historical examples:
    $examples_text
    """
)

# Task-specific prompt template
task_prompt_template = Template("""
Your task is to optimize molecules for the "$task" benchmark.

SCORING CRITERIA:
$scoring_criteria

Start optimization with initial GA parameters:
mutation_rate=0.5, crossover_rate=1.0, population_size=100.
Ensure that each generation's total generated molecule count is between 10 and 200.

Remember to balance exploration and exploitation to efficiently use your 5000 oracle call budget.
""")

def get_task_prompt(task_name):
    """
    Generate task-specific initial prompt based on task name
    
    Args:
        task_name (str): Name of the task
        
    Returns:
        str: Prompt with task-specific information
    """
    # Get scoring criteria for the task
    scoring_criteria = get_scoring_criteria(task_name)
    
    # Fill the template
    return task_prompt_template.substitute(
        task=task_name,
        scoring_criteria=scoring_criteria
    )

# Update system prompt to include dynamic task information
def get_system_message_with_task(task_name):
    """
    Generate system prompt with task information based on task name
    
    Args:
        task_name (str): Name of the task
        
    Returns:
        str: Updated system prompt
    """
    task_info = f"Current optimization task: {task_name}"
    return f"{system_message}\n{task_info}"