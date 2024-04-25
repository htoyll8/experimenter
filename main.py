import mlflow
import pandas as pd
import itertools
import argparse
from openai_utils import get_client
from mlflow_utils import set_mlflow_tracking_uri, create_mlflow_experiment

# Initialize OpenAI client
client = get_client()

def get_param_combinations(prompts_list, n_results, context_length):
    """
    Generate all possible combinations of parameters.

    Args:
        n_results (list): List of number of results to generate.
        context_lengths (list): List of context lengths.
        prompt_list (list): List of prompt templates.

    Returns:
        list: List of parameter combinations.
    """
    param_list = [prompts_list, n_results, context_length]
    return list(itertools.product(*param_list))

def generate_input(query, prompt, context):
    """
    Generates the input string for the language model based on the given parameters.
    """
    return prompt.format(query=query, context=context)

def chat_completion(query, prompt, context_length):
    """
    Sends a chat completion request to the OpenAI API and returns the response.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=context_length
    )

    return response.choices[0].message.content

def evaluate_prompt(comb, data, idx, experiment_id):
    """
    Evaluates a prompt combination and logs the results using MLflow.
    """
    prompt, n_results, context_length = comb
    with mlflow.start_run(run_name=f"EVALUATE_PROMPT_{idx}", experiment_id=experiment_id):
        mlflow.log_params({
            'n_results': n_results,
            'context_length': context_length,
            'llm_model': 'gpt-3.5-turbo',
            'prompt_instructions': str(prompt)
        })
        
        data['prompt_instructions'] = str(prompt)
        data['input'] = data['Queries'].apply(lambda x: generate_input(x, prompt, str(" ")))
        data['n_results'] = n_results
        data['context_length'] = context_length
        data['response'] = data['Queries'].apply(lambda x: chat_completion(x, prompt, context_length))

        print("Data: ", data.head())

        mlflow.log_table(data, artifact_file="qabot_eval_results.json")

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Evaluate prompts for question answering.')
    parser.add_argument('--experiment_name', type=str, default='EVALUATE_PROMPT', help='Name of the experiment')
    parser.add_argument('--artifact_location', type=str, default='mlruns', help='Location of the artifact')
    parser.add_argument('--n_results', nargs='+', type=int, default=[5], help='Number of results to return')
    parser.add_argument('--context_length', nargs='+', type=int, default=[4096], help='Length of context')
    parser.add_argument('--queries_file', type=str, default='queries.csv', help='Path to query file')

    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment_name
    ARTIFACT_LOCATION = args.artifact_location
    N_RESULTS = args.n_results
    CONTEXT_LENGTH = args.context_length
    QUERIES_FILE = args.queries_file
    PROMPT_LIST = [
        # TODO: Read prompt templates from file.
        "Here is a prompt for question answering: {query}\n\nArticle: {context}",
        "Given the following query: {query} and context: {context}, provide a relevant answer.",
        # Add more prompt templates here
    ]

    # Load queries from CSV file.
    data = pd.DataFrame(pd.read_csv(QUERIES_FILE))

    # Generate all possible combinations of parameters.
    combinations = get_param_combinations(PROMPT_LIST, N_RESULTS, CONTEXT_LENGTH)

    # Set MLflow tracking URI.
    set_mlflow_tracking_uri()

    # Create a new experiment.
    experiment_id = create_mlflow_experiment(experiment_name=EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION, tags={})

    # Evaluate each combination.
    for idx, comb in enumerate(combinations):
        evaluate_prompt(comb, data, idx, experiment_id)

if __name__ == '__main__':
    main()