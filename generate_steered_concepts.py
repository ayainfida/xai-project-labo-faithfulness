from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from jinja2 import Template
import conceptset_utils
from tqdm import tqdm
import pandas as pd
import dotenv
import torch
import json
import os
import argparse
import re

base_prompt=Template("""{{ prompt }}

    Your output should be in the following format:
    <tag>feature 1</tag>
    <tag>feature 2</tag>
    ...
    <tag>feature n</tag>""")

prompts = [
    """List the most important features for recognizing something as a "goldfish":
    <tag>bright orange color</tag>
    <tag>a small,round body</tag>
    <tag>a long flowing tail</tag>
    <tag>a small mouth</tag>
    <tag>orange fins</tag>

    List the most important features for recognizing something as a "scuba diver":
    <tag>a person wearing a scuba mask and breathing apparatus</tag>
    <tag>a wetsuit</tag>
    <tag>flippers</tag>
    <tag>diving tank</tag>

    List the most important features for recognizing something as a {{ class_name }}:
    """,

    """List the things most commonly seen around a "tench":
    <tag>a pond</tag>
    <tag>fish</tag>
    <tag>a net</tag>
    <tag>a rod</tag>
    <tag>a reel</tag>
    <tag>a hook</tag>
    <tag>bait</tag>

    List the things most commonly seen around a "mountain":
    <tag>a peak</tag>
    <tag>a valley</tag>
    <tag>a forest</tag>
    <tag>rocks</tag>
    <tag>dirt</tag>
    <tag>a trail</tag>
    <tag>animals</tag>

    List the things most commonly seen around a {{ class_name }}:
    """,

    """Give superclasses for the word "tench":
    <tag>fish</tag>
    <tag>vertebrate</tag>
    <tag>animal</tag>

    Give superclasses for the word "grasshopper":
    <tag>insect</tag>
    <tag>arthropod</tag>
    <tag>animal</tag>

    Give superclasses for the word {{ class_name }}:
    """
    ]

prompt_templates = [Template(prompt) for prompt in prompts]

def prepare_model_prompts(dataset, classes, prompt_templates):
    model_prompts = {}
    classes_modified = []

    if dataset == "flower":
        classes_modified = [class_name + ' flower' for class_name in classes]
    else:
        classes_modified = classes

    for class_name in classes_modified:
        model_prompts[class_name] = []
        for prompt_template in prompt_templates:
            prompt = base_prompt.render(prompt=prompt_template.render(class_name=class_name))
            model_prompts[class_name].append(prompt)
                
    return model_prompts

def generate_content(prompt):
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.9,
            "top_p": 1.0,
            "max_output_tokens": 1000
        }
    )
    return response.text

def generate_prompts(dataset, model_prompts, run=1):     
    dataset_results = []

    for class_name, prompts in model_prompts.items():
        for prompt in prompts:
            row = {
                "dataset": dataset,
                "class_name": class_name,
                "prompt": prompt,
                "content": ''
            }
            dataset_results.append(row)

    file_path = f"llm_out_steered/{dataset}/run_{run}.csv"  # fix 1: define before use
    os.makedirs(f"llm_out_steered/{dataset}", exist_ok=True)
    df = pd.DataFrame(dataset_results)            # fix 3: store df, call to_csv separately
    df.to_csv(file_path, index=False)

    return file_path, df

def run_generation(file_path):
    results = []
    df = pd.read_csv(file_path)
    for prompt in tqdm(df["prompt"], desc=f"Generating content {file_path.split('/')[-1]}"):
        content = generate_content(prompt)
        results.append(content)
    
    df["content"] = results
    df.to_csv(file_path, index=False)
    return results

def process_run(run, selected_dataset, model_prompts):
    file_path, df = generate_prompts(selected_dataset, model_prompts, run=run)
    run_generation(file_path)
    return run, file_path

device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")

t5_model = T5ForConditionalGeneration.from_pretrained(
    "t5-large-concept-extractor"
).to(device)

def extract_concept(text: str) -> list[str]:
    return re.findall(r"<tag>(.*?)</tag>", text, re.DOTALL)


def verify_content(dataset):
    for run in range(1, 3):
        file_path = os.path.join("llm_out_steered", dataset, f"run_{run}.csv")
        df = pd.read_csv(file_path)
        extracted_concepts = [extract_concept(str(content)) for content in df["content"].tolist()]

        df["concept_count"] = [len(c) for c in extracted_concepts]
        under_10_idx = df[df["concept_count"] != df['content'].str.count("<tag>")].index.tolist()

        if not under_10_idx:
            continue

        print(f"Run {run} has {len(under_10_idx)} prompts with less than 10 concepts. Re-generating content for these prompts.")
        for idx in tqdm(under_10_idx, desc=f"Re-running {os.path.basename(file_path)}"):
            prompt = df.at[idx, "prompt"]
            new_content = generate_content(prompt)
            new_concepts = extract_concepts(new_content)
            df.at[idx, "content"] = new_content
            df.at[idx, "concept_count"] = len(new_concepts)
            print(f"  {idx}[{df.at[idx, 'class_name']}]")

        df.to_csv(file_path, index=False)
        print(f"Saved {file_path}")

def concatenate_runs(dataset):
    all_dfs = []
    for run in range(1, 3):
        file_path = os.path.join("llm_out_steered", dataset, f"run_{run}.csv")
        df = pd.read_csv(file_path)
        df['run'] = [run] * len(df)
        df = df[['run','dataset', 'class_name', 'prompt', 'content']]  # reorder columns

        all_dfs.append(df)

    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    concatenated_df.to_csv(os.path.join('llm_out_steered', dataset, "combined_results.csv"), index=False)

def extract_concepts(dataset):
    combined_file = os.path.join("llm_out_steered", dataset, "combined_results.csv")
    df = pd.read_csv(combined_file)
    data_json = {dataset: {}}

    for class_name in tqdm(df['class_name'].unique(), desc=f"Extracting concepts for {dataset}"):
        extracted_concepts = [extract_concept(str(content)) for content in df[df['class_name'] == class_name]["content"].tolist()]
        data_json[dataset][class_name] = [s for sublist in extracted_concepts for s in sublist]  # flatten list of lists

        # dump json file
        with open(os.path.join('llm_out_steered', dataset, "data.json"), "w") as f:
            json.dump(data_json, f, indent=4)

# Minimal concept filtering
CLASS_SIM_CUTOFF = 0.85
OTHER_SIM_CUTOFF = 0.9
MAX_LEN = 30
PRINT_PROB = 1

def concept_filtering(dataset):
    with open(os.path.join('llm_out_steered', dataset, "data.json"), "r") as f:
        important_dict = json.load(f)[dataset]

    concepts = set()

    for values in important_dict.values():
        concepts.update(set(values))

    classes = list(important_dict.keys())
    concepts = conceptset_utils.remove_too_long(concepts, MAX_LEN, PRINT_PROB)
    concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, CLASS_SIM_CUTOFF, device, PRINT_PROB)
    concepts = conceptset_utils.filter_too_similar(concepts, OTHER_SIM_CUTOFF, device, PRINT_PROB)


    to_dump = {}
    dup_list = []

    for class_name, class_concepts in important_dict.items():
        to_dump[class_name] = []
        for concept in class_concepts:
            if concept in concepts and concept not in dup_list:
                to_dump[class_name].append(concept)
                dup_list.append(concept)
    
    with open(f"datasets/{dataset}/concepts/class2concepts_gemini_newf.json", "w") as f:
        json.dump(to_dump, f, indent=4)


if __name__ == "__main__":
    dotenv.load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # args
    parser = argparse.ArgumentParser(description="Generate concepts for a dataset")
    parser.add_argument("--dataset", type=str, choices=["flower", "food", "CIFAR10", "CUB"], help="Dataset to process")
    args = parser.parse_args()

    with open("all_datasets_classes.json", "r") as f:
        all_datasets_classes = json.load(f)

    selected_dataset = args.dataset
    classes = all_datasets_classes[selected_dataset]

    model_prompts = prepare_model_prompts(selected_dataset, classes, prompt_templates)

    total_prompts = sum(len(prompts) for class_name, prompts in model_prompts.items())
    print(f"Total number of prompts: {total_prompts * 10}")

    max_workers = 10

    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [
    #         executor.submit(process_run, run, selected_dataset, model_prompts)
    #         for run in range(1, 3)
    #     ]

    #     for future in as_completed(futures):
    #         try:
    #             run, file_path = future.result()
    #             print(f"Done run={run}, file={file_path}")
    #         except Exception as e:
    #             print(f"Run failed: {e}")

    # verify_content(selected_dataset)
    # concatenate_runs(selected_dataset)
    extract_concepts(selected_dataset)
    concept_filtering(selected_dataset)