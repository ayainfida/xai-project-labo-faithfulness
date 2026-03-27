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

base_prompt=Template("""Use 10 sentences to describe the {{ prompt }}:

    Your output should be in the following format:
    <sentence>sentence 1</sentence>
    <sentence>sentence 2</sentence>
    ...
    <sentence>sentence 10</sentence>""")

prompts = [
    """what the {{ class_name }} looks like""",
    """the appearance of the {{ class_name }}""",
    """the color of the {{ class_name }}""",
    """the pattern of the {{ class_name }}""",
    """the shape of the {{ class_name }}""",
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

    file_path = f"llm_out/{dataset}/run_{run}.csv"  # fix 1: define before use
    os.makedirs(f"llm_out/{dataset}", exist_ok=True)
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

def sentences2concepts(sentences, class_name, batch_size=128):

    task_prefix = "extract concepts from sentence: "

    concepts = []
    outputs = []

    for i in range(0, len(sentences), batch_size):

        batch = sentences[i:i + batch_size]

        inputs = t5_tokenizer(
            [task_prefix + s for s in batch],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            output_sequences = t5_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

        decoded = t5_tokenizer.batch_decode(
            output_sequences,
            skip_special_tokens=True
        )

        outputs.extend(decoded)

    for out in outputs:
        concepts.extend(out.split("; "))

    return concepts

def extract_sentences(text: str) -> list[str]:
    return re.findall(r"<sentence>(.*?)</sentence>", text, re.DOTALL)

def verify_content(dataset):
    for run in range(1, 11):
        file_path = os.path.join("llm_out", dataset, f"run_{run}.csv")
        df = pd.read_csv(file_path)
        extracted_sentences = [extract_sentences(str(content)) for content in df["content"].tolist()]

        df["sentence_count"] = [len(s) for s in extracted_sentences]
        under_10_idx = df[df["sentence_count"] != 10].index.tolist()

        if not under_10_idx:
            continue

        print(f"Run {run} has {len(under_10_idx)} prompts with less than 10 sentences. Re-generating content for these prompts.")
        for idx in tqdm(under_10_idx, desc=f"Re-running {os.path.basename(file_path)}"):
            prompt = df.at[idx, "prompt"]
            new_content = generate_content(prompt)
            new_sentences = extract_sentences(new_content)
            df.at[idx, "content"] = new_content
            df.at[idx, "sentence_count"] = len(new_sentences)
            print(f"  {idx}[{df.at[idx, 'class_name']}]")

        df.to_csv(file_path, index=False)
        print(f"Saved {file_path}")

def concatenate_runs(dataset):
    all_dfs = []
    for run in range(1, 11):
        file_path = os.path.join("llm_out", dataset, f"run_{run}.csv")
        df = pd.read_csv(file_path)
        df['run'] = [run] * len(df)
        df = df[['run','dataset', 'class_name', 'prompt', 'content']]  # reorder columns

        all_dfs.append(df)

    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    concatenated_df.to_csv(os.path.join('llm_out', dataset, "combined_results.csv"), index=False)

def extract_concepts(dataset):
    combined_file = os.path.join("llm_out", dataset, "combined_results.csv")
    df = pd.read_csv(combined_file)
    data_json = {dataset: {}}

    for class_name in tqdm(df['class_name'].unique(), desc=f"Extracting concepts for {dataset}"):
        data_json[dataset][class_name] = {"sentences": []}
        extracted_sentences = [extract_sentences(str(content)) for content in df[df['class_name'] == class_name]["content"].tolist()]
        data_json[dataset][class_name]["sentences"] = [s for sublist in extracted_sentences for s in sublist]  # flatten list of lists
        data_json[dataset][class_name]["concepts"] = sentences2concepts(data_json[dataset][class_name]["sentences"], class_name)

        # dump json file
        with open(os.path.join('llm_out', dataset, "data.json"), "w") as f:
            json.dump(data_json, f, indent=4)

def filter_concepts(dataset, super_class="object", verbose=False):
    data_json_path = os.path.join('llm_out', dataset, "data.json")
    with open(data_json_path, "r") as f:
        data_json = json.load(f)

    cleaned_data_json = conceptset_utils.clean_dataset_one_class_at_a_time(data_json, dataset, super_class=super_class, verbose=verbose)

    to_write = {}
    for classes in cleaned_data_json[dataset].keys():
        to_write[classes] = cleaned_data_json[dataset][classes]['extracted_concepts']

    with open(f'datasets/{dataset}/concepts/class2concepts_gemini_n.json', 'w') as f:
        json.dump(to_write, f, indent=4)

if __name__ == "__main__":
    dotenv.load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # args
    parser = argparse.ArgumentParser(description="Generate concepts for a dataset")
    parser.add_argument("--dataset", type=str, choices=["flower", "food", "CIFAR10", "CUB"], help="Dataset to process")
    args = parser.parse_args()

    # superclasses
    SUPER_CLASSES = {
        "flower": "flower",
        "food": "food",
        "CIFAR10": "object",
        "CUB": "bird",
    }

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
    #         for run in range(1, 11)
    #     ]

    #     for future in as_completed(futures):
    #         try:
    #             run, file_path = future.result()
    #             print(f"Done run={run}, file={file_path}")
    #         except Exception as e:
    #             print(f"Run failed: {e}")

    # verify_content(selected_dataset)
    # concatenate_runs(selected_dataset)
    # extract_concepts(selected_dataset)
    filter_concepts(selected_dataset, super_class=SUPER_CLASSES[selected_dataset], verbose=True)