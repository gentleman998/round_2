import os
import torch
import pandas as pd
from multiprocessing import Pool
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Set the CUDA_VISIBLE_DEVICES to use only two GPUs (GPU 0 and GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Define your instructions (replace these with your actual instructions)
MAIN_INSTRUCTION = "Classify the main category of this crime."
FINANCE_INSTRUCTION = "Classify the subcategory of this financial crime."
WOMEN_INSTRUCTION = "Classify the subcategory of this crime against women or children."
CYBER_ATTACK_INSTRUCTION = "Classify the subcategory of this cyber attack crime."
OTHER_INSTRUCTION = "Classify the subcategory of this other crime."

# Function to classify the main category
def classify_main_category(crime_info, model, tokenizer, instruction, device):
    messages = [{"role": "user", "content": f"{instruction}\n\nCrime Information: {crime_info}"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        parts = full_output.split("model\n")
        if len(parts) > 1:
            category = parts[-1].strip().split("\n")[0].strip()
            return category
        return "Error: No category found"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to classify the subcategory
def classify_subcategory(crime_info, model, tokenizer, instruction, device):
    messages = [{"role": "user", "content": f"{instruction}\n\nCrime Information: {crime_info}"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        parts = full_output.split("model\n")
        if len(parts) > 1:
            subcategory = parts[-1].strip().split("\n")[0].strip()
            return subcategory
        return "Error: No subcategory found"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to classify crime information using a specific GPU
def get_crime_classification(crime_info, device):
    # Load models and tokenizers onto the specified GPU
    main_model, main_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_crime_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
    ).to(device), get_chat_template("gemma-3")
    
    finance_model, finance_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_finance_crime_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
    ).to(device), get_chat_template("gemma-3")
    
    women_model, women_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_women_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
    ).to(device), get_chat_template("gemma-3")
    
    cyber_attack_model, cyber_attack_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_cyber_attack_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
    ).to(device), get_chat_template("gemma-3")
    
    other_model, other_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_other_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
    ).to(device), get_chat_template("gemma-3")
    
    # Step 1: Classify the main category
    category = classify_main_category(crime_info, main_model, main_tokenizer, MAIN_INSTRUCTION, device)
    
    # Step 2: Classify the subcategory based on the main category
    if category == "financial_crimes":
        subcategory = classify_subcategory(crime_info, finance_model, finance_tokenizer, FINANCE_INSTRUCTION, device)
    elif category == "crime_against_women_and_children":
        subcategory = classify_subcategory(crime_info, women_model, women_tokenizer, WOMEN_INSTRUCTION, device)
    elif category == "cyber_attack_or_dependent_crimes":
        subcategory = classify_subcategory(crime_info, cyber_attack_model, cyber_attack_tokenizer, CYBER_ATTACK_INSTRUCTION, device)
    elif category == "other_cyber_crime":
        subcategory = classify_subcategory(crime_info, other_model, other_tokenizer, OTHER_INSTRUCTION, device)
    else:
        subcategory = "Error: Invalid category"
    
    return {"category": category, "subcategory": subcategory}

# Worker function for multiprocessing
def worker(chunk, device):
    return [get_crime_classification(info, device) for info in chunk]

# Main function to process the CSV file
def process_csv(input_csv, output_csv, num_workers=2):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    crime_infos = df['crimeaditionalinfo'].tolist()
    
    # Split the data into chunks for each worker
    chunk_size = len(crime_infos) // num_workers
    chunks = [crime_infos[i:i + chunk_size] for i in range(0, len(crime_infos), chunk_size)]
    
    # Assign GPU devices to workers (cuda:0 and cuda:1)
    devices = [f"cuda:{i}" for i in range(num_workers)]
    
    # Create a pool of workers and process the chunks in parallel
    with Pool(num_workers) as pool:
        results = pool.starmap(worker, zip(chunks, devices))
    
    # Flatten the results into a single list
    classification_results = [item for sublist in results for item in sublist]
    
    # Add 'category' and 'subcategory' columns to the DataFrame
    df[['category', 'subcategory']] = pd.DataFrame(classification_results)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Usage example
process_csv('your_csv_file.csv', 'updated_crimes.csv')
