# Import necessary libraries
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import time
import argparse
import numpy as np
import torch.multiprocessing as mp
from functools import partial

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cudnn.benchmark = True

# Configuration
BATCH_SIZE = 32  # Adjust based on your GPU memory

# Instructions (unchanged)
MAIN_INSTRUCTION = """Classify the crime information into one of these categories: 'financial_crimes', 'crime_against_women_and_children', 'cyber_attack_or_dependent_crimes', 'other_cyber_crime'.
1. financial_crimes: Cyber offenses that involve deceptive schemes designed to secure unauthorized financial gain. This category includes incidents such as online fraud, phishing scams, digital impersonation aimed at tricking victims into transferring money, unauthorized fund transfers, and fraudulent transactions using digital platforms.
2. crime_against_women_and_children: Cyber crimes that specifically target the safety and dignity of women and children. These offenses include online sexual exploitation, harassment, stalking, bullying, non-consensual sharing of intimate images, and other forms of digital abuse aimed at inflicting harm on these vulnerable groups.
3. cyber_attack_or_dependent_crimes: Crimes that involve unauthorized intrusion into digital systems or networks. This category covers hacking, data breaches, ransomware attacks, and other acts of digital system exploitation intended to disrupt, damage, or unlawfully access computer networks and information systems.
4. other_cyber_crime: Digital offenses that do not fall under the above specific categories. This includes general cyber crimes affecting individuals irrespective of gender, such as defamation, online hate speech, cyber stalking (not specifically targeting women or children), and miscellaneous offenses that do not clearly align with financial fraud, targeted abuse, or direct system attacks.
"""

FINANCE_INSTRUCTION = """Classify the crime information into one of these subcategories under 'financial_crimes':
1. investment_scam_trading_scam: Fraudulent investment or trading schemes promising high returns to deceive victims.
2. online_job_fraud: Fake job offers requiring upfront payments or personal information to exploit job seekers.
3. tech_support_scam_customer_care_scam: Scammers impersonating tech support or customer service to extract money or data.
4. online_loan_fraud: Fraudulent loan offers demanding fees or personal details without providing actual financial help.
5. matrimonial_romance_scam_honey_trapping_scam: Romance scams where fraudsters deceive victims for money or blackmail.
6. impersonation_of_govt_servant: Scammers pretending to be government officials to extort money or steal data.
7. cheating_by_impersonation_other_than_government_servant: Fraudsters posing as non-government officials to commit financial fraud.
8. sim_swap_fraud: Cloning SIM cards to gain unauthorized access to financial accounts.
9. sextortion_nude_video_call: Blackmailing individuals using explicit videos or fabricated content.
10. others: Other financial frauds conducted via digital platforms.
"""

WOMEN_INSTRUCTION = """Classify the crime information into one of these subcategories under 'crime_against_women_and_children':
1. rape_gang_rape: Threats, blackmail, or dissemination of content related to rape or gang rape targeting women and children through digital means.
2. sexual_harassment: Unwanted online sexual advances, explicit messages, or harassment directed at women and children.
3. cyber_voyeurism: Secretly recording or sharing intimate images/videos of women or children without consent.
4. cyber_stalking: Persistent online harassment or tracking of women and children, causing fear or distress.
5. cyber_bullying: Online harassment targeting children or women, leading to emotional or psychological harm.
6. child_pornography_csam: Production, distribution, or possession of child sexual abuse material (CSAM) online.
7. child_sexual_exploitative_material_csem: Any digital content depicting or facilitating the sexual exploitation of minors.
8. publishing_and_transmitting_obscene_material_sexually_explicit_material: Sharing obscene or explicit content involving women or children without consent.
9. computer_generated_csam_csem: AI-generated or manipulated sexual content involving children.
10. fake_social_media_profile: Creating fraudulent social media profiles to deceive or exploit women and children.
11. defamation: Spreading false or malicious content online to damage the reputation of women and children.
12. cyber_blackmailing_threatening: Threatening or coercing women and children online using sensitive content or personal data.
13. online_human_trafficking: Using digital platforms for trafficking women and children for forced labor or sexual exploitation.
14. others: Any other cyber crimes that specifically target women and children.
"""

CYBER_ATTACK_INSTRUCTION = """Classify the crime information into one of these subcategories under 'cyber_attack_or_dependent_crimes':
1. malware_attack: Deploying malicious software to disrupt systems or steal data.
2. ransomware_attack: Encrypting files and demanding ransom for access.
3. hacking_defacement: Unauthorized system access to alter content or steal information.
4. data_breach_theft: Illegally accessing and exposing confidential data.
5. tampering_with_computer_source_documents: Altering or deleting digital records for fraud or concealment.
6. denial_of_service_dos_distributed_denial_of_service_ddos_attacks: Overloading systems with traffic to make them inaccessible.
7. sql_injection: Injecting malicious SQL queries to extract or manipulate database data.
"""

OTHER_INSTRUCTION = """Classify the crime information into one of these subcategories:
1. fake_profile: Creating fraudulent online identities to deceive or defraud men.
2. phishing: Using deceptive techniques to steal sensitive financial information from men.
3. cyber_terrorism: Using digital platforms to promote terrorism or disrupt national security, impacting men.
4. social_media_account_hacking: Unauthorized access to men's social media accounts for misuse or exploitation.
5. online_gambling_betting_frauds: Illegal or fraudulent gambling and betting schemes targeting men.
6. business_email_compromise_email_takeover: Manipulating email systems to defraud men in business transactions.
7. provocative_speech_for_unlawful_acts: Spreading hate speech or inciting violence online, often impacting men.
8. matrimonial_honey_trapping_scam: Using matrimonial platforms to deceive or exploit men for financial or personal gain.
9. fake_news: Spreading false or misleading information online, affecting men.
10. cyber_stalking_bullying: Online harassment, threats, or intimidation primarily targeting men.
11. defamation: Publishing false statements online to damage the reputation of men.
12. cyber_pornography: Illegal distribution or sharing of explicit digital content related to men.
13. sending_obscene_material: Distributing obscene content online without consent, affecting men.
14. intellectual_property_ipr_thefts: Online theft of copyrighted materials, trademarks, or patents, impacting male professionals and businesses.
15. cyber_blackmailing_threatening: Coercing men online using threats or blackmail tactics.
16. others: Any other general cyber crimes affecting men.
"""

# Function to load models on a specific GPU
def load_models(gpu_id):
    # Set the device for this process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"Loading models on GPU {gpu_id}...")
    
    # Load models specific to this GPU
    main_model, main_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_crime_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map=device,
    )

    finance_model, finance_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_finance_crime_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map=device,
    )

    women_model, women_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_women_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map=device,
    )

    cyber_attack_model, cyber_attack_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_cyber_attack_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map=device,
    )

    other_model, other_tokenizer = FastModel.from_pretrained(
        model_name="gemma3_other_classifier",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map=device,
    )

    # Apply Gemma-3 chat template to all tokenizers
    main_tokenizer = get_chat_template(main_tokenizer, chat_template="gemma-3")
    finance_tokenizer = get_chat_template(finance_tokenizer, chat_template="gemma-3")
    women_tokenizer = get_chat_template(women_tokenizer, chat_template="gemma-3")
    cyber_attack_tokenizer = get_chat_template(cyber_attack_tokenizer, chat_template="gemma-3")
    other_tokenizer = get_chat_template(other_tokenizer, chat_template="gemma-3")
    
    return {
        "main": (main_model, main_tokenizer),
        "finance": (finance_model, finance_tokenizer),
        "women": (women_model, women_tokenizer),
        "cyber_attack": (cyber_attack_model, cyber_attack_tokenizer),
        "other": (other_model, other_tokenizer),
        "device": device
    }

# Function to classify main category in batch
def classify_main_category_batch(crime_infos, model, tokenizer, instruction, device):
    messages_list = []
    for crime_info in crime_infos:
        messages = [{
            "role": "user",
            "content": f"{instruction}\n\nCrime Information: {crime_info}"
        }]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        messages_list.append(text)
    
    inputs = tokenizer(messages_list, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    
    results = []
    for output in outputs:
        full_output = tokenizer.decode(output, skip_special_tokens=True)
        try:
            parts = full_output.split("model\n")
            if len(parts) > 1:
                category = parts[-1].strip().split("\n")[0].strip()
                results.append(category)
            else:
                results.append("Error: No category found")
        except Exception as e:
            results.append(f"Error: {str(e)}")
    
    return results

# Function to classify subcategory in batch
def classify_subcategory_batch(crime_infos, model, tokenizer, instruction, device):
    if not crime_infos:
        return []
        
    messages_list = []
    for crime_info in crime_infos:
        messages = [{
            "role": "user",
            "content": f"{instruction}\n\nCrime Information: {crime_info}"
        }]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        messages_list.append(text)
    
    inputs = tokenizer(messages_list, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
    
    results = []
    for output in outputs:
        full_output = tokenizer.decode(output, skip_special_tokens=True)
        try:
            parts = full_output.split("model\n")
            if len(parts) > 1:
                subcategory = parts[-1].strip().split("\n")[0].strip()
                results.append(subcategory)
            else:
                results.append("Error: No subcategory found")
        except Exception as e:
            results.append(f"Error: {str(e)}")
    
    return results

# Process a chunk of data on a specific GPU
def process_chunk(chunk_data, gpu_id, chunk_id, output_dir):
    # Load models for this GPU
    models = load_models(gpu_id)
    device = models["device"]
    
    # Unpack models
    main_model, main_tokenizer = models["main"]
    finance_model, finance_tokenizer = models["finance"]
    women_model, women_tokenizer = models["women"]
    cyber_attack_model, cyber_attack_tokenizer = models["cyber_attack"]
    other_model, other_tokenizer = models["other"]
    
    # Get crime info list
    crime_infos = chunk_data['crimeaditionalinfo'].tolist()
    total_records = len(crime_infos)
    
    # Process in batches
    all_results = []
    
    for i in tqdm(range(0, total_records, BATCH_SIZE), 
                 desc=f"GPU {gpu_id} - Chunk {chunk_id} Progress", 
                 position=gpu_id):
        batch = crime_infos[i:min(i + BATCH_SIZE, total_records)]
        
        # Classify main categories
        categories = classify_main_category_batch(batch, main_model, main_tokenizer, MAIN_INSTRUCTION, device)
        
        # Group by category
        financial_crimes = []
        financial_indices = []
        women_children_crimes = []
        women_indices = []
        cyber_attack_crimes = []
        cyber_indices = []
        other_crimes = []
        other_indices = []
        
        for idx, (crime_info, category) in enumerate(zip(batch, categories)):
            if category == "financial_crimes":
                financial_crimes.append(crime_info)
                financial_indices.append(idx)
            elif category == "crime_against_women_and_children":
                women_children_crimes.append(crime_info)
                women_indices.append(idx)
            elif category == "cyber_attack_or_dependent_crimes":
                cyber_attack_crimes.append(crime_info)
                cyber_indices.append(idx)
            elif category == "other_cyber_crime":
                other_crimes.append(crime_info)
                other_indices.append(idx)
            else:
                # For errors or unrecognized categories
                all_results.append({"category": category, "subcategory": "Error: Invalid category"})
        
        # Process subcategories in parallel for each group
        batch_results = [None] * len(batch)
        
        # Process each category if there are any entries
        if financial_crimes:
            financial_subcategories = classify_subcategory_batch(
                financial_crimes, finance_model, finance_tokenizer, FINANCE_INSTRUCTION, device
            )
            for i, idx in enumerate(financial_indices):
                batch_results[idx] = {"category": "financial_crimes", "subcategory": financial_subcategories[i]}
        
        if women_children_crimes:
            women_subcategories = classify_subcategory_batch(
                women_children_crimes, women_model, women_tokenizer, WOMEN_INSTRUCTION, device
            )
            for i, idx in enumerate(women_indices):
                batch_results[idx] = {"category": "crime_against_women_and_children", "subcategory": women_subcategories[i]}
        
        if cyber_attack_crimes:
            cyber_subcategories = classify_subcategory_batch(
                cyber_attack_crimes, cyber_attack_model, cyber_attack_tokenizer, CYBER_ATTACK_INSTRUCTION, device
            )
            for i, idx in enumerate(cyber_indices):
                batch_results[idx] = {"category": "cyber_attack_or_dependent_crimes", "subcategory": cyber_subcategories[i]}
        
        if other_crimes:
            other_subcategories = classify_subcategory_batch(
                other_crimes, other_model, other_tokenizer, OTHER_INSTRUCTION, device
            )
            for i, idx in enumerate(other_indices):
                batch_results[idx] = {"category": "other_cyber_crime", "subcategory": other_subcategories[i]}
        
        # Add results that are not None
        all_results.extend([r for r in batch_results if r is not None])
    
    # Add results to dataframe
    result_df = pd.DataFrame(all_results)
    output_df = chunk_data.copy()
    output_df['category'] = result_df['category']
    output_df['subcategory'] = result_df['subcategory']
    
    # Save chunk results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"chunk_{chunk_id}_gpu_{gpu_id}.csv")
    output_df.to_csv(output_file, index=False)
    
    print(f"GPU {gpu_id} completed chunk {chunk_id}, saved to {output_file}")
    return output_file

# Main function
def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Crime Classification')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--chunk_size', type=int, default=25000, help='Records per chunk')
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    df = pd.read_csv(args.input)
    total_records = len(df)
    print(f"Total records: {total_records}")
    
    # Split data into chunks
    num_chunks = (total_records + args.chunk_size - 1) // args.chunk_size
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * args.chunk_size
        end_idx = min((i + 1) * args.chunk_size, total_records)
        chunks.append(df.iloc[start_idx:end_idx].copy())
    
    print(f"Split data into {len(chunks)} chunks")
    
    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Distribute chunks across GPUs
    chunk_assignments = []
    for i, chunk in enumerate(chunks):
        gpu_id = i % args.num_gpus
        chunk_assignments.append((chunk, gpu_id, i, args.output))
    
    # Group chunks by GPU for sequential processing on each GPU
    gpu_tasks = [[] for _ in range(args.num_gpus)]
    for chunk, gpu_id, chunk_id, output_dir in chunk_assignments:
        gpu_tasks[gpu_id].append((chunk, gpu_id, chunk_id, output_dir))
    
    # Process chunks on each GPU
    processes = []
    output_files = []
    
    for gpu_id, tasks in enumerate(gpu_tasks):
        if not tasks:
            continue
            
        # Create a process for each GPU
        p = mp.Process(target=process_gpu_tasks, args=(tasks, gpu_id))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Combine all output files
    print("Combining results...")
    output_files = []
    for root, _, files in os.walk(args.output):
        for file in files:
            if file.startswith("chunk_") and file.endswith(".csv"):
                output_files.append(os.path.join(root, file))
    
    combined_df = pd.concat([pd.read_csv(f) for f in output_files])
    combined_output = os.path.join(args.output, "combined_results.csv")
    combined_df.to_csv(combined_output, index=False)
    
    end_time = time.time()
    print(f"Processing complete! Total time: {(end_time - start_time)/60:.2f} minutes")
    print(f"Results saved to {combined_output}")

# Function to process all tasks assigned to a GPU
def process_gpu_tasks(tasks, gpu_id):
    for chunk, assigned_gpu_id, chunk_id, output_dir in tasks:
        process_chunk(chunk, assigned_gpu_id, chunk_id, output_dir)

# Script to merge output files
def merge_outputs(output_dir):
    # Find all chunk files
    output_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.startswith("chunk_") and file.endswith(".csv"):
                output_files.append(os.path.join(root, file))
    
    # Check if any files were found
    if not output_files:
        print("No chunk files found to merge!")
        return
        
    # Merge all files
    combined_df = pd.concat([pd.read_csv(f) for f in output_files])
    combined_output = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_output, index=False)
    print(f"Merged {len(output_files)} files into {combined_output}")

if __name__ == "__main__":
    main()
