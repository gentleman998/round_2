# Import necessary libraries
from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
import pandas as pd

# Load all models and tokenizers
main_model, main_tokenizer = FastModel.from_pretrained(
    model_name="gemma3_crime_classifier",
    max_seq_length=2048,
    load_in_4bit=True,
)

finance_model, finance_tokenizer = FastModel.from_pretrained(
    model_name="gemma3_finance_crime_classifier",
    max_seq_length=2048,
    load_in_4bit=True,
)
women_model, women_tokenizer = FastModel.from_pretrained(
    model_name="gemma3_women_classifier",
    max_seq_length=2048,
    load_in_4bit=True,
)
cyber_attack_model, cyber_attack_tokenizer = FastModel.from_pretrained(
    model_name="gemma3_cyber_attack_classifier",
    max_seq_length=2048,
    load_in_4bit=True,
)
other_model, other_tokenizer = FastModel.from_pretrained(
    model_name="gemma3_other_classifier",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply Gemma-3 chat template to all tokenizers
main_tokenizer = get_chat_template(main_tokenizer, chat_template="gemma-3")
finance_tokenizer = get_chat_template(finance_tokenizer, chat_template="gemma-3")
women_tokenizer = get_chat_template(women_tokenizer, chat_template="gemma-3")
cyber_attack_tokenizer = get_chat_template(cyber_attack_tokenizer, chat_template="gemma-3")
other_tokenizer = get_chat_template(other_tokenizer, chat_template="gemma-3")

# Instructions for classification
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

# Function to classify main category
def classify_main_category(crime_info, model, tokenizer, instruction):
    messages = [{
        "role": "user",
        "content": f"{instruction}\n\nCrime Information: {crime_info}"
    }]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
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

# Function to classify subcategory
def classify_subcategory(crime_info, model, tokenizer, instruction):
    messages = [{
        "role": "user",
        "content": f"{instruction}\n\nCrime Information: {crime_info}"
    }]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
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

# Modified function to return a dictionary instead of JSON
def get_crime_classification(crime_info):
    # Step 1: Get main category
    category = classify_main_category(crime_info, main_model, main_tokenizer, MAIN_INSTRUCTION)
    
    # Step 2: Get subcategory based on category
    if category == "financial_crimes":
        subcategory = classify_subcategory(crime_info, finance_model, finance_tokenizer, FINANCE_INSTRUCTION)
    elif category == "crime_against_women_and_children":
        subcategory = classify_subcategory(crime_info, women_model, women_tokenizer, WOMEN_INSTRUCTION)
    elif category == "cyber_attack_or_dependent_crimes":
        subcategory = classify_subcategory(crime_info, cyber_attack_model, cyber_attack_tokenizer, CYBER_ATTACK_INSTRUCTION)
    elif category == "other_cyber_crime":
        subcategory = classify_subcategory(crime_info, other_model, other_tokenizer, OTHER_INSTRUCTION)
    else:
        subcategory = "Error: Invalid category"
    
    # Step 3: Return as dictionary
    return {"category": category, "subcategory": subcategory}

# Load the CSV file (replace 'your_csv_file.csv' with the actual file path)
df = pd.read_csv('your_csv_file.csv')

# Apply the classification to the 'crimeaditionalinfo' column
classification_results = df['crimeaditionalinfo'].apply(get_crime_classification)

# Add the 'category' and 'subcategory' columns to the DataFrame
df[['category', 'subcategory']] = classification_results.apply(pd.Series)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_crimes.csv', index=False)

# Optional: Print the first few rows to verify
print(df.head())
