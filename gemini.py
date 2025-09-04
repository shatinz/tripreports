import google.generativeai as genai
import os
from config import GOOGLE_API_KEY



# Configure the Gemini API key
# Ensure you have the API key stored in an environment variable or replace 'YOUR_API_KEY'
genai.configure(api_key = (GOOGLE_API_KEY))

# Specify the Gemini model to be used
model_name = "gemini-1.5-flash"



import json

with open('cleaned_reports.json', 'r') as f:
    cleaned_reports = json.load(f)



schema_fields = [
    "previous_experience",
    "set_and_setting",
    "before_after_changes",
    "intention",
    "experience_phases",
    "onset_description",
    "perceived_realness",
    "objective_elements",
    "entities_or_other_beings",
    "childhood_trauma",
    "sex_effects",
    "self_love_experience",
    "most_important_element",
    "experiencing_fear"
]

extraction_prompt = f"""
You are an AI assistant tasked with extracting specific information from a drug experience report.
Your goal is to read the provided report text and extract information for the following fields:
{', '.join(schema_fields)}

For each field, provide a concise answer directly related to the field name.
If the information for a field is not present in the report, state "Not specified".

Here is the report text:
{{report_text}}

Please provide the extracted information in a structured format (e.g., JSON or a clear list of key-value pairs).
"""

print(extraction_prompt)


import time

extracted_data_from_llm = []
# A rough estimate of token limit for Gemini 1.5 Flash, considering prompt size and response size
# This might need adjustment based on actual usage and model capabilities
TOKEN_LIMIT = 10000  # Example token limit, adjust as needed
SLEEP_TIME = 5  # seconds

for report_data in cleaned_reports:
    link = report_data['link']
    report_text = report_data['report_text']
    report_chunks = []

    # Simple chunking mechanism based on character count as a proxy for tokens
    # A more sophisticated approach would use a proper tokenizer
    if len(report_text) > TOKEN_LIMIT * 0.8: # Chunk if text is close to the limit
        # Split into chunks roughly based on token limit
        chunk_size = int(TOKEN_LIMIT * 0.7) # Make chunks smaller than the limit
        report_chunks = [report_text[i:i + chunk_size] for i in range(0, len(report_text), chunk_size)]
        print(f"Report {link} chunked into {len(report_chunks)} parts.")
    else:
        report_chunks = [report_text]
        print(f"Report {link} does not require chunking.")

    for i, chunk in enumerate(report_chunks):
        print(f"Processing chunk {i+1}/{len(report_chunks)} for report {link}")
        prompt = extraction_prompt.format(report_text=chunk)

        try:
            # Send prompt to Gemini API
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            # Process the response
            # Assuming the response is in a format that can be directly used or easily parsed
            # This part might need significant adjustment based on Gemini's actual output format
            extracted_info = {
                'link': link,
                'chunk_index': i,
                'extracted_data': response.text # Store the raw response text for now
            }
            extracted_data_from_llm.append(extracted_info)
            print(f"Successfully processed chunk {i+1} for {link}")

        except Exception as e:
            print(f"Error processing chunk {i+1} for {link}: {e}")
            extracted_data_from_llm.append({
                'link': link,
                'chunk_index': i,
                'extracted_data': f"Error: {e}"
            })

        # Implement sleep time
        time.sleep(SLEEP_TIME)
        print(f"Sleeping for {SLEEP_TIME} seconds.")

# The extracted_data_from_llm list now contains the results for all reports and chunks
# You can further process or save this list as needed
# For this subtask, we just populate the list.