import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os


load_dotenv('.env')
OPENAI_API_KEY : str = os.getenv('OPENAI_API_KEY')


client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

system_prompt = f"""
    Universally, payrolls follow a standard bridge for each period.  

    Gross Salary 
    + Additions (e.g. bonuses, employee claims reimbursements, commissions)
    + Deductions (e.g. employee statutory contributions of all types, voluntary contributions of all types, income taxes withheld)
    = Take-home Salary
    + Employer Contributions 
    = Wages Payable (total cost to company)

    Provide a structured response using the specified format below:
    ```
    {{
        "operationName": "CreateJournal",
        "contactResourceId": "xxx",
        "valueDate": "xxx",
        "internalNotes": "Test",
        "journalEntries: [ {{
            "organizationAccountResourceId": "xxx",
            "debitAmount": "",
            "creditAmount": "",
            "description": "xxx",
            "organizationResourceId": "xxx"
        }} ],
        "reference": "JournalRef",
        "status": "ACTIVE",
        "tags": null,
        "type": "JOURNAL_MANUAL"
    }}
    ```
    Your response should only follow this structure and should not deviate from it.
    """



def call_gpt(df, prompt, user_prompt):
    try:
        full_prompt = f"""
            {prompt}
            
            Additional User Instructions: {user_prompt}

            Here is the file to create the journal entry. If there are multiple journal entry lines, create multiple journal entries under journalEntries. 
            {df}
        """
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled accountant, with knowledge in all aspects of accounting and will be assisting with analyzing financial payroll files."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def main():
    st.title("Smart Journal Entry Maker")


    uploaded_files = st.file_uploader("Upload payroll Excel files", type=['xlsx'], accept_multiple_files=True)

    if st.sidebar.checkbox("Enable prompt editing"):
        default_prompt = "Please analyze the payroll data."
        custom_prompt = st.sidebar.text_area("Edit GPT prompt:", value=default_prompt)
    else:
        custom_prompt = None


    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = pd.read_excel(uploaded_file)
            response = call_gpt(df, system_prompt, custom_prompt) 
            st.write(f"Analysis for {uploaded_file.name}:")
            st.write(response)

if __name__ == "__main__":
    main()
