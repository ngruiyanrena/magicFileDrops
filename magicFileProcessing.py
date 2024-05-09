import streamlit as st
import pandas as pd
import openai
from datetime import timezone 
import datetime 
import json
import numpy as np 

OPENAI_API_KEY = '' 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

system_prompt = f"""
    Extract and return payroll details for each indivdual employee listed in the data. 
    
    For each employee, retrieve the following six key values:
        1. employee_name: The full name of employee. 
        2. gross_salary: The basic pay or salary before any deductions. This is the total amount earned by the employee before tax or other deductions. 
        3. additions: The total of all bonuses, reimbursements, or claims, including meal and transport allowances, added to the basic pay.
        4. deductions: 
            - The total of all deductions from the salary, including employee contributions such as CPF, CDAC, ECF, or other voluntary deductions. 
            - DO NOT include SDL or employer CPF contribution. 
        5. clawbacks: The total of all deductions for leave payment, including no pay leave, or any unpaid leaves taken by the employee.
        6. take_home_salary: The net salary amount that the employee receives after all additions and deductions. Ensure the value is directly extracted from the data. Do not perform any calculations for this field.

     Compliance checks:
    - Verify that the following equation holds true for each record: take_home_salary = gross_salary + additions - deductions - clawbacks.
    - If the equation does not balance, review all components for accuracy.
    - Present all values in absolute terms (no '+' or '-'). Use '0' for any missing data.
    
    Present the data in the following structured JSON format:
    ```
    {{
        "employees": [
            {{
                "employee_name": "xxx", 
                "gross_salary": "xxx",
                "additions": "xxx",
                "deductions": "xxx",
                "clawbacks": "xxx",
                "take_home_salary": "xxx"
            }}
        ],
        "date": "xxx" // Find month and year this payroll file is for. eg. "Jan 2023"
    }}
    ```
    
    Response should only follow the provided structure and not deviate from it. Return a valid json. 
    """



def call_gpt(df, prompt, user_prompt):
    try:
        full_prompt = f"""
            Here is the payroll file to analyze. 
            {df}
            {prompt}
            Additional User Instructions: {user_prompt}
        """
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled accountant with comprehensive knowledge in all aspects of payroll accounting. \
                 You will be analyzing financial payroll files, ensuring accuracy and compliance with the stated requirements."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )

        content = response.choices[0].message.content
        content = content.strip("```").strip()
        content = content.strip("json")
        json_content = json.loads(content) 
        print("response: ", json_content)
        return json_content
    except Exception as e:
        return str(e)
    

def convert_to_df_employees(employees):
    print("employees: ", employees)
    df = pd.DataFrame(employees)

    columns = ["gross_salary", "additions", "deductions", "clawbacks", "take_home_salary"]
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

    total = df.select_dtypes(include=[np.number]).sum()
    total['employee_name'] = "Total"
    df = df.append(total, ignore_index=True)

    output_df = pd.DataFrame({
            "Name of Employee": df["employee_name"],
            "Gross Salary": df["gross_salary"],
            "Add: Additions": df["additions"],
            "Less: Deductions": df["deductions"],
            "Less: Clawbacks/Paybacks/NoPay Leave": df["clawbacks"],
            "= Take-home Salary": df["take_home_salary"]
        })
    
    return output_df

# def convert_to_df_summary_totals(summary_totals):
#     print("summary totals: ", summary_totals)
#     df = pd.DataFrame(summary_totals)

#     output_df = pd.DataFrame({
#                 "Add: Deductions": df["deductions"],
#                 "Add: Employer Contributions": df["employer_contributions"],
#                 "= Wages Payable": df["wages_payable"]
#             })
    
#     return output_df


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
            dt = datetime.datetime.now(timezone.utc) 
            utc_time = dt.replace(tzinfo=timezone.utc) 
            utc_timestamp = utc_time.timestamp()

            df = pd.read_excel(uploaded_file, sheet_name=None) # some excel files have multiple sheets

            response = call_gpt(df, system_prompt, custom_prompt) 
            processed_data_employees = convert_to_df_employees(response['employees'])
            # processed_data_summary_totals = convert_to_df_summary_totals(response['summaryTotals'])

            st.write(f"Analysis for {uploaded_file.name}:")
            st.info(f"Payroll for {response['date']}")
            st.write(processed_data_employees)
            # st.write(processed_data_summary_totals)
            st.write(f"utc timestamp: {utc_timestamp}")

if __name__ == "__main__":
    main()
