import streamlit as st
import pandas as pd
import openai
from datetime import timezone 
import datetime 
import json

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
            - Important: DO NOT include SDL or employer CPF contribution. 
        5. clawbacks: The total of all deductions for leave payment, including no pay leave, or any unpaid leaves taken by the employee.
        6. take_home_salary: The net salary amount that the employee receives after all additions and deductions. Ensure this value is directly extracted from the data. Do not perform any calculations for this field.

    This equation MUST hold: take_home_salary = gross_salary + additions - deductions - clawbacks. If equation does not hold, recheck numbers in all components.
    Ensure all returned values are in absolute terms (no '+' or '-'). 
    If a particular value is not available, return 0 for that field. 
    
    For these 2 fields: 
        1. employer_contributions: 
            - The total of all employer contributions for all employees, including CPF (OW and AW), SHG, and SDL contributions.
            - DO NOT include employee CPF contribution. 
        2. date: The month and year this payroll file is for. eg. "Jan 2023"

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
        "employer_contributions": "xxx",
        "date": "xxx"
    }}
    ```
    
    Response must only follow the provided structure and not deviate from it. Return a valid json. 
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
            temperature=0,
            max_tokens=600
        )

        content = response.choices[0].message.content
        content = content.strip("```").strip()
        content = content.strip("json")
        json_content = json.loads(content) 
        st.write("response: ", json_content)
        return json_content
    except Exception as e:
        return str(e)
    

def convert_to_df_employees(employees):
    print("employees: ", employees)
    df = pd.DataFrame(employees)

    columns = ["gross_salary", "additions", "deductions", "clawbacks", "take_home_salary"]
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

    df.loc['Total']= df.sum()
    df.loc[df.index[-1], 'employee_name'] = 'Total'

    output_df = pd.DataFrame({
            "Name of Employee": df["employee_name"],
            "Gross Salary": df["gross_salary"],
            "Add: Additions": df["additions"],
            "Less: Deductions": df["deductions"],
            "Less: Clawbacks/Paybacks/NoPay Leave": df["clawbacks"],
            "= Take-home Salary": df["take_home_salary"]
        })
    
    return output_df

def calculate_wages_payable(employees_df, employer_contributions):
    employer_contributions = pd.to_numeric(employer_contributions)
    take_home_salary = employees_df.at['Total', "= Take-home Salary"]
    deductions = employees_df.at['Total', "Less: Deductions"]
    wages_payable = take_home_salary + deductions + employer_contributions
    return wages_payable



pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 

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
            wages_payable = calculate_wages_payable(processed_data_employees, response['employer_contributions'])

            st.write(f"Analysis for {uploaded_file.name}:")
            st.info(f"Payroll for {response['date']}")
            st.write(processed_data_employees)
            st.warning(f"Wages Payable (total cost to company) = {wages_payable}")
            st.write(f"utc timestamp: {utc_timestamp}")

if __name__ == "__main__":
    main()
