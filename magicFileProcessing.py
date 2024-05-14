import streamlit as st
import pandas as pd
import openai
from datetime import timezone 
import datetime 
import json
import time

OPENAI_API_KEY = '' 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

system_prompt = f"""
    Extract and return payroll details for each indivdual employee listed in the data. 
    
    For each employee, retrieve the following eight key values:
        1. employee_name: The full name of employee. 
        2. gross_salary: The basic pay or salary before any deductions. This is the total amount earned by the employee before tax or other deductions. 
        3. claims: List of individual claims, each with a type and amount. If there are no claims, return an empty list. 
        4. SHG: The total SHG amount for this employee, broken down into MBMD, SINDA, CDAC, and ECF contributions. If no value, return 0. 
        5. employee_cpf: The CPF (Central Provident Fund) amount contributed by this employee. If no value, return 0. 
        6. net_payable_to_employee: The net amount payable to this employee.
        7. employer_cpf: The CPF amount contributed by the employer for this employee. If no value, return 0. 
        8. other_employer_contributions: The SDL (Skills Development Levy) amount contributed by the employer for this employee. If no value, return 0. 

    Ensure all returned values are in absolute terms (no '+' or '-').

    Include the payroll date in the format 'Month YYYY' (e.g., 'Jan 2023').

    Present the data in the following structured JSON format:
    ```
    {{
        "employees": [
            {{
                "employee_name": "Name", 
                "gross_salary": "Amount",
                "claims": [ 
                    {{
                        "type_of_claim": "Type",
                        "claim_amount": "Amount"
                    }} 
                ],
                "SHG": "Amount",
                "employee_cpf": "Amount",
                "net_payable_to_employee": "Amount",
                "employer_cpf": "Amount",
                "other_employer_contributions": "Amount"
            }}
        ],
        "date": "Month YYYY"
    }}
    ```
    
    Response must only follow the provided structure and not deviate from it. You must return a valid json with the provided strucutre. 
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
            # model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled accountant with comprehensive knowledge in all aspects of payroll accounting. \
                 You will be analyzing financial payroll files, ensuring accuracy and compliance with the stated requirements."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
            max_tokens=900
        )
        print(response)
        st.write(response.usage)
        content = response.choices[0].message.content
        content = content.strip("```").strip()
        content = content.strip("json").strip()
        print("content: ", content)
        try: 
            json_content = json.loads(content) 
        except json.JSONDecodeError as e:
            content = content.strip("```").strip()
            json_content = json.loads(content) 
        st.write("response: ", json_content)
        return json_content
    except Exception as e:
        return str(e)
    

def convert_to_df_employees(employees):
    print("employees: ", employees)

    df = pd.json_normalize(employees)
    df['total_claims'] = df['claims'].apply(lambda x: sum(float(claim['claim_amount']) for claim in x))
    df = df.apply(pd.to_numeric, errors='ignore')
    print("df", df)

    claims_details = pd.DataFrame(columns=['employee_name', 'type_of_claim', 'claim_amount'])
    for index, row in df.iterrows():
        if isinstance(row['claims'], list):
            for claim in row['claims']:
                claims_details = claims_details.append({
                    'employee_name': row['employee_name'],
                    'type_of_claim': claim['type_of_claim'],
                    'claim_amount': claim['claim_amount']
                }, ignore_index=True)
    print("claims", claims_details)

    output_df = df.drop(columns='claims')
    output_df['computed_take_home'] = output_df['gross_salary'] + output_df['total_claims'] - output_df['SHG'] - output_df['employee_cpf']
    output_df.loc['Total']= output_df.sum()
    output_df.loc[output_df.index[-1], 'employee_name'] = 'Total'
    
    return output_df, claims_details

def calculate_cost_to_company(employees_df):
    # take home pay + employer cpf + other employer contributions + shg + employee cpf
    net_payable_to_employee = employees_df.at['Total', "net_payable_to_employee"]
    employer_cpf = employees_df.at['Total', "employer_cpf"]
    other_employer_contributions = employees_df.at['Total', "other_employer_contributions"]
    SHG = employees_df.at['Total', "SHG"]
    employee_cpf = employees_df.at['Total', "employee_cpf"]
    cost_to_company = net_payable_to_employee + employer_cpf + other_employer_contributions + SHG + employee_cpf
    return cost_to_company

# def convert_to_journals(employees_df, claims_details):


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
            start_time = time.time()

            dt = datetime.datetime.now(timezone.utc) 
            utc_time = dt.replace(tzinfo=timezone.utc) 
            utc_timestamp = utc_time.timestamp()

            df = pd.read_excel(uploaded_file, sheet_name=None) # some excel files have multiple sheets
            response = call_gpt(df, system_prompt, custom_prompt) 
            employees_df, claims_details = convert_to_df_employees(response['employees'])
            cost_to_company = calculate_cost_to_company(employees_df)

            st.write(f"Analysis for {uploaded_file.name}:")
            st.info(f"Payroll for {response['date']}")
            st.write(employees_df)
            st.write(claims_details)
            st.warning(f"Cost to company = {cost_to_company}")
            st.write(f"utc timestamp: {utc_timestamp}")
            st.write("time taken: ", str(round((time.time() - start_time), 2)) + " seconds")

if __name__ == "__main__":
    main()
