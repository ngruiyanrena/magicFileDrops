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
    I require two extraction and fill formats: (i) individual lines (per employee) and (ii) summary totals (aggregate).
    
    Universally, payrolls follow a standard bridge for each period.  
    ```
    Gross Salary 
    Add: Additions (e.g. bonuses, employee claims reimbursements, commissions)
    Less: Deductions (e.g. employee statutory contributions of all types, voluntary contributions of all types, income taxes withheld)
    Less: Clawbacks/Paybacks/NoPay Leave
    = Take-home Salary

    Add: Deductions (from above)
    Add: Employer Contributions
    = Wages Payable ("total cost to company")
    ```

    For each individual line per employee, provide it in this specified json structure as follows:
    Return only absolute values (i.e no '+' or '-'). 
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
        ]
    }}
    ```

    For each summary total, provide it in this specified json strucutre as follows:
    Return only absolute values (i.e no '+' or '-'). 
    ```
    {{
        "summaryTotals": [
            {{
                "deductions": "xxx",
                "employer_contributions": "xxx",
                "wages_payable": "xxx"
            }}
        ]
    }}
    ```
 
    Find month and year this payroll file is for. 
    ```
    {{
        "date": "Jan 2023"
    }}
    ```
    
    Response should only follow the provided structures and not deviate from it. Return only a combined valid json. 
    """



def call_gpt(df, prompt, user_prompt):
    try:
        full_prompt = f"""
            {prompt}
            
            Additional User Instructions: {user_prompt}

            Here is the file to create the journal entry. 
            {df}
        """
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled accountant, with knowledge in all aspects of accounting. \
                 You will be analyzing financial payroll files and creating journal entires for the company."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
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

    output_df = pd.DataFrame({
            "Name of Employee": df["employee_name"],
            "Gross Salary": df["gross_salary"],
            "Add: Additions": df["additions"],
            "Less: Deductions": df["deductions"],
            "Less: Clawbacks/Paybacks/NoPay Leave": df["clawbacks"],
            "= Take-home Salary": df["take_home_salary"]
        })
    
    return output_df

def convert_to_df_summary_totals(summary_totals):
    print("summary totals: ", summary_totals)
    df = pd.DataFrame(summary_totals)

    output_df = pd.DataFrame({
                "Add: Deductions": df["deductions"],
                "Add: Employer Contributions": df["employer_contributions"],
                "= Wages Payable": df["wages_payable"]
            })
    
    return output_df

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
            processed_data_summary_totals = convert_to_df_summary_totals(response['summaryTotals'])

            st.write(f"Analysis for {uploaded_file.name}:")
            st.info(f"Payroll for {response['date']}")
            st.write(processed_data_employees)
            st.write(processed_data_summary_totals)
            st.write(f"utc timestamp: {utc_timestamp}")

if __name__ == "__main__":
    main()
