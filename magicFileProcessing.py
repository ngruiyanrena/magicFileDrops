import streamlit as st
import pandas as pd
import openai
from datetime import timezone
from datetime import datetime 
import json
import time
import numpy as np

OPENAI_API_KEY = '' 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

payboy_prompt = f"""
    For each column name from a payroll data file, assign it to the most appropriate category from the list below. 
    If a column does not fit any of these categories, exclude it from the output. 
    Ensure the response adheres to the specified JSON structure without deviation.

    Categories:
        1. employee_name: Full name of the employee.
        2. gross_salary: Amount earned by the employee before tax or other deductions
        3. bonus: On top of gross salary. 'Pay Item: Bonus', 'Pay Item: Fixed Bonus'
        4. deductions: Total of Leave Payments or Deductions. 'Unpaid leaves deductions'. Do not include 'Pay Item: xxx'. 
        5. claims: If no claims column exists, this should be an empty list in the JSON. 'Claim Type: xxx'
        6. SHG (Self-Help Group): Columns that indicate SHG amount.
        7. employee_cpf: CPF contributions by the employee.
        8. net_payable_to_employee: Net salary payable to the employee.
        9. employer_cpf: CPF contributions by the employer.
        10. other_employer_contributions: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy).

    For each category, if no column name matches, return empty list. 

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "bonus": ["column_name"],
        "deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "SHG": ["column_name", ...],
        "employee_cpf": ["column_name"],
        "net_payable_to_employee": ["column_name"],
        "employer_cpf": ["column_name"],
        "other_employer_contributions": ["column_name", ...]
    }}
    ```
    
    Provide the column names in a list format and adhere to the structure above to map each column to its appropriate category.
    """

talenox_prompt = f"""
    For each column name from a payroll data file, assign it to the most appropriate category from the list below. 
    If a column does not fit any of these categories, exclude it from the output. 
    Ensure the response adheres to the specified JSON structure without deviation.

    Categories:
        1. employee_name: Full name of the employee.
        2. gross_salary: Use "total recurring full" column instead of 'gross salary'.
        3. bonus: Use 'total adhoc' instead of 'total bonus'.
        4. deductions: Total of Leave Payments or Deductions.
        5. claims: If no claims column exists, this should be an empty list in the JSON.
        6. SHG (Self-Help Group): Columns that indicate SHG amount. Not the type. 
        7. employee_cpf: CPF contributions by the employee, excluding year-to-date figures. 
        8. net_payable_to_employee: Net salary payable to the employee.
        9. employer_cpf: CPF contributions by the employer, exclude year-to-date and total amount columns.
        10. other_employer_contributions: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy).

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "bonus": ["column_name"],
        "deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "SHG": ["column_name", ...],
        "employee_cpf": ["column_name"],
        "net_payable_to_employee": ["column_name"],
        "employer_cpf": ["column_name"],
        "other_employer_contributions": ["column_name", ...]
    }}
    ```
    
    Provide the column names in a list format and adhere to the structure above to map each column to its appropriate category.
    """


hreasily_prompt = f"""
    For each column name from a payroll data file, assign it to the most appropriate category from the list below. 
    If a column does not fit any of these categories, exclude it from the output. 
    Ensure the response adheres to the specified JSON structure without deviation.

    Categories:
        1. employee_name: Full name of the employee. Do not include foreign name. 
        2. gross_salary: Basic salary. Amount earned by the employee before tax or other deductions. 
        3. bonus: On top of gross salary. 
        4. deductions: Total of Leave Payments or Deductions. SSO employee contribution, tax contribution. 
        5. claims: If no claims column exists, this should be an empty list in the JSON. 
        6. SHG (Self-Help Group): Columns that indicate SHG amount.
        7. employee_cpf: CPF contributions by the employee.
        8. net_payable_to_employee: Net salary payable to the employee.
        9. employer_cpf: CPF contributions by the employer.
        10. other_employer_contributions: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy). sso employer contribution.

    For each category, if no column name matches, return empty list. 

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "bonus": ["column_name"],
        "deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "SHG": ["column_name", ...],
        "employee_cpf": ["column_name"],
        "net_payable_to_employee": ["column_name"],
        "employer_cpf": ["column_name"],
        "other_employer_contributions": ["column_name", ...]
    }}
    ```
    
    Provide the column names in a list format and adhere to the structure above to map each column to its appropriate category.
    """

other_prompt = f"""
    For each column name from a payroll data file, assign it to the most appropriate category from the list below. 
    If a column does not fit any of these categories, exclude it from the output. 
    Ensure the response adheres to the specified JSON structure without deviation.

    Categories:
        1. employee_name: Full name of the employee. 
        2. gross_salary: Basic salary. Amount earned by the employee before tax or other deductions. 
        3. bonus: On top of gross salary. 
        4. deductions: Total of Leave Payments or Deductions. 
        5. claims: If no claims column exists, this should be an empty list in the JSON. 
        6. SHG (Self-Help Group): Columns that indicate SHG amount.
        7. employee_cpf: CPF contributions by the employee.
        8. net_payable_to_employee: Net salary payable to the employee.
        9. employer_cpf: CPF contributions by the employer.
        10. other_employer_contributions: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy).

    For each category, if no column name matches, return empty list. 

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "bonus": ["column_name"],
        "deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "SHG": ["column_name", ...],
        "employee_cpf": ["column_name"],
        "net_payable_to_employee": ["column_name"],
        "employer_cpf": ["column_name"],
        "other_employer_contributions": ["column_name", ...]
    }}
    ```
    
    Provide the column names in a list format and adhere to the structure above to map each column to its appropriate category.
    """

def call_gpt(column_names, system_prompt, user_prompt):
    try:
        full_prompt = f"""
            Here are the column names from the payroll file to map. 
            {column_names}
            {system_prompt}
            Additional User Instructions: {user_prompt}
        """
        response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled accountant with comprehensive knowledge in payroll accounting. \
                 You will be analyzing the list of column names from the payroll file and map each to the appropriate category as per the instructions."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
            max_tokens=1000
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

    
def is_numeric(value):
    try: 
        str_value = str(value) # convert value to str
        pd.to_numeric(str_value)
        return True
    except:
        return False
    
def convert_to_numeric(value):
    try: 
        str_value = str(value) # convert value to str
        return pd.to_numeric(str_value)
    except:
        return value

def identify_numeric_column(columns, data):
    numeric_columns = []

    for column in columns:
        if column != '' and not pd.isna(column):
            non_empty_values = data[data[column] != ''][column].astype(str).tolist()

            if any(is_numeric(value) for value in non_empty_values):
                numeric_columns.append(column)

    return numeric_columns if numeric_columns else None

def extract_columns(response, table):
    columns_to_extract = []
    all_column_names = []
    for category, column_list in response.items():
        for column_name in column_list:
            columns_to_extract.append((category, column_name))
            all_column_names.append(column_name)

    extracted_table = table[all_column_names]

    multi_index = pd.MultiIndex.from_tuples(columns_to_extract, names=['Category', 'Column'])
    extracted_table.columns = multi_index

    numeric_columns = identify_numeric_column(extracted_table.columns, extracted_table)
    for numeric_column in numeric_columns:
        extracted_table[numeric_column] = extracted_table[numeric_column].apply(convert_to_numeric)
    extracted_table[numeric_columns] = extracted_table[numeric_columns].fillna(0) # replace nan with zero
    extracted_table[numeric_columns] = extracted_table[numeric_columns].abs() # make sure all numbers are absolute 

    print("extracted table: ", extracted_table)
    return extracted_table

def format_table(table):
    for column in table.columns:
        if column[0] == 'employee_name':
            continue
        if table[column].isna().all() or (table[column] == '-').all():
            table[column] = 0

    print("formatted table: ", table)
    return table

def calculate_summary_total(table):
    result = {}
    categories = [category for category in table.columns.get_level_values(0).unique() if category != 'employee_name']
    
    for category in categories:
        group = table.loc[:, category]
        group = group.apply(pd.to_numeric, errors='coerce')
        total_sum = group.sum().sum()
        result[category] = total_sum
    
    result_df = pd.DataFrame([result], columns=categories)
    print("summary total table: ", result_df)
    return result_df

def calculate_employee_totals(table):
    employee_info = table.loc[:, 'employee_name']
    employee_name = employee_info.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    result = {'employee_name': employee_name}
    categories = [category for category in table.columns.get_level_values(0).unique() if category != 'employee_name']

    for category in categories:
        group = table.loc[:, category]
        group = group.apply(pd.to_numeric, errors='coerce')
        total_by_employee = group.sum(axis=1, min_count=1)  
        result[category] = total_by_employee

    result_df = pd.DataFrame(result, columns=['employee_name'] + categories)
    print("employee totals table: ", result_df)
    return result_df


def calculate_cost_to_company(employees_df):
    # take home pay + employer cpf + other employer contributions + shg + employee cpf
    net_payable_to_employee = employees_df.at['Total', "net_payable_to_employee"]
    employer_cpf = employees_df.at['Total', "employer_cpf"]
    other_employer_contributions = employees_df.at['Total', "other_employer_contributions"]
    SHG = employees_df.at['Total', "SHG"]
    employee_cpf = employees_df.at['Total', "employee_cpf"]
    cost_to_company = net_payable_to_employee + employer_cpf + other_employer_contributions + SHG + employee_cpf
    return cost_to_company

# def convert_to_journals(table):

def load_and_combine_sheets(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    sheets = xls.sheet_names
    print("sheets: ", sheets)

    # Determine the base currency from the first sheet's name
    base_currency = sheets[0].split()[2]  # Assumes currency is the third item in the name format 'Month Year (Currency)'
    base_currency = base_currency.strip("()")
    print("base currency: ", base_currency)

    # Filter sheet names for those containing the base currency or conversion like "to SGD"
    relevant_sheets = [sheet for sheet in sheets if base_currency in sheet]
    print("relevant sheets: ",relevant_sheets)

    dfs = []
    for sheet_name in relevant_sheets:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        dfs.append(df)
        dfs.append(pd.DataFrame([np.nan]))  # Add an empty row 
    
    if dfs:
        dfs.pop()
    
    combined_df = pd.concat(dfs, ignore_index=True) # Concatenate all dataframes into one dataframe
    
    return combined_df

def extract_tables_from_csv(df):
    empty_rows = df.index[df.isnull().all(axis=1)] # Identify the indices where all columns in a row have NaN values
    print("empty rows: ", empty_rows)

    tables = []
    start_idx = 0

    for end_idx in empty_rows:
        if start_idx != end_idx:
            table = df.iloc[start_idx:end_idx]
            if not table.empty:
                tables.append(table)
        start_idx = end_idx + 1 # Update the starting index to the row after the current empty row
    
    # Add the last table if any rows remain after the last empty row
    if start_idx < len(df):
        table = df.iloc[start_idx:]
        if not table.empty:
            tables.append(table)

    cleaned_tables = []
    for table in tables:
        new_header = table.iloc[0]
        table = table[1:]  
        table.columns = new_header 
        table = table.loc[:, ~(table.isna().all() & table.columns.isna())] # drop columns that are completely NaN and have a NaN column name 
        table = table.dropna(how='all') # drop rows that are completely NaN
        table.reset_index(drop=True, inplace=True)
        if not table.empty:
            cleaned_tables.append(table)
    
    return cleaned_tables

def group_tables_by_columns(tables):
    grouped_tables = {}
    for table in tables:
        columns_signature = tuple(table.columns)
        print("columns signature: ", columns_signature)
        if columns_signature not in grouped_tables:
            grouped_tables[columns_signature] = []
        grouped_tables[columns_signature].append(table)
    return grouped_tables

def merge_and_select_relevant_table(grouped_tables):
    best_table = None
    max_relevant_columns = 0
    other_tables = []

    for group in grouped_tables.values():
        merged_table = pd.concat(group, ignore_index=True)

        relevant_columns = merged_table.columns

        # If the current table has more relevant columns, update the best_table
        if len(relevant_columns) > max_relevant_columns:
            print("best table updated")
            if best_table is not None:
                other_tables.append(best_table)
            best_table = merged_table
            max_relevant_columns = len(relevant_columns)
        else:
            # If the current table is not the best, add it to the list of other tables
            other_tables.append(merged_table)

    # If a table with relevant columns is found, return it along with other tables
    if best_table is not None:
        return best_table, other_tables
    else:
        # If no relevant tables are found, return None for best_table and the list of other tables
        return None, other_tables

def clean_relevant_table(relevant_table):
    temp_table = relevant_table.copy()
    temp_table = temp_table.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    if 'total' in temp_table.iloc[-1].values:
        relevant_table = relevant_table.iloc[:-1] # remove last row if 'total' found
    
    return relevant_table


pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 

def main():
    st.title("Convert Payroll into Import Journals")

    payroll_company = st.selectbox(
    "Select Payroll Company",
    (None, "Payboy", "Talenox", "HReasily", "Other"))
    st.write("You selected:", payroll_company)

    uploaded_file = st.file_uploader("Upload Payroll Excel file", type=['xlsx'])

    if st.sidebar.checkbox("Enable prompt editing"):
        default_prompt = "Please analyze the payroll data."
        custom_prompt = st.sidebar.text_area("Edit GPT prompt:", value=default_prompt)
    else:
        custom_prompt = None

    if uploaded_file and payroll_company:
        start_time = time.time()

        dt = datetime.now(timezone.utc) 
        utc_time = dt.replace(tzinfo=timezone.utc) 
        utc_timestamp = utc_time.timestamp()

        if payroll_company == 'Talenox':
            system_prompt = talenox_prompt
            df = load_and_combine_sheets(uploaded_file)
        elif payroll_company == 'Payboy':
            system_prompt = payboy_prompt
            df = pd.read_excel(uploaded_file, header=None)
        elif payroll_company == 'HReasily':
            system_prompt = hreasily_prompt
            df = pd.read_excel(uploaded_file, header=None)
        elif payroll_company == 'Other':
            system_prompt = other_prompt
            df = pd.read_excel(uploaded_file, header=None)
            
        tables = extract_tables_from_csv(df)
        grouped_tables = group_tables_by_columns(tables)
        relevant_table, other_tables = merge_and_select_relevant_table(grouped_tables)
        relevant_table = clean_relevant_table(relevant_table)
        print("relevant table: ", relevant_table)
        print("relevant table columns: ", relevant_table.columns)
        
        response = call_gpt(relevant_table.columns, system_prompt, custom_prompt) 
        extracted_table = extract_columns(response, relevant_table)
        formatted_table = format_table(extracted_table)
        summary_total_table = calculate_summary_total(formatted_table)
        employee_totals_table = calculate_employee_totals(formatted_table)
        # cost_to_company = calculate_cost_to_company(employees_df)

        st.write(f"Analysis for {uploaded_file.name}:")
        st.info("Employee Level")
        st.table(formatted_table)
        st.table(employee_totals_table)
        st.info("Summary Total")
        st.table(summary_total_table)
        # st.warning(f"Cost to company = {cost_to_company}")
        st.write("utc timestamp :", str(round(utc_timestamp)))
        st.write("time taken: ", str(round((time.time() - start_time), 2)) + " seconds")

if __name__ == "__main__":
    main()
