import streamlit as st
import pandas as pd
import openai
from datetime import timezone
from datetime import datetime 
import json
import time
import numpy as np
import anthropic
from fuzzywuzzy import process

OPENAI_API_KEY = '' 
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY= ''
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY
)

payboy_prompt = f"""
    For each column name from a payroll data file, assign it to the most appropriate category from the list below. 
    If a column does not fit any of these categories, exclude it from the output. 
    Ensure the response adheres to the specified JSON structure without deviation.

    Categories:
        1. employee_name: Full name of the employee.
        2. gross_salary: Amount earned by the employee before tax or other deductions
        3. gross_bonus: On top of gross salary. 'Pay Item: Bonus', 'Pay Item: Fixed Bonus'
        4. gross_deductions: Total of Leave Payments or Deductions. 'Unpaid leaves deductions'. Do not include 'Pay Item: xxx'. 
        5. claims: If no claims column exists, this should be an empty list in the JSON. 'Claim Type: xxx'
        6. employee_contributions_cpf: CPF contributions by the employee.
        7. employee_contributions_other: Columns that indicate SHG (Self-Help Group) amount.
        8. net_payable_employee: Net salary payable to the employee.
        9. employer_contributions_cpf: CPF contributions by the employer.
        10. employer_contributions_other: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy).

    For each category, if no column name matches, return empty list. 

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "gross_bonus": ["column_name"],
        "gross_deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "employee_contributions_cpf": ["column_name"],
        "employee_contributions_other": ["column_name", ...],
        "net_payable_employee": ["column_name"],
        "employer_contributions_cpf": ["column_name"],
        "employer_contributions_other": ["column_name", ...]
    }}
    ```
    
    Provide the column names in a list format and adhere to the structure above to map each column to its appropriate category. Only return valid json.
    """

talenox_prompt = f"""
    For each column name from a payroll data file, assign it to the most appropriate category from the list below. 
    If a column does not fit any of these categories, exclude it from the output. 
    Ensure the response adheres to the specified JSON structure without deviation.

    Categories:
        1. employee_name: Full name of the employee.
        2. gross_salary: Use 'total recurring prorated' column instead of 'gross salary'.
        3. gross_bonus: Use 'additional wage' instead of 'total bonus'.
        4. gross_deductions: return empty list. 
        5. claims: If no claims column exists, this should be an empty list in the JSON.
        6. employee_contributions_cpf: CPF contributions by the employee, excluding year-to-date figures and total amount columns. 
        7. employee_contributions_other: Columns that indicate SHG (Self-Help Group) amount. Not the type. 
        8. net_payable_employee: Net salary payable to the employee.
        9. employer_contributions_cpf: CPF contributions by the employer, exclude year-to-date and total amount columns.
        10. employer_contributions_other: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy).

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "gross_bonus": ["column_name"],
        "gross_deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "employee_contributions_cpf": ["column_name"],
        "employee_contributions_other": ["column_name", ...],
        "net_payable_employee": ["column_name"],
        "employer_contributions_cpf": ["column_name"],
        "employer_contributions_other": ["column_name", ...]
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
        3. gross_bonus: On top of gross salary. 
        4. gross_deductions: tax contribution. Do not include Total Deductions. 
        5. claims: If no claims column exists, this should be an empty list in the JSON. 
        6. employee_contributions_cpf: CPF contributions by the employee.
        7. employee_contributions_other: Columns that indicate SHG (Self-Help Group) amount. SSO employee contribution.
        8. net_payable_employee: Net salary payable to the employee.
        9. employer_contributions_cpf: CPF contributions by the employer.
        10. employer_contributions_other: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy). SSO employer contribution.

    For each category, if no column name matches, return empty list. 

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "gross_bonus": ["column_name"],
        "gross_deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "employee_contributions_cpf": ["column_name"],
        "employee_contributions_other": ["column_name", ...],
        "net_payable_employee": ["column_name"],
        "employer_contributions_cpf": ["column_name"],
        "employer_contributions_other": ["column_name", ...]
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
        3. gross_bonus: On top of gross salary. 
        4. gross_deductions: Total of Leave Payments or Deductions. 
        5. claims: If no claims column exists, this should be an empty list in the JSON. 
        6. employee_contributions_cpf: CPF contributions by the employee.
        7. employee_contributions_other: Columns that indicate SHG (Self-Help Group) amount.
        8. net_payable_employee: Net salary payable to the employee.
        9. employer_contributions_cpf: CPF contributions by the employer.
        10. employer_contributions_other: Includes SDL (Skills Development Levy), FWL (Foreign Worker Levy).

    For each category, if no column name matches, return empty list. 

    JSON Structure:
    ```
    {{
        "employee_name": ["column_name", ...], 
        "gross_salary": ["column_name"],
        "gross_bonus": ["column_name"],
        "gross_deductions": ["column_name"],
        "claims": ["column_name", ...], 
        "employee_contributions_cpf": ["column_name"],
        "employee_contributions_other": ["column_name", ...],
        "net_payable_employee": ["column_name"],
        "employer_contributions_cpf": ["column_name"],
        "employer_contributions_other": ["column_name", ...]
    }}
    ```
    
    Provide the column names in a list format and adhere to the structure above to map each column to its appropriate category.
    """

def call_gpt(column_names, system_prompt):
    try:
        full_prompt = f"""
            Here are the column names from the payroll file to map. 
            {column_names}
            {system_prompt}
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
    
def call_anthropic(column_names, system_prompt):
    try:
        full_prompt = f"""
            Here are the column names from the payroll file to map. 
            {column_names}
            {system_prompt}
        """
        response = client.messages.create(
            model = "claude-3-haiku-20240307",
            system = "You are a skilled accountant with comprehensive knowledge in payroll accounting. \
                    You will be analyzing the list of column names from the payroll file and map each to the appropriate category as per the instructions.",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        print(response)
        st.write(response.usage)
        content = response.content[0].text
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

def summarize_table(table):
    numeric_df = table.select_dtypes(include=[np.number])
    summary_series = numeric_df.sum() # sum each column
    summary_df = pd.DataFrame(summary_series).transpose()
    print("summarized table: ", summary_df)
    return summary_df

def calculate_employee_totals(table):
    employee_info = table.loc[:, 'employee_name']
    employee_name = employee_info.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    result = {'employee_name': employee_name}
    categories = [category for category in table.columns.get_level_values(0).unique() if category != 'employee_name']

    for category in categories:
        if category == 'claims':
            # Process each claim as a separate column, no summing
            for claim in table[category].columns:
                result[f"{category}_{claim}"] = table[category][claim]
        else:
            group = table.loc[:, category]
            group = group.apply(pd.to_numeric, errors='coerce')
            total_by_employee = group.sum(axis=1, min_count=1)  
            result[category] = total_by_employee

    # result_df = pd.DataFrame(result, columns=['employee_name'] + categories)
    result_df = pd.DataFrame(result)
    print("employee totals table: ", result_df)
    return result_df

def calculate_variables(table):
    table = table.copy()

    if 'gross_bonus' not in table:
        table['gross_bonus'] = 0
    if 'gross_deductions' not in table:
        table['gross_deductions'] = 0
    if 'employee_contributions_cpf' not in table:
        table['employee_contributions_cpf'] = 0
    if 'employee_contributions_other' not in table:
        table['employee_contributions_other'] = 0
    if 'employer_contributions_cpf' not in table:
        table['employer_contributions_cpf'] = 0
    if 'employer_contributions_other' not in table:
        table['employer_contributions_other'] = 0
    
    # Dynamically sum all claims columns
    claim_columns = [col for col in table.columns if col.startswith('claims_')]
    if claim_columns:
        table['total_claims'] = table[claim_columns].sum(axis=1)
    else:
        table['total_claims'] = 0

    # Calculate new columns
    table['salary_excluding_contributions'] = table['gross_salary'] - table['employee_contributions_cpf'] - table['employee_contributions_other'] - table['gross_deductions']
    table['take_home_earnings'] = table['salary_excluding_contributions'] + table['gross_bonus'] + table['total_claims']
    table['total_contributions'] = table['employee_contributions_cpf'] + table['employee_contributions_other'] + table['employer_contributions_cpf'] + table['employer_contributions_other']

    print("calculated table: ", table)
    return table

def filter_directors_employees(director_names, table):
    table = table.copy()
    if director_names.strip():
        directors_list = [name.strip() for name in director_names.split(',')]
        matches = {name: process.extractOne(name, table['employee_name']) for name in directors_list}
        print("matches: ", matches)
        table['is_director'] = table['employee_name'].apply(
            lambda x: any(x == match[0] and match[1] >= 80 for match in matches.values())
        )
    else:
        table['is_director'] = False

    # Split the table into directors and employees
    directors_table = table[table['is_director']].drop(columns='is_director')
    employees_table = table[~table['is_director']].drop(columns='is_director')

    print("directors table: ", directors_table)
    print("employees table: ", employees_table)
    return directors_table, employees_table

def compile_summary(table_summary, directors_table_summary, employees_table_summary):
    result = pd.DataFrame()

    result['employees_salary_excluding_contributions'] = employees_table_summary['salary_excluding_contributions']
    result['employees_gross_bonus'] = employees_table_summary['gross_bonus']
    result['employees_employee_contributions_cpf'] = employees_table_summary['employee_contributions_cpf']
    result['employees_employee_contributions_other'] = employees_table_summary['employee_contributions_other']
    result['employees_employer_contributions_cpf'] = employees_table_summary['employer_contributions_cpf']
    result['employees_employer_contributions_other'] = employees_table_summary['employer_contributions_other']

    result['directors_salary_excluding_contributions'] = directors_table_summary['salary_excluding_contributions']
    result['directors_gross_bonus'] = directors_table_summary['gross_bonus']
    result['directors_employee_contributions_cpf'] = directors_table_summary['employee_contributions_cpf']
    result['directors_employee_contributions_other'] = directors_table_summary['employee_contributions_other']
    result['directors_employer_contributions_cpf'] = directors_table_summary['employer_contributions_cpf']
    result['directors_employer_contributions_other'] = directors_table_summary['employer_contributions_other']

    # extracting claims dynamically
    claim_columns = [col for col in table_summary.columns if col.startswith('claims_')]
    for claim in claim_columns:
        result[claim] = table_summary[claim]

    result['total_take_home_earnings'] = table_summary['take_home_earnings']
    result['total_contributions'] = table_summary['total_contributions']

    print("final table: ", result)
    return result

def convert_to_import_journals(table):
    table = table.copy()
    table = table.loc[:, (table != 0).any(axis=0)] # drop 0 columns

    entries = []

    # debits: employees
    if 'employees_salary_excluding_contributions' in table.columns:
        entries.append(['Salary (excluding contributions)', table['employees_salary_excluding_contributions'].iloc[0], 0])
    if 'employees_gross_bonus' in table.columns:
        entries.append(['Bonus (excluding contributions)', table['employees_gross_bonus'].iloc[0], 0])
    if 'employees_employee_contributions_cpf' in table.columns:
        entries.append(['Employee CPF Contribution', table['employees_employee_contributions_cpf'].iloc[0], 0])
    if 'employees_employee_contributions_other' in table.columns:
        entries.append(['Employee Other Contribution', table['employees_employee_contributions_other'].iloc[0], 0])
    if 'employees_employer_contributions_cpf' in table.columns:
        entries.append(['Employer CPF Contribution', table['employees_employer_contributions_cpf'].iloc[0], 0])
    if 'employees_employer_contributions_other' in table.columns:
        entries.append(['Employer Other Contribution', table['employees_employer_contributions_other'].iloc[0], 0])

    # debits: directors 
    if 'directors_salary_excluding_contributions' in table.columns:
        entries.append(['Directors: Salary (excluding contributions)', table['directors_salary_excluding_contributions'].iloc[0], 0])
    if 'directors_gross_bonus' in table.columns:
        entries.append(['Directors: Bonus (excluding contributions)', table['directors_gross_bonus'].iloc[0], 0])
    if 'directors_employee_contributions_cpf' in table.columns:
        entries.append(['Directors: Employee CPF Contribution', table['directors_employee_contributions_cpf'].iloc[0], 0])
    if 'directors_employee_contributions_other' in table.columns:
        entries.append(['Directors: Employee Other Contribution', table['directors_employee_contributions_other'].iloc[0], 0])
    if 'directors_employer_contributions_cpf' in table.columns:
        entries.append(['Directors: Employer CPF Contribution', table['directors_employer_contributions_cpf'].iloc[0], 0])
    if 'directors_employer_contributions_other' in table.columns:
        entries.append(['Directors: Employer Other Contribution', table['directors_employer_contributions_other'].iloc[0], 0])

    # debits: claims 
    claim_columns = [col for col in table.columns if col.startswith('claims_')]
    for claim in claim_columns:
        claim_description = claim.replace('claims_', '') 
        entries.append([f"Claims: {claim_description}", table[claim].iloc[0], 0])

    # credits
    if 'total_take_home_earnings' in table.columns:
        entries.append(['Total Take-Home Earnings', 0, table['total_take_home_earnings'].iloc[0]])
    if 'total_contributions' in table.columns:
        entries.append(['Total Contributions', 0, table['total_contributions'].iloc[0]])

    result_df = pd.DataFrame(entries, columns=['Description', 'Debits (SGD)', 'Credits (SGD)'])
    print("import journals output: ", result_df)
    return result_df


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

    director_names = st.text_input("Name of directors (Separate names with a comma)", placeholder="Enter name of directors")
    st.write("Directors are:", director_names)

    uploaded_file = st.file_uploader("Upload Payroll Excel file", type=['xlsx'])

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
        
        # response = call_gpt(relevant_table.columns, system_prompt) 
        response = call_anthropic(relevant_table.columns, system_prompt)
        extracted_table = extract_columns(response, relevant_table)
        formatted_table = format_table(extracted_table)
        employee_totals_table = calculate_employee_totals(formatted_table)
        summary_total_table = calculate_summary_total(formatted_table)

        calculated_employee_totals_table = calculate_variables(employee_totals_table)
        table_summary = summarize_table(calculated_employee_totals_table)

        directors_table, employees_table = filter_directors_employees(director_names, calculated_employee_totals_table)
        directors_table_summary = summarize_table(directors_table)
        employees_table_summary = summarize_table(employees_table)

        final_summary = compile_summary(table_summary, directors_table_summary, employees_table_summary)
        import_journals_output = convert_to_import_journals(final_summary)
        import_journals_output_summary = summarize_table(import_journals_output)

        output_df = pd.DataFrame({
            'Journal Reference': None,
            'Contact': None,
            'Date': None,
            'Account': None,
            'Description': import_journals_output['Description'],
            'Tax Included in Amount': None,
            'Debit Amount (SGD)': import_journals_output['Debits (SGD)'],
            'Credit Amount (SGD)': import_journals_output['Credits (SGD)']
        })
        
        st.write(f"Analysis for {uploaded_file.name}:")
        st.info("Employee Level")
        st.table(formatted_table)
        st.table(employee_totals_table)
        st.table(calculated_employee_totals_table)
        st.table(table_summary)
        st.info("Directors & Employees")
        st.table(directors_table)
        st.table(directors_table_summary)
        st.table(employees_table)
        st.table(employees_table_summary)
        st.info("Summary Total")
        st.table(summary_total_table)

        st.warning("Final Output")
        st.table(final_summary)
        st.table(import_journals_output)
        st.table(import_journals_output_summary)

        st.success("Import Journals Template")
        st.dataframe(output_df)
        csv = output_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "payroll_import_journals.csv", "text/csv", key='download-csv')

        st.write("utc timestamp :", str(round(utc_timestamp)))
        st.write("time taken: ", str(round((time.time() - start_time), 2)) + " seconds")

if __name__ == "__main__":
    main()
