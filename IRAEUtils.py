import pandas as pd
import numpy as np
import os, math, json, spacy
from sklearn.metrics import confusion_matrix


## Returns 1,0 if it first finds a Yes, No/Not token in the response. 
## Otherwise, returns -1
##
def convert_llm_response_to_01(llm_response_yes_no, nlp = spacy.load('en_core_web_sm')) :
    if llm_response_yes_no.startswith("LLMException") :
        return -1
    return tokenize_llm_response(llm_response_yes_no, nlp)

## Perform tokenization to convert Yes/No llm reponses to 0/1/-1
##
def tokenize_llm_response(llm_response_yes_no, nlp) :
    tokenized_response = nlp(llm_response_yes_no)
    for token in tokenized_response :
        if token.text.casefold() == 'Yes'.casefold() :
            return 1
        elif token.text.casefold() == 'No'.casefold() :
            return 0
        elif token.text.casefold() == 'Not'.casefold() :
            return 0
        elif token.text.casefold() == 'unlikely'.casefold() :
            return 0
    return -1

## Reads 1 column (skips first line/header) from a csv format file into a list 
##
def read1col_2list_skip1(filepath, col1, delim):
    return read1col_2list(filepath, col1, delim, True)

## Reads 1 column (all lines) from a csv format file into a list 
##
def read1col_2list_all(filepath, col1, delim):
    return read1col_2list(filepath, col1, delim, False)

## Reads 1 column from a csv format file into a list
##
def read1col_2list(filepath, col1, delim, SKIP_FLAG):
    list = []
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        if SKIP_FLAG :
            trimmed_lines = [line.strip() for line in lines[1:]]
        else :
            trimmed_lines = [line.strip() for line in lines]

        for line in trimmed_lines :
            toks = line.split(delim)
            if col1 >= len(toks) :
                raise AssertionError(f"Colum index [{col1}] is outside range({len(toks)})")
            list.append(toks[col1])
    except FileNotFoundError:
        print(f"adibejan.io.read1col_2list: File not found: {filepath}")
    return list

## Reads 2 columns from a csv format file (skips first line/header) into a dict
##
def read2cols_2dict_skip1(filepath, col1, col2, delim):
    return read2cols_2dict(filepath, col1, col2, delim, True)


## Reads 2 columns from a csv format file (all lines) into a dict
##
def read2cols_2dict_all(filepath, col1, col2, delim):
    return read2cols_2dict(filepath, col1, col2, delim, False)


## Reads 2 columns from a csv format file into a dict
##
def read2cols_2dict(filepath, col1, col2, delim, SKIP_FLAG):
    cols_dict = dict()
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        if SKIP_FLAG :
            trimmed_lines = [line.strip() for line in lines[1:]]
        else :
            trimmed_lines = [line.strip() for line in lines]

        for line in trimmed_lines :
            toks = line.split(delim)
            if col1 >= len(toks) or col2 >= len(toks) :
                raise AssertionError(f"Colum indices (col1:{col1}, col2:{col2}) outside range({len(toks)})")
            cols_dict[toks[col1]] = toks[col2]            
    except FileNotFoundError:
        print(f"adibejan.io.read_col2list: File not found: {filepath}")
    return cols_dict

## Reads 2 columns from a csv format file (all lines) into a list of tuples
##
def read2cols_2list_all(filepath, col1, col2, delim):
    return read2cols_2list(filepath, col1, col2, delim, False)

## Reads 2 columns from a csv format file into a list of tuples
##
def read2cols_2list(filepath, col1, col2, delim, SKIP_FLAG):
    tuple2_list = []
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        if SKIP_FLAG :
            trimmed_lines = [line.strip() for line in lines[1:]]
        else :
            trimmed_lines = [line.strip() for line in lines]

        for line in trimmed_lines :
            toks = line.split(delim)
            if col1 >= len(toks) or col2 >= len(toks) :
                raise AssertionError(f"Colum indices (col1:{col1}, col2:{col2}) outside range({len(toks)})")
            tuple2_list.append((toks[col1], toks[col2]))
    except FileNotFoundError:
        print(f"adibejan.io.read_col2list: File not found: {filepath}")
    return tuple2_list


## Returns all the file names from a specific folder
##
def file_names(folder_path):
    file_names = []
    try:
        # Get the list of files and directories in the folder
        names_list = os.listdir(folder_path)

        for name in names_list:
            if os.path.isfile(os.path.join(folder_path, name)) :
                file_names.append(name)
        
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return file_names

## Reads the content of a file
##
def read(filepath):
    try:
        with open(filepath, 'r') as file:
            data = file.read()
        return data
    
    except FileNotFoundError:
        print(f"adibejan.io.read: File not found: {filepath}")

## Write text into a file
##
def write(filepath, text):
    with open(filepath, 'w') as file:
        file.write(text)

# Check if a string is json fromated
def is_json(input_string):
    try:
        json.loads(input_string)
        return True
    except ValueError:
        return False
    
# Multi-label evaluation
def irae_eval(df_y_gold, df_y_llm, list_irae_label) :
    list_precision = []
    list_recall = []
    list_specificity = []
    list_f1 = []
    list_acc = []

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    df_eval = pd.DataFrame(columns=['irAE', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'Specificity', 'F1', 'Accuracy', 'Support'])

    for index, irae in enumerate(list_irae_label):
        y_gold_vector = np.array(df_y_gold[:, index]).astype(int)
        y_llm_vector = np.array(df_y_llm[:, index]).astype(int)

        cm = confusion_matrix(y_gold_vector, y_llm_vector)
        #print(cm)

        if len(cm) == 1 :
            cmTP = cmFP = cmFN = 0
            cmTN = cm[0][0]
        else :
            cmTP = cm[1,1]
            cmFP = cm[0,1]
            cmFN = cm[1,0]
            cmTN = cm[0,0]

            TP += cmTP
            FP += cmFP
            FN += cmFN
            TN += cmTN

        cm_positives = cmTP+cmFN
        cm_negatives = cmFP+cmTN
        cm_total = cm_positives + cm_negatives

        if cmTP + cmFP == 0 : cm_precision = 0
        else : cm_precision = cmTP / (cmTP + cmFP)                

        if cmTP + cmFN == 0 : cm_recall = 0
        else : cm_recall = cmTP / (cmTP + cmFN)

        if cmTN + cmFP == 0 : cm_specificity = 0
        else: cm_specificity = cmTN / (cmTN + cmFP)

        if cm_precision + cm_recall == 0 : cm_f1 = 0
        else : cm_f1 = 2 * cm_precision * cm_recall / (cm_precision + cm_recall)

        if cm_total == 0 : cm_acc = 0
        else : cm_acc = (cmTP + cmTN) / cm_total

        if math.isnan(cm_precision): cm_precision = 0
        if math.isnan(cm_recall): cm_recall = 0
        if math.isnan(cm_specificity): cm_specificity = 0
        if math.isnan(cm_f1): cm_f1 = 0
        if math.isnan(cm_acc): cm_acc = 0

        list_precision.append(cm_precision)
        list_recall.append(cm_recall)
        list_specificity.append(cm_specificity)
        list_f1.append(cm_f1)
        list_acc.append(cm_acc)
        
        df_row = {'irAE' : irae, 'TP' : cmTP, 'FP' : cmFP, 'FN' : cmFN, 'TN' : cmTN, 'Precision' : cm_precision, 'Recall' : cm_recall, 'Specificity' : cm_specificity, 'F1' : cm_f1, 'Accuracy' : cm_acc, 'Support' :cm_positives}    
        df_eval = pd.concat([df_eval, pd.DataFrame([df_row])], ignore_index=True)    

    # macro/micro averaged results
    Positives = TP + FN
    Negatives = FP + TN
    Total = Positives + Negatives

    if TP + FP == 0 : microPrecision = 0
    else : microPrecision = TP / (TP + FP)

    if TP + FN == 0 : microRecall = 0
    else : microRecall = TP / (TP + FN)

    if TN + FP == 0 : microSpecificity = 0
    else : microSpecificity = TN / (TN + FP)

    if microPrecision + microRecall == 0 : microF1 = 0
    else : microF1 = 2 * microPrecision * microRecall / (microPrecision + microRecall)

    if Total == 0 : microAcc = 0
    else : microAcc = (TP + TN) / Total

    if math.isnan(microPrecision): microPrecision = 0
    if math.isnan(microRecall): microRecall = 0
    if math.isnan(microSpecificity): microSpecificity = 0
    if math.isnan(microF1): microF1 = 0
    if math.isnan(microAcc): microAcc = 0

    # add empty row
    df_row = {'irAE' : '', 'TP' : '', 'FP' : '', 'FN' : '', 'TN' :'', 'Precision' : '', 'Recall' : '', 'Specificity' : '', 'F1' : '', 'Accuracy' : '', 'Support' : ''}
    df_eval = pd.concat([df_eval, pd.DataFrame([df_row])], ignore_index=True)

    ## micro-average
    df_row = {'irAE' : 'micro avg', 'TP': TP, 'FP' : FP, 'FN' : FN, 'TN' : TN, 'Precision' : microPrecision, 'Recall' : microRecall, 'Specificity' : microSpecificity, 'F1' : microF1, 'Accuracy' : microAcc, 'Support' : Positives}
    df_eval = pd.concat([df_eval, pd.DataFrame([df_row])], ignore_index=True)

    ## macro-average
    df_row = {'irAE' : 'macro avg', 'TP': TP, 'FP' : FP, 'FN' : FN, 'TN' : TN, 'Precision' : np.mean(list_precision), 'Recall' : np.mean(list_recall), 'Specificity' : np.mean(list_specificity), 'F1' : np.mean(list_f1), 'Accuracy' : np.mean(list_acc), 'Support' : Positives}
    df_eval = pd.concat([df_eval, pd.DataFrame([df_row])], ignore_index=True)

    return df_eval

#
def df_count_perc(df, col_name):
    # Get counts of each category
    counts = df[col_name].value_counts()

    # Calculate percentages
    percentages = (counts / len(df)) * 100

    # Create a new dataframe to store counts and percentages
    summary_df = pd.DataFrame({'Count': counts, 'Percentage': percentages})

    # Sort the dataframe by counts
    summary_df = summary_df.sort_values(by='Count', ascending=False)

    # Calculate cumulative count
    cumulative_count = summary_df['Count'].cumsum()
    
    # Append row with cumulative count to the dataframe
    summary_df.loc['Total'] = [cumulative_count[-1], '100']

    return summary_df

def df_count_perc2(df, col_name):
    # Calculate counts and percentages of distinct values
    value_counts = df[col_name].value_counts()
    percentages = df[col_name].value_counts(normalize=True) * 100

    # Create a new DataFrame with the counts and percentages
    result_df = pd.DataFrame({'Count': value_counts, 'Percentage': percentages})

    # Sort the DataFrame by counts in descending order
    result_df = result_df.sort_values(by='Count', ascending=False)
    
    # Calculate cumulative sum and add as a new row
    cumulative_sum = result_df['Count'].sum()
    cumulative_row = pd.DataFrame({'Count': cumulative_sum, 'Percentage': [result_df['Percentage'].sum()]}, index=[f'{col_name} total'])

    # Append the cumulative sum row to the DataFrame    
    result_df = pd.concat([result_df, cumulative_row], ignore_index=False)   

    return result_df