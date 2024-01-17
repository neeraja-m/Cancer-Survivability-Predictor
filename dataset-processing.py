import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import mean_squared_error
import category_encoders as cat_encoder
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn import preprocessing as prp
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest,f_classif
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr

directory_path = '/Users/neerajamenon/Desktop/uni/ip/model/datasets/all-unrelated/all.csv'
dfs = []

# Get dataset information for each individual cancer dataset

for folder in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder)
    if os.path.isdir(folder_path) and 'all.csv' in os.listdir(folder_path):
        file_path = os.path.join(folder_path, 'all.csv')
        df = pd.read_csv(file_path)
        print(file_path, df.info())
        print('======================')

concatenated_df = pd.concat(dfs)
concatenated_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/concatenated.csv', index=False)



# Combine all individual cancer datasets to final concatenated dataset
for folder in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder)
    if os.path.isdir(folder_path) and 'all.csv' in os.listdir(folder_path):
        file_path = os.path.join(folder_path, 'all.csv')
        df = pd.read_csv(file_path)
        dfs.append(df)

concatenated_df = pd.concat(dfs, sort=False)

concatenated_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/all-unrelated/concatenated.csv', index=False)


# Get combined datasets within each cancer
concatenated_df = pd.DataFrame()

for filename in os.listdir(directory_path):
    if filename.endswith("all.csv"):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        column_name = "OS_MONTHS"
        df = df.replace(['', None], np.nan)
        df.dropna(subset=[column_name], inplace=True)
        concatenated_df = pd.concat([concatenated_df, df], axis=0, join="outer")
        df.fillna(value=0, inplace=True)
        df.to_csv(file_path, index=False)

new_path = os.path.join(directory_path, "all.csv")

concatenated_df.to_csv(new_path, index=False)


###########################################################
# Dataset preprocessing


# import datasets 
tsv_file1='/Users/neerajamenon/Desktop/uni/ip/model/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt'
csv_table1=pd.read_table(tsv_file1,sep='\t')
csv_table1.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/data/mrna.csv',index=False)

tsv_file2='/Users/neerajamenon/Desktop/uni/ip/model/data_clinical_patient.txt'
csv_table2=pd.read_table(tsv_file2,sep='\t')
csv_table2.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/data/clinical.csv',index=False)

#### cleaning for mutation file #####

mutations_data = pd.read_csv('/Users/neerajamenon/Desktop/uni/ip/model/data/mrna.csv')
mutations_df = pd.DataFrame(mutations_data)

mutations_df = mutations_df.drop(['Entrez_Gene_Id'],axis=1)

unique_cols = ['Hugo_Symbol']

mutations_df.sort_values(by=unique_cols, inplace=True)

mutations_df.drop_duplicates(subset=unique_cols, keep='first', inplace=True)

mutations_df = mutations_df.set_index('Hugo_Symbol').stack().reset_index()
mutations_df.columns = ['Hugo_Symbol', 'PATIENT_ID', 'gene_expression']

mutations_df = mutations_df.pivot(index='PATIENT_ID', columns='Hugo_Symbol', values='gene_expression')

mutations_df['PATIENT_ID'] = mutations_df['PATIENT_ID'].str[:-3]

mutations_df.to_csv("/Users/neerajamenon/Desktop/uni/ip/model/data/mrna.csv",index=True)


def keep_columns(df, column_names):
    return df.loc[:, column_names]

clinical_data = pd.read_csv("/Users/neerajamenon/Desktop/uni/ip/model/data/clinical.csv")
clinical_df = pd.DataFrame(clinical_data)

clinical_df = keep_columns(clinical_df,['PATIENT_ID','OS_STATUS','OS_MONTHS'])

dss_status_codes = {'0:LIVING':0, '1:DECEASED':1}
clinical_df['OS_STATUS'] = clinical_df['OS_STATUS'].map(dss_status_codes)

clinical_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/data/clinical.csv', index=False) 


# Merge both datasets and make patient ID index
x_data_all = pd.merge(mutations_df, clinical_df, on='PATIENT_ID',how='inner')
clinical_df['PATIENT_ID'] = clinical_df.index
x_data_all.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/data/mrna_all.csv', index=False)

####################################################
# Prelimanary data anlaysis

def remove_na_columns(df):
    threshold = 2 * len(df) / 3 
    return df.dropna(thresh=threshold, axis=1) 

def remove_breast_data(df, column_name, value):
    to_move = df.loc[df[column_name] == value] 
    return pd.DataFrame(to_move)  

def make_new(df, column_name, value):
    new_df = df.drop(df[df[column_name] == value].index) 
    return pd.DataFrame(new_df)

def make_new2(df, column_name, cancers):
    return df[df[column_name].isin(cancers)]

def calculate_vaf(df):
    vaf = df['t_alt_count'] / (df['t_ref_count'] + df['t_alt_count'])
    df['Var_Allele_Freq'] = vaf
    return df

# # Calculate VAF for each mutation
# mutations_df = round(calculate_vaf(mutations_df),3)
# # Keep relevant columns
# mutations_df = keep_columns(mutations_df,['Hugo_Symbol','Tumor_Sample_Barcode','Var_Allele_Freq'])
# # Get patient ID
# mutations_df['Tumor_Sample_Barcode'] = mutations_df['Tumor_Sample_Barcode'].str[:-8]

# Return the new DataFrame
# mutations_df = mutations_df.iloc[:, list(range(17)) + list(range(-4, 0))]
# mutations_df = mutations_df.drop(['Annotation_Status','Entrez_Gene_Id','NCBI_Build','Center','dbSNP_Val_Status','dbSNP_RS'],axis=1) 
# mutations_df = keep_columns(mutations_df,['Hugo_Symbol','Tumor_Sample_Barcode'])
# mutations_df['Tumor_Sample_Barcode'] = mutations_df['Tumor_Sample_Barcode'].str[:-8]
# mutations_df = mutations_df.rename(columns={'Tumor_Sample_Barcode': 'Patient ID'})
# mutations_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/pan_mutations_test.csv', index=False)

# For mutations_clean_2
# mutations_df = mutations_df.loc[:, ['Hugo_Symbol', 'PATIENT_ID']]
# mutations_df = pd.get_dummies(mutations_df,
#                      columns = ['Hugo_Symbol'])
# mutations_df = mutations_df.groupby('Patient ID').sum().reset_index()
# mutations_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/pan-cancer_mutations.csv', index=False)

# ##### cleaning for clinical file #####

# mutations_df =keep_columns(mutations_df,['Entrez_Gene_Id'])
# Remove
# clinical_df = pd.remove_na_rows(clinical_df)
# clinical_df = remove_na_columns(clinical_df)

# bc_df = remove_breast_data(clinical_df, 'Cancer Type', 'Breast Cancer') 
# bc_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/breast-cancer.csv', index=False) 

# nonbc_all_df = make_new(clinical_df, 'Cancer Type', 'Breast Cancer')
# nonbc_all_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/pancancer.csv', index=False) 

# nonbc_df = make_new2(clinical_df,'Cancer Type', ['Endometrial Cancer','Ovarian Cancer', 'Non-Small Cell Lung Cancer','Uterine Sarcoma','Pancreatic Cancer','Thyroid Cancer','Prostate Cancer'])
# clinical_df.to_csv('/Users/neerajamenon/Desktop/uni/ip/model/datasets/pancancer2.csv', index=False) 

# clinical_df = clinical_df.drop(['ICD_10','ICD_O_3_SITE','DAYS_TO_BIRTH','INFORMED_CONSENT_VERIFIED','OTHER_PATIENT_ID','SUBTYPE','FORM_COMPLETION_DATE','ETHNICITY','RACE','DFS_STATUS','DFS_MONTHS','DSS_STATUS'],axis=1) 
# clinical_df['ICD_O_3_HISTOLOGY'] = clinical_df['ICD_O_3_HISTOLOGY'].str[:-2]

# # #convert datatypes
# os_status_codes = {'0:LIVING':0, '1:DECEASED':1}
# clinical_df['Overall Survival Status'] = clinical_df['Overall Survival Status'].map(os_status_codes)

# pfs_status_codes = {'0:CENSORED':0, '1:PROGRESSION':1}
# clinical_df['PFS_STATUS'] = clinical_df['PFS_STATUS'].map(pfs_status_codes)
# tumor_status_codes = {'Tumor Free':0, 'With Tumour':1}
# clinical_df['PERSON_NEOPLASM_CANCER_STATUS'] = clinical_df['PERSON_NEOPLASM_CANCER_STATUS'].map(tumor_status_codes)
# binary_status_codes = {'No':0, 'Yes':1}
# clinical_df['RADIATION_THERAPY'] = clinical_df['RADIATION_THERAPY'].map(binary_status_codes)
# clinical_df['HISTORY_NEOADJUVANT_TRTYN'] = clinical_df['HISTORY_NEOADJUVANT_TRTYN'].map(binary_status_codes)
# clinical_df['NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT'] = clinical_df['NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT'].map(binary_status_codes)
# clinical_df['IN_PANCANPATHWAYS_FREEZE'] = clinical_df['IN_PANCANPATHWAYS_FREEZE'].map(binary_status_codes)
# clinical_df['PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT'] = clinical_df['PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT'].map(binary_status_codes)
# clinical_df['PRIOR_DX'] = clinical_df['PRIOR_DX'].map(binary_status_codes)
# sex_codes = {'Male':0, 'Female':1}
# clinical_df['SEX'] = clinical_df['SEX'].map(sex_codes)

# # # Encode discrete values
# encoder = LabelEncoder()
# encoded_cols = ['CANCER_TYPE_ACRONYM', 'AJCC_PATHOLOGIC_TUMOR_STAGE','AJCC_STAGING_EDITION','PATH_M_STAGE','PATH_N_STAGE','PATH_T_STAGE']
# clinical_df[encoded_cols] = clinical_df[encoded_cols].apply(lambda x: encoder.fit_transform(x))
# clinical_df = pd.concat([clinical_df[['PATIENT_ID', 'AGE','SEX','DAYS_LAST_FOLLOWUP','DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS','PERSON_NEOPLASM_CANCER_STATUS','HISTORY_NEOADJUVANT_TRTYN','ICD_O_3_HISTOLOGY','NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT','PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT','PRIOR_DX','RADIATION_THERAPY','WEIGHT','IN_PANCANPATHWAYS_FREEZE','OS_STATUS','OS_MONTHS','DSS_MONTHS','PFS_STATUS','PFS_MONTHS']], clinical_df[encoded_cols]], axis=1)

# clinical_df = clinical_df.filter(['Patient ID', 'Overall Survival Status'])
# all_data = pd.read_csv('/Users/neerajamenon/Desktop/uni/ip/mrna_all.csv')
# all_df = pd.DataFrame(all_data)
# print(all_df.info())


# merged_df = pd.merge(clinical_df, mutations_df, on="patient_id",how='inner')
# pivoted_df = merged_df.pivot_table(index="Patient ID", columns="Hugo_Symbol", values="Var_Allele_Freq", aggfunc='sum')
# final_df = pd.merge(pivoted_df, clinical_df[["Patient ID", 'Overall Survival (Months)','Overall Survival Status']], on="Patient ID")
# final_df.to_csv("final_dataset.csv", index=False)

# # clinical_df = clinical_df.set_index('at').stack().reset_index()
# # clinical_df.columns = ['at', 'patient_id', 'gene_expression']

# # df_pivoted = clinical_df.pivot(index='patient_id', columns='at', values='gene_expression')

# # print(df_pivoted)
# all_df = all_df.drop(all_df.filter(regex="Unname"),axis=1, inplace=True)