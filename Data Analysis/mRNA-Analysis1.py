import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Inspect the Data ---
# Ensure 'os_target_gdc_clinical_data.tsv' is uploaded to device
file_path = 'os_target_gdc_clinical_data.tsv'
df = pd.read_csv(file_path, sep='\t')

print("--- DataFrame Information ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

# --- 2. Descriptive Analysis ---
total_patients = len(df)
mean_age = df['Diagnosis Age'].mean()
vital_status_counts = df["Patient's Vital Status"].value_counts(dropna=False)
sex_counts = df['Sex'].value_counts()
survival_status_counts = df['Overall Survival Status'].value_counts(dropna=False)

print(f"\n--- Descriptive Statistics ---")
print(f"Total Patients: {total_patients}")
print(f"Mean Diagnosis Age: {mean_age:.2f}")
print("\nVital Status Counts:")
print(vital_status_counts)
print("\nSex Distribution:")
print(sex_counts)
print("\nSurvival Status Distribution:")
print(survival_status_counts)

# --- 3. Missing Data Analysis ---
print("\n--- Missing Data Summary (Top 10 Columns) ---")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# --- 4. Visualization ---
# Setup Seaborn styling
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot A: Age Distribution
sns.histplot(df['Diagnosis Age'].dropna(), kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribution of Diagnosis Age')
axes[0, 0].set_xlabel('Age')

# Plot B: Vital Status Distribution
sns.countplot(x="Patient's Vital Status", data=df, ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title("Distribution of Patient's Vital Status")

# Plot C: Sex Distribution
sns.countplot(x='Sex', data=df, ax=axes[1, 0], palette='muted')
axes[1, 0].set_title('Distribution by Sex')

# Plot D: Survival Status
sns.countplot(x='Overall Survival Status', data=df, ax=axes[1, 1], palette='deep')
axes[1, 1].set_title('Overall Survival Status Distribution')
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
# This saves the figure to your Colab files area
plt.savefig('clinical_data_analysis_summary.png')
print("\nVisualization saved as 'clinical_data_analysis_summary.png'")
