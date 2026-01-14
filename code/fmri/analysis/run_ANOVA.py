import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. LOAD YOUR DETAILED DATA ---
# The ANOVA needs the data from every single subject, not the summary table.
file_path = './id_analysis_results_by_condition/group_id_full_results_by_condition.csv'
results_df = pd.read_csv(file_path)

# Ensure the columns are treated as categories for the analysis
results_df['Condition'] = pd.Categorical(results_df['Condition'])
results_df['ROI'] = pd.Categorical(results_df['ROI'])

print("--- Data Loaded Successfully ---")
print(results_df.head())


# --- 2. RUN THE TWO-WAY REPEATED MEASURES ANOVA ---
print("\n\n--- Running Two-Way Repeated Measures ANOVA ---")
# dv = dependent variable
# within = the two within-subject factors
# subject = the subject identifier column
aov = pg.rm_anova(
    data=results_df,
    dv='Intrinsic_Dim',
    within=['Condition', 'ROI'],
    subject='Subject',
    detailed=True
)

print("\nANOVA Results:")
print(aov)


# --- 3. RUN POST-HOC TESTS TO ANALYZE SIGNIFICANT EFFECTS ---
# This is essential if the ANOVA shows a significant interaction or main effect.
# We run it to see *which specific pairs* of conditions/ROIs are different.
print("\n\n--- Running Post-Hoc Pairwise T-Tests (Bonferroni-corrected) ---")

posthocs = pg.pairwise_tests(
    data=results_df,
    dv='Intrinsic_Dim',
    within=['Condition', 'ROI'],
    subject='Subject',
    padjust='bonf'  # 'bonf' = Bonferroni correction. Other options: 'sidak', 'holm', 'fdr_bh'
)

# Set pandas to display all rows so you can see the full results
pd.set_option('display.max_rows', None)

print("\nPost-Hoc Test Results:")
print(posthocs)

# Reset display option
pd.reset_option('display.max_rows')
