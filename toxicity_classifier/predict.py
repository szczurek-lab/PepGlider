from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score
import classifier as c
import pandas as pd

with open('./toxicity_classifier/hemolytic.fasta', 'r', encoding='utf-8') as f:
    hemolytic_content = f.read()
with open('./toxicity_classifier/nonhemolytic.fasta', 'r', encoding='utf-8') as f:
    nonhemolytic_content = f.read()

# hemolytic_df = c.parse_fasta_to_df(hemolytic_content, 1, 0.8)
# nonhemolytic_df = c.parse_fasta_to_df(nonhemolytic_content, 0, 0.2)

# final_df = pd.concat([hemolytic_df, nonhemolytic_df], ignore_index=True)
final_df = pd.read_csv('./toxicity_classifier/hydramp.csv')
print(final_df)
hemolytic_classifier = c.HemolyticClassifier('hemolytic_model.xgb')
features = hemolytic_classifier.get_input_features(final_df['sequence'].to_numpy())
labels = final_df['nontoxicity'].to_numpy()
mask_high_quality_idxs = final_df['weight'].to_numpy()
hemolytic_classifier.eval_trained_classifier(features,labels, mask_high_quality_idxs=mask_high_quality_idxs)

