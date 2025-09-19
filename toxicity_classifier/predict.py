from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score


hemolytic_classifier = c.HemolyticClassifier('hemolytic_model.xgb')
features = hemolytic_classifier.get_input_features(final_df['sequence'].to_numpy())
labels = final_df['nontoxicity'].to_numpy()
mask_high_quality_idxs = final_df['weight'].to_numpy()
print(f'features shape = {features.shape}, labels shape = {labels.shape}, mask_high_quality_idxs shape = {mask_high_quality_idxs.shape}')
train_input, eval_input, train_labels, eval_labels, train_mask_high_quality_idxs, eval_mask_high_quality_idxs = train_test_split(
    features, labels, mask_high_quality_idxs , test_size=0.03, random_state=42, stratify=labels
)

