import numpy as np
from sklearn.utils import compute_class_weight
import torch.nn as nn
import xgboost as xgb
from xgboost import DMatrix
import torch
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
DATA_DIR = "data/"
HQ_AMPs_FILE = DATA_DIR + "activity-data/curated-AMPs.fasta"
# from project.synthetic_data import generate_synthetic_sequences
from seq_properties import calculate_physchem_prop, calculate_aa_frequency, calculate_positional_encodings
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score
    

def focal_loss(predt: np.ndarray, dtrain: xgb.DMatrix, gamma=2.0):
    """
    Computes the gradient and hessian for Focal Loss in XGBoost.

    The loss function is:
    -alpha * [(1-p)^gamma * y * log(p) + p^gamma * (1-y) * log(1-p)]

    This function computes the derivatives of the loss with respect to the
    raw margin score 'predt', not the probability 'p'.
    """
    if predt.shape[0] == 0:
        print('Empty predt!')
        return np.array([]), np.array([])
    # 1. Get true labels and alpha weights
    y = dtrain.get_label()
    alpha = dtrain.get_weight()

    # 2. Transform raw margin score 'predt' to probabilities 'p'
    p = 1.0 / (1.0 + np.exp(-predt))
    
    # Clip probabilities to avoid log(0) and division by zero
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # 3. Compute the first derivative of the loss with respect to p (your original grad)
    grad_wrt_p = alpha * (
        -(p**gamma * (y - 1))/(1 - p) 
        + gamma * p**(gamma - 1) * (y - 1) * np.log(1 - p) 
        - ((1 - p)**gamma * y)/p 
        + gamma * (1 - p)**(gamma - 1) * y * np.log(p)
    )

    # 4. Compute the second derivative of the loss with respect to p (your original hess)
    hess_wrt_p = - alpha * (
        (p**gamma * (y - 1))/(1 - p)**2 
        + (2 * gamma * p**(gamma - 1) * (y - 1))/(1 - p) 
        - (gamma - 1) * gamma * p**(gamma - 2) * (y - 1) * np.log(1 - p) 
        - ((1 - p)**gamma * y)/p**2 
        - (2 * gamma * (1 - p)**(gamma - 1) * y)/p 
        + (gamma - 1) * gamma * (1 - p)**(gamma - 2) * y * np.log(p)
    )
    
    # 5. Apply the chain rule to get derivatives with respect to 'predt'
    # Derivative of sigmoid: p * (1 - p)
    sigmoid_deriv = p * (1 - p)

    # Gradient with respect to predt
    grad = grad_wrt_p * sigmoid_deriv

    # Hessian with respect to predt (using the approximation for stability)
    # hess ≈ (d²L/dp²) * (dp/dpredt)²
    hess = hess_wrt_p * (sigmoid_deriv**2) # + grad_wrt_p * (p * (1 - p) * (1 - 2*p))

    return grad, hess


class PeptideClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        if model_path is not None:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                         early_stopping_rounds=50, n_estimators=5000)
        self.model.n_classes_ = 2
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.decision_threshold = 0.5

    def get_input_features(self, sequences):
        """To be implemented by child classes"""
        raise NotImplementedError

    def train_classifier(self, input_features, labels, weight_balancing="balanced_with_adjustment_for_high_quality", mask_high_quality_idxs=[], return_feature_importances=False, verbose=True, objective='focal_loss'):
        """To be implemented by child classes"""
        raise NotImplementedError

    def eval_with_k_fold_cross_validation(self, input_features, labels, weight_balancing="balanced_with_adjustment_for_high_quality", k=5, mask_high_quality_idxs=[], reference_file=HQ_AMPs_FILE, objective='focal_loss'):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        accuracies = []
        f1_scores = []
        mcc_scores = []
        high_quality_accuracies = []
        high_quality_f1_scores = []
        high_quality_mcc_scores = []
        confusion_matrices = []
        high_quality_confusion_matrices = []
        # random_hit_rate = []
        # shuffled_hit_rate = []
        # mutated_hit_rate = []
        # added_deleted_hit_rate = []
        precision_at_100, high_quality_precision_at_100 = [], []

        mutations=5 # FIXME hardcoded values
        additions=5 # FIXME hardcoded values
        # random_sequences, shuffled_sequences, mutated_sequences, added_deleted_sequences = generate_synthetic_sequences(reference_file, 10000, mutations, additions) # FIXME hardcoded values
        no_synthetic_sequences_for_precision_computation = 1000 # FIXME hardcoded values

        # random_input_features = self.get_input_features(random_sequences)
        # shuffled_input_features = self.get_input_features(shuffled_sequences)
        # mutated_input_features = self.get_input_features(mutated_sequences)
        # added_deleted_input_features = self.get_input_features(added_deleted_sequences)

        for train_index, test_index in kf.split(input_features):
            train_features = [input_features[i] for i in train_index]
            test_features = [input_features[i] for i in test_index]
            train_labels = [labels[i] for i in train_index]
            test_labels = [labels[i] for i in test_index]
            train_mask_high_quality_idxs = [mask_high_quality_idxs[i] for i in train_index]
            test_mask_high_quality_idxs = [mask_high_quality_idxs[i] for i in test_index]

            self.train_classifier(train_features, train_labels, weight_balancing=weight_balancing, mask_high_quality_idxs=train_mask_high_quality_idxs, verbose=False, objective=objective)

            predictions = self.predict_from_features(test_features)
            scores = self.predict_from_features(test_features, proba=True)
            
            high_quality_test_labels = np.array(test_labels)[test_mask_high_quality_idxs]
            high_quality_predictions = predictions[test_mask_high_quality_idxs]
            high_quality_scores = scores[test_mask_high_quality_idxs]

            # random_predictions = self.predict_from_features(random_input_features)
            # random_scores = self.predict_from_features(random_input_features, proba=True)
            # shuffled_predictions = self.predict_from_features(shuffled_input_features)
            # shuffled_scores = self.predict_from_features(shuffled_input_features, proba=True)
            # mutated_predictions = self.predict_from_features(mutated_input_features)
            # mutated_scores = self.predict_from_features(mutated_input_features, proba=True)
            # added_deleted_predictions = self.predict_from_features(added_deleted_input_features)
            # added_deleted_scores = self.predict_from_features(added_deleted_input_features, proba=True)

            # random_hit_rate.append(random_predictions.mean())
            # shuffled_hit_rate.append(shuffled_predictions.mean())
            # mutated_hit_rate.append(mutated_predictions.mean())
            # added_deleted_hit_rate.append(added_deleted_predictions.mean())

            accuracies.append(accuracy_score(test_labels, predictions))
            f1_scores.append(f1_score(test_labels, predictions))
            mcc_scores.append(matthews_corrcoef(test_labels, predictions))

            high_quality_accuracies.append(accuracy_score(high_quality_test_labels, high_quality_predictions))
            high_quality_f1_scores.append(f1_score(high_quality_test_labels, high_quality_predictions))
            high_quality_mcc_scores.append(matthews_corrcoef(high_quality_test_labels, high_quality_predictions))

            confusion_matrices.append(confusion_matrix(test_labels, predictions, normalize='true'))
            high_quality_confusion_matrices.append(confusion_matrix(high_quality_test_labels, high_quality_predictions, normalize='true'))

            all_predictions = np.concatenate([high_quality_predictions, 
                                            #   random_predictions[:no_synthetic_sequences_for_precision_computation], 
                                            #   shuffled_predictions[:no_synthetic_sequences_for_precision_computation],
                                            #   mutated_predictions[:no_synthetic_sequences_for_precision_computation], 
                                            #   added_deleted_predictions[:no_synthetic_sequences_for_precision_computation]
                                              ])
            all_scores = np.concatenate([high_quality_scores, 
                                        #  random_scores[:no_synthetic_sequences_for_precision_computation],
                                        #  shuffled_scores[:no_synthetic_sequences_for_precision_computation],
                                        #  mutated_scores[:no_synthetic_sequences_for_precision_computation],
                                        #  added_deleted_scores[:no_synthetic_sequences_for_precision_computation]
                                         ])
            all_labels = np.concatenate([high_quality_test_labels, [0]*4*no_synthetic_sequences_for_precision_computation])

            top_100_idxs = np.argsort(all_scores)[-100:]
            precision_at_100.append(precision_score(np.array(all_labels)[top_100_idxs], all_predictions[top_100_idxs]))
            
            if len(high_quality_scores) >= 100:
                top_100_high_quality_idxs = np.argsort(high_quality_scores)[-100:]
                high_quality_precision_at_100.append(precision_score(high_quality_test_labels[top_100_high_quality_idxs], high_quality_predictions[top_100_high_quality_idxs]))

        print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
        print(f"Average MCC Score: {np.mean(mcc_scores):.4f} (+/- {np.std(mcc_scores):.4f})")
        
        print("Average Confusion Matrix:")
        average_confusion_matrix = np.mean(confusion_matrices, axis=0)
        print(average_confusion_matrix)
        print("Positive Likelihood Ratio:")
        print(average_confusion_matrix[1, 1] / (average_confusion_matrix[0, 1] + 1e-10))

        if high_quality_accuracies:
            print(f"Average High Quality Accuracy: {np.mean(high_quality_accuracies):.4f} (+/- {np.std(high_quality_accuracies):.4f})")
        if high_quality_f1_scores:
            print(f"Average High Quality F1 Score: {np.mean(high_quality_f1_scores):.4f} (+/- {np.std(high_quality_f1_scores):.4f})")
        if high_quality_mcc_scores:
            print(f"Average High Quality MCC Score: {np.mean(high_quality_mcc_scores):.4f} (+/- {np.std(high_quality_mcc_scores):.4f})")

        print("Average High Quality Confusion Matrix:")
        average_high_quality_confusion_matrix = np.mean(high_quality_confusion_matrices, axis=0)
        print(average_high_quality_confusion_matrix)
        print("High Quality Positive Likelihood Ratio:")
        print(average_high_quality_confusion_matrix[1, 1] / (average_high_quality_confusion_matrix[0, 1] + 1e-10))
        
        # print(f"Probability of random sequences being AMPs: {np.mean(random_hit_rate):.4f}")
        # print(f"Probability of shuffled sequences being AMPs: {np.mean(shuffled_hit_rate):.4f}")
        # print(f"Probability of mutated sequences (mutations={mutations}) being AMPs: {np.mean(mutated_hit_rate):.4f}")
        # print(f"Probability of added-deleted sequences (added-deleted={additions}) being AMPs: {np.mean(added_deleted_hit_rate):.4f}")

        print(f"Precision at Top 100: {np.mean(precision_at_100):.4f} (+/- {np.std(precision_at_100):.4f})")
        if high_quality_precision_at_100:
            print(f"High-Quality Precision at Top 100: {np.mean(high_quality_precision_at_100):.4f} (+/- {np.std(high_quality_precision_at_100):.4f})")

    def forward(self, sequences):
        input = self.get_input_features(sequences)
        probas = self.model.predict_proba(input)[:, 1]
        return (probas >= self.decision_threshold).astype(int)
    
    def predict_from_features(self, input_features, proba=False):
        probas = self.model.predict_proba(input_features)[:, 1]
        if proba:
            return probas
        return (probas >= self.decision_threshold).astype(int)
    
    def predict_proba(self, sequences):
        input = self.get_input_features(sequences)
        return self.model.predict_proba(input)[:, 1]

    def save(self, path):
        self.model.save_model(path)

class HemolyticClassifier(PeptideClassifier):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.decision_threshold = 0.5

    def get_input_features(self, sequences):
        positional_encodings = pd.DataFrame(calculate_positional_encodings(sequences))
        properties = pd.DataFrame(calculate_physchem_prop(sequences, all_scales=True))
        frequencies = pd.DataFrame(calculate_aa_frequency(sequences))
        return pd.concat([properties, frequencies, positional_encodings], axis=1)

    def train_classifier(self, input_features, labels, weight_balancing="balanced_with_adjustment_for_high_quality", mask_high_quality_idxs=[], return_feature_importances=False, verbose=True, objective='focal_loss'):
        train_input, val_input, train_labels, val_labels, train_mask_high_quality_idxs, _ = train_test_split(#
            input_features, labels, mask_high_quality_idxs , test_size=0.03, random_state=42, stratify=labels#
        )

        high_quality_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_mask_high_quality_idxs), y=train_mask_high_quality_idxs)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
        
        weights = np.array([class_weights[c] + high_quality_weights[int(hq)] for (c,hq) in zip(train_labels, train_mask_high_quality_idxs)])

        dtrain = DMatrix(train_input, label=train_labels, weight=weights)#
        dval = DMatrix(val_input, label=val_labels)

        params = self.model.get_xgb_params()

        obj_param = focal_loss if objective == 'focal_loss' else 'binary:logistic'

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.model.n_estimators,
            evals=[(dval, 'eval')],
            early_stopping_rounds=self.model.early_stopping_rounds,
            obj=obj_param,
            verbose_eval=verbose,
        )
        self.model._Booster = booster

        if return_feature_importances:
            return self.model.feature_importances_

