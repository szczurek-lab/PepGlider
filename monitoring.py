from typing import Optional, Literal, List, Tuple, Union
import numpy as np
import clearml
from  torch import tensor, distributed
import data.dataset as dataset_lib
import data.data_describe as d
from sklearn.metrics import mean_absolute_error
from math import inf

def is_main_process():
    return not distributed.is_available() or not distributed.is_initialized() or distributed.get_rank() == 0

def report_scalars(
    logger: clearml.Logger,
    hue: str,
    epoch: int,
    scalars: List[Union[Tuple[str, float], Tuple[str, str, float]]],
):
    if not is_main_process():
        return
    for tpl in scalars:
        if len(tpl) == 2:
            name, val = tpl
            current_hue = hue
        else:
            name, hue_suffix, val = tpl
            current_hue = f"{hue} - {hue_suffix}"
        logger.report_scalar(title=name, series=current_hue, value=val, iteration=epoch)

def report_sequence_char(
    logger: clearml.Logger,
    hue: str,
    epoch: int,
    seq_true: np.ndarray,
    model_out: np.ndarray,
    metrics: dict,
    physchem_original: np.ndarray,
    pool
):
    if not is_main_process():
       return
    seq_pred = model_out.argmax(axis=2)
    src_pred = dataset_lib.decoded(tensor(seq_pred).permute(1, 0), "")
    filtered_list = [item for item in src_pred if item.strip()]
    if not filtered_list:
        print('All predicted sequences are empty')
    else:
        physchem_decoded_async = d.calculate_physchem(pool, filtered_list)
        physchem_decoded = d.gather_physchem_results(physchem_decoded_async)
    len_true = seq_true.argmin(axis=0)
    len_pred = seq_pred.argmin(axis=0)

    pred_len_acc = (len_true == len_pred).mean()
    pred_len_mae = np.abs(len_true - len_pred).mean()

    correct, overall = 0, 0
    amino_correct, amino_total = 0, 0
    empty_correct, empty_total = 0, 0

    for len_ in range(len_pred.max() + 1):
        idx = len_pred == len_
        true_sub = seq_true[:, idx]
        pred_sub = seq_pred[:, idx]

        # Overall token accuracy (within the predicted length)
        correct += (true_sub[:len_].reshape(-1) == pred_sub[:len_].reshape(-1)).sum()
        overall += len_ * idx.sum()

        # Amino acid accuracy (non-padding tokens)
        amino_mask = true_sub > 0
        amino_correct += (true_sub[amino_mask] == pred_sub[amino_mask]).sum()
        amino_total += amino_mask.sum()

        # Empty (padding) token accuracy
        empty_mask = true_sub == 0
        empty_correct += (true_sub[empty_mask] == pred_sub[empty_mask]).sum()
        empty_total += empty_mask.sum()

    on_predicted_acc = correct / overall if overall > 0 else 0
    amino_acc = amino_correct / amino_total if amino_total > 0 else 0
    empty_acc = empty_correct / empty_total if empty_total > 0 else 0

    if logger:
        logger.report_scalar(
            title="Length Prediction Accuracy",
            series=hue,
            value=pred_len_acc,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Length Loss [mae]", series=hue, value=pred_len_mae, iteration=epoch
        )
        logger.report_scalar(
            title="Token Prediction Accuracy (on predicted length)",
            series=hue,
            value=on_predicted_acc,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Amino Token Accuracy", series=hue, value=amino_acc, iteration=epoch
        )
        logger.report_scalar(
            title="Empty Token Accuracy", series=hue, value=empty_acc, iteration=epoch
        )
        if filtered_list:
            logger.report_scalar(
                title="MAE between original length metric vs. decoded",
                series=hue,
                value=mean_absolute_error(physchem_original, physchem_decoded[1]),
                iteration=epoch
            )
            logger.report_scalar(
                title="MAE between original charge metric vs. decoded",
                series=hue,
                value=mean_absolute_error(physchem_original, physchem_decoded[2]),
                iteration=epoch
            )
            logger.report_scalar(
                title="MAE between original hydrophobicity moment metric vs. decoded",
                series=hue,
                value=mean_absolute_error(physchem_original, physchem_decoded[0]),
                iteration=epoch
            )
        for attr in metrics.keys():
                if attr == 'Interpretability':
                    for subattr in metrics[attr].keys():
                        logger.report_scalar(
                            title=f"Interpretability - {subattr} of latent space", series="ar-vae metrics", value=metrics["Interpretability"][subattr][1], iteration=epoch
                        )
                else:
                    logger.report_scalar(
                        title=f"{attr} of latent space", series="ar-vae metrics", value=metrics[attr], iteration=epoch
                    )
        metrics_list = []
    else:
        metrics_list = [pred_len_acc, pred_len_mae, on_predicted_acc, amino_acc, empty_acc, mean_absolute_error(physchem_original, physchem_decoded[1]), mean_absolute_error(physchem_original, physchem_decoded[2]), mean_absolute_error(physchem_original, physchem_decoded[0])]
        for attr in metrics.keys():
                if attr == 'Interpretability':
                    for subattr in metrics[attr].keys():
                        metrics_list = metrics_list + metrics["Interpretability"][subattr][1]
                else:
                    metrics_list + metrics[attr]
    return metrics_list

def report_sequence_char_test(
    logger: clearml.Logger,
    hue: str,
    epoch: int,
    seq_true: np.ndarray,
    model_out: np.ndarray,
    metrics: dict,
    physchem_original: np.ndarray
):
    if not is_main_process():
        return
    seq_pred = model_out.argmax(axis=2)
    src_pred = dataset_lib.decoded(tensor(seq_pred).permute(1, 0), "")
    filtered_list = [item for item in src_pred if item.strip()]
    indexes = [index for index, item in enumerate(src_pred) if item.strip()]
    if not filtered_list:
        print('All predicted sequences are empty')
    else:
        physchem_decoded = d.calculate_physchem_test(filtered_list)
    len_true = seq_true.argmin(axis=0)
    len_pred = seq_pred.argmin(axis=0)

    pred_len_acc = (len_true == len_pred).mean()
    pred_len_mae = np.abs(len_true - len_pred).mean()

    correct, overall = 0, 0
    amino_correct, amino_total = 0, 0
    empty_correct, empty_total = 0, 0

    for len_ in range(len_pred.max() + 1):
        idx = len_pred == len_
        true_sub = seq_true[:, idx]
        pred_sub = seq_pred[:, idx]

        # Overall token accuracy (within the predicted length)
        correct += (true_sub[:len_].reshape(-1) == pred_sub[:len_].reshape(-1)).sum()
        overall += len_ * idx.sum()

        # Amino acid accuracy (non-padding tokens)
        amino_mask = true_sub > 0
        amino_correct += (true_sub[amino_mask] == pred_sub[amino_mask]).sum()
        amino_total += amino_mask.sum()

        # Empty (padding) token accuracy
        empty_mask = true_sub == 0
        empty_correct += (true_sub[empty_mask] == pred_sub[empty_mask]).sum()
        empty_total += empty_mask.sum()

    on_predicted_acc = correct / overall if overall > 0 else 0
    amino_acc = amino_correct / amino_total if amino_total > 0 else 0
    empty_acc = empty_correct / empty_total if empty_total > 0 else 0

    if logger:
        logger.report_scalar(
            title="Length Prediction Accuracy",
            series=hue,
            value=pred_len_acc,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Length Loss [mae]", series=hue, value=pred_len_mae, iteration=epoch
        )
        logger.report_scalar(
            title="Token Prediction Accuracy (on predicted length)",
            series=hue,
            value=on_predicted_acc,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Amino Token Accuracy", series=hue, value=amino_acc, iteration=epoch
        )
        logger.report_scalar(
            title="Empty Token Accuracy", series=hue, value=empty_acc, iteration=epoch
        )
        if filtered_list:
            logger.report_scalar(
                title="MAE between original length metric vs. decoded",
                series=hue,
                value=mean_absolute_error(physchem_original, physchem_decoded[1]),
                iteration=epoch
            )
            logger.report_scalar(
                title="MAE between original charge metric vs. decoded",
                series=hue,
                value=mean_absolute_error(physchem_original, physchem_decoded[2]),
                iteration=epoch
            )
            logger.report_scalar(
                title="MAE between original hydrophobicity moment metric vs. decoded",
                series=hue,
                value=mean_absolute_error(physchem_original, physchem_decoded[0]),
                iteration=epoch
            )
        for attr in metrics.keys():
                for subattr in metrics[attr].keys():
                        logger.report_scalar(
                            title=f"{attr} - {subattr} of latent space", series="ar-vae metrics", value=metrics[attr][subattr][1], iteration=epoch
                        )
        metrics_list = []
    else:
        if not filtered_list:
            metrics_list = [pred_len_acc, pred_len_mae, on_predicted_acc, amino_acc, empty_acc, inf, inf, inf]
        else:
            metrics_list = [pred_len_acc, pred_len_mae, on_predicted_acc, amino_acc, empty_acc, mean_absolute_error(physchem_original[indexes,1], physchem_decoded.iloc[:,1]), mean_absolute_error(physchem_original[indexes,2], physchem_decoded.iloc[:,2]), mean_absolute_error(physchem_original[indexes,0], physchem_decoded.iloc[:,0])]
        for attr in metrics.keys():
                for subattr in metrics[attr].keys():
                        metrics_list = metrics_list + metrics[attr][subattr][1]
    return metrics_list
