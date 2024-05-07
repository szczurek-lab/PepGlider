import torch

# def get_generation_acc():
#     def metric_(y_true, y_pred):
#         exceeds_zero_threshold = torch.zeros_like(y_pred, dtype=torch.bool)
#         exceeds_zero_threshold.scatter_(2, output.unsqueeze(2), True)
#         exceeds_zero_threshold = exceeds_zero_threshold.float()
#         exceeds_threshold_flag = torch.cumsum(exceeds_zero_threshold, dim=1) > 0
#         amino_tp = (torch.argmax(y_pred[:, :, 1:], dim=-1) + 1).float() == y_true.float()
#         empty_tp = (y_true == 0).float()
#         amino_tp = torch.sum(torch.where(
#             exceeds_threshold_flag,
#             torch.zeros_like(amino_tp, dtype=torch.float),
#             amino_tp,
#         ), dim=-1)
#         empty_tp = torch.sum(torch.where(
#             exceeds_threshold_flag,
#             empty_tp,
#             torch.zeros_like(empty_tp, dtype=torch.float),
#         ), dim=-1)
#         empty_entries_sum = torch.sum(
#             exceeds_threshold_flag.float(), dim=-1)
#         non_empty_entries_sum = torch.sum(1 - exceeds_threshold_flag.float(), dim=-1)
#         amino_acc = torch.where(
#             non_empty_entries_sum > 0,
#             amino_tp / non_empty_entries_sum,
#             torch.zeros_like(amino_tp, dtype=torch.float),
#         )
#         empty_acc = torch.where(
#             empty_entries_sum > 0,
#             empty_tp / empty_entries_sum,
#             torch.ones_like(empty_tp, dtype=torch.float)
#         )
#         return amino_acc, empty_acc
#     return metric_

def compare_tensors(tensor1, tensor2):
    comparison1 = torch.eq(tensor1, tensor2)
    comparison1 = torch.logical_and(comparison1, tensor1 != 0)  
    result1 = torch.all(comparison1).item()  
    
    comparison2 = torch.eq(tensor1, tensor2)
    comparison2 = torch.logical_and(comparison2, tensor1 == 0)  
    result2 = torch.all(comparison2).item()  
    
    result3 = result1 / (tensor2 != 0).sum().item()
    
    result4 = result2 / (tensor2 != 0).sum().item()
    
    return (1-result3), (1-result4)

def kl_loss(z_mean, z_sigma):
    return -0.5 * torch.sum(
        1 + z_sigma - torch.square(z_mean) - torch.exp(z_sigma),
        axis=-1
    )
