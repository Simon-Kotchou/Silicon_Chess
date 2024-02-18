import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class PolicyDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        # Remove labels if present and compute student model outputs
        inputs.pop("labels", None)
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits

        # Compute teacher model outputs with no gradient
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        # Apply temperature scaling to teacher and student logits
        logits_teacher_scaled = logits_teacher / self.temperature
        logits_student_scaled = logits_student / self.temperature

        # Compute KL divergence loss between softened teacher and student distributions
        loss = F.kl_div(
            F.log_softmax(logits_student_scaled, dim=-1),
            F.softmax(logits_teacher_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by temperature squared as per Hinton et al.

        return (loss, outputs_student) if return_outputs else loss 