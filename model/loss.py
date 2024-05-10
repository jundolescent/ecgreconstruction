import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LabelGeneratorLoss(nn.Module):
    def forward(self, lg_output, input_image):
        target_loss = torch.mean(torch.abs(input_image - lg_output))
        total_disc_loss = target_loss
        return total_disc_loss, target_loss

class DiscriminatorLoss(nn.Module):
    def forward(self, disc_real_output, disc_generated_output):
        loss_object = nn.BCEWithLogitsLoss()
        real_loss = loss_object(torch.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(torch.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

class InferenceGeneratorLoss(nn.Module):
    def __init__(self, lambda_, alpha):
        super(InferenceGeneratorLoss, self).__init__()
        self.lambda_ = lambda_
        self.alpha = alpha
        self.loss_object = nn.BCEWithLogitsLoss()

    def forward(self, disc_generated_output, ig_output, target, ig_lv, lg_lv):
        gan_loss = self.loss_object(torch.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = torch.mean(torch.abs(target - ig_output))
        vector_loss = torch.mean(torch.abs(ig_lv - lg_lv))
        total_gen_loss = l1_loss * self.lambda_ + gan_loss + vector_loss * self.alpha
        return total_gen_loss, gan_loss, l1_loss, vector_loss

class MyLRSchedule:
    def __init__(self, initial_learning_rate, path, batch_size):
        self.initial_learning_rate = initial_learning_rate
        self.path = path
        self.batch_size = batch_size

    def __call__(self, step):
        if step < self.path * 5:
            return self.initial_learning_rate
        elif step % self.path == 0:
            return self.initial_learning_rate * 0.95
        else:
            return self.initial_learning_rate

# # Example usage:
# lg_loss = LabelGeneratorLoss()
# disc_loss = DiscriminatorLoss()
# ig_loss = InferenceGeneratorLoss(lambda_=0.1, alpha=0.01)
# optimizer = optim.Adam(model.parameters(), lr=MyLRSchedule(initial_learning_rate=0.001, path=1000, batch_size=32))