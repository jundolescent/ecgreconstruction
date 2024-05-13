import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def label_generator_loss(lg_output, input_image):
    target_loss = torch.mean(torch.abs(input_image - lg_output))
    total_disc_loss = target_loss
    return total_disc_loss, target_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    criterion = torch.nn.BCEWithLogitsLoss()
    real_loss = criterion(torch.ones_like(disc_real_output), disc_real_output)
    generated_loss = criterion(torch.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv, alpha):
    criterion = torch.nn.BCEWithLogitsLoss()
    gan_loss = criterion(torch.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = torch.mean(torch.abs(target - ig_output))
    vector_loss = torch.mean(torch.abs(ig_lv - lg_lv))
    total_gen_loss = l1_loss * lambda_ + gan_loss + vector_loss * alpha
    return total_gen_loss, gan_loss, l1_loss, vector_loss