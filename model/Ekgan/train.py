import torch
import torch.nn as nn
from ekgan import InferenceGenerator, Discriminator, LabelGenerator
from loss import label_generator_loss, inference_generator_loss, discriminator_loss
from Dataloader.ECGDataLoader import min_max_scaling, ECGDataLoader
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
class ECGekgan:
    def __init__(self, input_size=1000, encoding_dim=128, learning_rate=0.001, batch_size=64,
                 num_epochs=10, path='', test_fold=10, sampling_rate=100, lead=2):
        self.test_fold = test_fold
        self.path = path
        self.sampling_rate = sampling_rate

        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GI = InferenceGenerator().to(self.device)
        self.GL = LabelGenerator().to(self.device)
        self.DC = Discriminator().to(self.device)

        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.data_loader = ECGDataLoader(path, sampling_rate)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_loader.preprocess_data(self.test_fold)
        self.X_train_scaled = min_max_scaling(self.X_train)
        self.X_test_scaled = min_max_scaling(self.X_test)
        # self.logger = JSLogger(level=level, log_file='../log.txt')
        self.lead = lead

    def train(self, inference_generator_optimizer, discriminator_optimizer, label_generator_optimizer, lambda_, alpha):

        train_dataset = TensorDataset(torch.tensor(self.X_train_scaled, dtype=torch.float32),
                                      torch.tensor(self.y_train[:, :, self.lead], dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.GI.train()
        self.GL.train()
        self.DC.train()

        for epoch in range(self.num_epochs):
            with tqdm(train_loader, unit="batch") as tepoch:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                tepoch.set_description(f"Epoch {epoch}")
                # input_image = input_image.to(self.device)
                # target = target.to(self.device)

                inference_generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                label_generator_optimizer.zero_grad()

                ig_lv, ig_output = self.GI(inputs)
                lg_lv, lg_output = self.GL(inputs)
                disc_real_output = self.DC([inputs, targets], dim=1) # dim=1이 맞나...?
                disc_generated_output = self.DC([inputs, ig_output])

                total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, inputs)
                total_ig_loss, ig_adversarial_loss, ig_l1_loss, vector_loss = inference_generator_loss(
                    disc_generated_output, ig_output, targets, lambda_, ig_lv, lg_lv, alpha)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

                total_ig_loss.backward(retain_graph=True)
                ig_grads = [p.grad for p in self.GI.parameters() if p.grad is not None]

                disc_loss.backward(retain_graph=True)
                disc_grads = [p.grad for p in self.DC.parameters() if p.grad is not None]

                total_lg_loss.backward(retain_graph=True)
                lg_grads = [p.grad for p in self.GL.parameters() if p.grad is not None]

                inference_generator_optimizer.step()
                discriminator_optimizer.step()
                label_generator_optimizer.step()

                print('epoch {} gen_total_loss {} ig_adversarial_loss {} ig_l1_loss {} lg_l2_loss {} vector_loss {}  '.format(
                    epoch, total_ig_loss, ig_adversarial_loss, ig_l1_loss, lg_l1_loss, vector_loss))

