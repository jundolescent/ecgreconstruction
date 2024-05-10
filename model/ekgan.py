import torch
import torch.nn as nn
import torch.nn.init as init

initializer = init.normal_

class InferenceGenerator(nn.Module):
    def __init__(self):
        super(InferenceGenerator, self).__init__()

        # 모델의 필터 수 및 커널 크기 정의
        filters_encoder = [64, 128, 256, 512, 1024]
        filters_decoder = [512, 256, 128, 64, 1]
        kernel_size = (2, 4)
        stride = [2, 2, 2, 2, (1, 2)]

        # FirstBlock
        self.conv2d_1 = torch.nn.Conv2d(in_channels=1, out_channels=filters_encoder[0], kernel_size=kernel_size, stride=stride[0], padding='same', bias=False)
        for param in self.conv2d_1.parameters():
            if param.requires_grad:
                initializer(param.data, mean=0., std=0.02)
        self.bn = torch.nn.BatchNorm2d(num_features=1, eps=0.001, momentum=0.99)
        self.activation = nn.LeakyReLU(negative_slope=0.3) # , inplace=True
        self.activation_d = nn.ReLU() # inplace=True

        # SecondBlock
        self.conv2d_2 = torch.nn.Conv2d(in_channels=1, out_channels=filters_encoder[2], kernel_size=kernel_size,
                                        stride=stride[2], padding='same', bias=False)
        # ThirdBlock
        self.conv2d_3 = torch.nn.Conv2d(in_channels=1, out_channels=filters_encoder[3], kernel_size=kernel_size,
                                        stride=stride[3], padding='same', bias=False)
        # FourthBlock
        self.conv2d_4 = torch.nn.Conv2d(in_channels=1, out_channels=filters_encoder[4], kernel_size=kernel_size,
                                        stride=stride[4], padding='same', bias=False)
        # FifthBlock
        self.conv2d_5 = torch.nn.Conv2d(in_channels=1, out_channels=filters_encoder[5], kernel_size=kernel_size,
                                        stride=stride[5], padding='same', bias=False)
        # Decoder FirstBlock -> exclude padding='same'
        self.conv2d_d_1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=filters_decoder[0], kernel_size=kernel_size,
                                                   stride=stride[0], bias=False)
        # Decoder SecondBlock
        self.conv2d_d_2 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=filters_decoder[1], kernel_size=kernel_size,
                                                   stride=stride[1], bias=False)
        # Decoder ThirdBlock
        self.conv2d_d_3 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=filters_decoder[2], kernel_size=kernel_size,
                                                   stride=stride[2], bias=False)
        # Decoder FourthBlock
        self.conv2d_d_4 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=filters_decoder[3], kernel_size=kernel_size,
                                                   stride=stride[3], bias=False)
        # Decoder FifthBlock
        self.conv2d_d_5 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=filters_decoder[4], kernel_size=kernel_size,
                                                   stride=stride[4], bias=False)
    def forward(self, x):
        # Encoder FirstBlock
        x = self.conv2d_1(x)
        e1 = self.activation(x)

        # Encoder SecondBlock
        x = self.conv2d_2(e1)
        x = self.bn(x)
        e2 = self.activation(x)

        # Encoder ThridBlock
        x = self.conv2d_3(e2)
        x = self.bn(x)
        e3 = self.activation(x)

        # Encoder FourthBlock
        x = self.conv2d_4(e3)
        x = self.bn(x)
        e4 = self.activation(x)

        # Encoder FifthBlock
        x = self.conv2d_5(e4)
        x = self.bn(x)
        e5 = self.activation(x)
        # Encoder latent vector -> e5

        # Decoder FirstBlock
        x = self.conv2d_d_1(e5)
        x = self.bn(x)
        d1 = self.activation_d()

        # Decoder SecondBlock
        x = torch.cat((d1, e4), dim=-1)
        x = self.conv2d_d_2(x)
        x = self.bn(x)
        d2 = self.activation_d()

        # Decoder ThirdBlock
        x = torch.cat((d2, e3), dim=-1)
        x = self.conv2d_d_3(x)
        x = self.bn(x)
        d3 = self.activation_d()

        # Decoder FourthBlock
        x = torch.cat((d3, e2), dim=-1)
        x = self.conv2d_d_4(x)
        x = self.bn(x)
        d4 = self.activation_d()

        # Decoder FifthBlock
        x = torch.catr((d4, e1), dim=-1)
        x = self.conv2d_d_5(x)
        # Encoder Latent Vector, Output
        return e5, x

class LabelGenerator(nn.Module):
    def __init(self):
        super(LabelGenerator, self).__init__()
        filters_encoder = [32, 64, 128, 256, 512]
        filters_decoder = [256, 128, 64, 32, 1]
        kernel = [64, 32, 16, 8, 4]
        stride = [4, 4, 4, 2, 2]





