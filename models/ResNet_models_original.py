import torch
import torch.nn as nn
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np

"""prior net
    input: x
    output: latent Gaussian variable z, mu, sigma"""
class Encoder_x(nn.Module):
    def __init__(self, input_channels,latent_size,channels= 64):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        # no more resolution reduction!
        # self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(channels * 8)
        # self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        # self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)

        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)


        # when latent size = 4
        # self.fc1_3 = nn.Linear(channels * 4 * 6 * 6, latent_size)
        # self.fc2_3 = nn.Linear(channels * 4 * 6 * 6, latent_size)

        # when latent size = 8
        self.fc1_3 = nn.Linear(channels * 8 * 6 * 6, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 6 * 6, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        # print(input.shape)
        # [128, 1, 48, 48]
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        print("output_size ",output.size())
        # output_size = [128, 256, 6, 6]
        
        # No more resolution decrease
        # output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        # output = self.leakyrelu(self.bn4(self.layer5(output)))

        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            """latent gaussian variable"""
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            
            output = output.view(-1, self.channel * 8 * 11 * 11)


            mu = self.fc1_2(output) #mu
            
            logvar = self.fc2_2(output) #varaince
            #gaussian distribution
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else: # almost always into here
            # print('************************bigger********************')
            # output [128, 256, 6, 6]
            # latent var = 6
            # print(output.shape)
            # latent_dim = 4
            # output = output.view(-1, self.channel * 4 * 6 * 6 )
            # latent_dim = 8
            output = output.view(-1, self.channel * 8 * 6 * 6 )
    

            mu = self.fc1_3(output)
            print("mu: " ,mu.shape)
            logvar = self.fc2_3(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar



# # 여기서 몇 개 뽑아가고
# class Generator(nn.Module):
#     def __init__(self, channel, latent_dim):
#         super(Generator, self).__init__()
#         self.relu = nn.ReLU(inplace=True)

#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         # self.xy_encoder = Encoder_xy(4, latent_dim)
#         self.x_encoder = Encoder_x(1, latent_dim)
#         self.sal_endecoder = Saliency_feat_endecoder(channel, latent_dim)
#         self.tanh = nn.Tanh()

#     def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
#         return block(dilation_series, padding_series, NoLabels, input_channel)

#     def kl_divergence(self, posterior_latent_space, prior_latent_space):
#         kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
#         return kl_div

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = torch.cuda.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)

#     def forward(self, x, y=None, training=True):
#         if training:
#             # #posterior net
#             # self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x,y),1))
#             #prior net
#             self.prior, mux, logvarx = self.x_encoder(x)  #mu,var,(6,8), (batch,latent_dim)
#             #kl two dist
#             mu=torch.zeros(mux.shape).cuda()
#             logvar=torch.ones(logvarx.shape).cuda()
#             z_noise = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
#             #latent gaussian variable
#             latent_loss = torch.mean(self.kl_divergence(z_noise, self.prior))
#             z_noise_prior = self.reparametrize(mux, logvarx)
          
#             #generator model
#             # self.pred_post_init, self.pred_post_ref  =  self.sal_endecoder (x,z_noise_post)
#             self.pred_prior_init, self.pred_prior_ref = self.sal_endecoder (x ,z_noise_prior)
#             return  self.pred_prior_init, self.pred_prior_ref, latent_loss
#         else:
#             #sample for prior distribution
#             _, mux, logvarx = self.x_encoder(x) #inference net
#             z_noise_prior = self.reparametrize(mux, logvarx)
#             _, self.prob_pred  = self.sal_endecoder(x,z_noise_prior)
#             return self.prob_pred




