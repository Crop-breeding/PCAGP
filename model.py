import torch
import torch.nn as nn

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=48, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=48, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=48, kernel_size=1, stride=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        s_h = nn.functional.interpolate(s_h, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        s_w = nn.functional.interpolate(s_w, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        return x * s_h * s_w


class PCAGP(nn.Module):
    def __init__(self, h, w) -> None:
        super(PCAGP, self).__init__()

        self.parallel_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(),
                nn.MaxPool2d((4, 4), 4)
            ),
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(),
                nn.MaxPool2d((4, 4), 4)
            ),
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3, bias=True),
                nn.LeakyReLU(),
                nn.MaxPool2d((4, 4), 4)
            )
        ])

        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU()

        self.ca_block = CA_Block(48,46,46, reduction=16)


        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(48*46*46, 32),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Dropout(0.05)
        )

    def forward(self, x):

        conv_outs = [conv(x) for conv in self.parallel_conv_layers]
        x = torch.cat(conv_outs, dim=1)  # Concatenate the parallel convolution outputs

        x = self.dropout(x)
        x = self.leaky_relu(x)

        x = self.ca_block(x)

        x = self.fc_layers(x)
        return x