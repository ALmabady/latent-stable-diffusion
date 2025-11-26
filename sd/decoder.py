import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.module):
    def __init__(self, channels):
        super.__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1 ,channels)
    
    def forward(self, x: torch.tensor) :

        # X --> (Batch_Size, features, Width, Height)
        residu = x

        n, c, w, h = x.shape

        x = x.view(n, c, w*h).permute(0,2,1)  # (Batch_size, Width*Height, features)

        x= self.attention(x)

        x = x.permute(0,2,1).view(n, c, w, h)

        x += residu

        return x


class VAE_ResidualBlock(nn.module):
    def __init__(self,in_channels, out_channels):
        super.__init__()

        self.groupnorm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32,in_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels :
            self.residual_layer = nn.Identity()
        else : 
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x: torch.tensor) :
        # X --> (Batch_Size, in_channels, Width, Height)
        residu = x

        x =  self.groupnorm_1(x)
        
        x = F.silu(x)

        x = self.conv_1(x)

        x =  self.groupnorm_2(x)
        
        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residu)
    
class VAE_Decoder(nn.Sequential):
    def __init__(self,):
        super().__init__(
            nn.Conv2d(8, 8, kernel_size=3, padding=0),

            nn.Conv2d(8, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512,512),

            VAE_AttentionBlock(512),
            
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),
            
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/4, height/4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_size, 512, width/4, height/4) -> (Batch_size, 512, width/4, height/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),


            # (Batch_size, 512, width/4, height/4) -> (Batch_size, 512, width/2, height/2)
            nn.Upsample(scale_factor=2),

            VAE_ResidualBlock(512,256),

            VAE_ResidualBlock(256,256),

            VAE_ResidualBlock(256,256),

            # (Batch_size, 256, width/2, height/4) -> (Batch_size, 256, width, height)
            nn.Upsample(scale_factor=2),


            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256,128),

            VAE_ResidualBlock(128,128),

            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),
            
            # (Batch_size, 128, width, height) -> (Batch_size, 3, width, height)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
            
            )
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale latent then pass through sequential modules
        x = x / 0.18215
        for module in self:
            x = module(x)
        return x
        
