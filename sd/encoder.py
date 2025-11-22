import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_ResidualBlock, VAE_AttentionBlock

class VAE_encoder(nn.Sequential):
    def __init__(self):
        super.__init(

            # (Batch_size, channel, width, height) --> (Batch_size, 128, width, height)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_size, 128, width, height) --> (Batch_size, 128, width, height)
            VAE_ResidualBlock(128,128),

            # (Batch_size, 128, width, height) --> (Batch_size, 128, width, height)
            VAE_ResidualBlock(128,128),

            # (Batch_size, 128, width, height) --> (Batch_size, 128, width/2, height/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),

            # (Batch_size, 128, width/2, height/2) --> (Batch_size, 256, width/2, height/2)
            VAE_ResidualBlock(128,256),

            # (Batch_size, 256, width/2, height/2) --> (Batch_size, 256, width/2, height/2)
            VAE_ResidualBlock(256,256),

            # (Batch_size, 256, width/2, height/2) --> (Batch_size, 256, width/4, height/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),

            # (Batch_size, 256, width/4, height/4t) --> (Batch_size, 512, width/4, height/4)
            VAE_ResidualBlock(256,512),

            # (Batch_size, 512, width/4, height/4) --> (Batch_size, 512, width/4, height/4)
            VAE_ResidualBlock(512,512),

            # (Batch_size, 256, width/4, height/4) --> (Batch_size, 256, width/8, height/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),

            # (Batch_size, 512, width/8, height/8) --> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),

            # (Batch_size, 512, width/8, height/8) --> (Batch_size, 512, width/8, height/8)
            VAE_AttentionBlock(512),

            # (Batch_size, 512, width/8, height/8) --> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512,512),

            # (Batch_size, 512, width/8, height/8) --> (Batch_size, 512, width/8, height/8)
            nn.GroupNorm(32, 512),

            # (Batch_size, 512, width/8, height/8) --> (Batch_size, 512, width/8, height/8)
            nn.SiLU(),


            # (Batch_size, 512, width/8, height/8) --> (Batch_size, 8, width/8, height/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_size, 8, width/8, height/8) --> (Batch_size, 8, width/8, height/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),

        )

    """
    learn the Mu and sigma of tha latent space to sample from it not the compressed images
    """
    
    def forward(self, x: torch.tensor, noise: torch.tensor):
        # x : (Batch_size, Input_channel, Width, Height)
        # noise : ((Batch_size, Input_channel, Width/8, Height/8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2) :
                # (padding_left, Padding_Rigth, padding_Top, padding_bottom)
                x = F.pad(x, (0,1,0,1))
            
            x = module(x)
        
        # (Batch_size, 8, width/8, height/8) --> two tensors of shape : (Batch_size, 4, width/8, height/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Batch_size, 4, width/8, height/8) -> Batch_size, 4, width/8, height/8)
        log_variance = torch.clamp(log_variance,-30, 20)

        # Batch_size, 4, width/8, height/8) --> Batch_size, 4, width/8, height/8)
        variance = log_variance.exp()
        
        # Batch_size, 4, width/8, height/8) --> Batch_size, 4, width/8, height/8)
        stdev = variance.sqrt()


        # Z = N(0,1) -->N(mean,variance)
        x = mean + stdev * noise 

        # Scale by constant 
        
        x *= 0.18215

        return x
