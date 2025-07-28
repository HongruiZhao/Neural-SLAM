import torch
import numpy as np
import tinycudann as tcnn
import torch.nn.functional as F
import math
class TensorCP(torch.nn.Module):
    def __init__(self, tensors_dim, cp_rank, tensor_f_dim, uncertainty):
        """
            @param tensors_dim: list. the first item for coarse xyz tensors, the second for fine
            @param cp_rank: rank for tensor CP decomposition 
            @param tensor_f_dim: dimension of feature output from tensor encoder
            @param uncertainty: if tensor CP should learn uncertainty
        """
        super(TensorCP, self).__init__()
        self.tensorxyz_coarse = self.init_tensors(cp_rank, tensors_dim[0], 0.2)
        self.tesnorxyz_fine = self.init_tensors(cp_rank, tensors_dim[1], 0.2)
        self.feature_tensor_coarse = torch.nn.Linear(cp_rank, tensor_f_dim, bias=False)
        self.feature_tensor_fine = torch.nn.Linear(cp_rank, tensor_f_dim, bias=False)

        self.uncertainty_flag = uncertainty
        if self.uncertainty_flag == 'tensor':
            uncert_rank = cp_rank
            # initial spatial uncertainty = 3
            scale = math.pow(3/uncert_rank, 1/3)
            self.xyz_uncert = \
                self.init_uncer_tensors(uncert_rank, tensors_dim[0], scale)
        elif self.uncertainty_flag == 'grid':
            self.xyz_uncert = self.get_uncert_grid(tensors_dim[0])


    def get_uncert_grid(self, xyz_dim):
        Nx = xyz_dim[0]
        Ny = xyz_dim[1]
        Nz = xyz_dim[2]
        # Uncertainty initialize to 3
        self.uncert_grid = torch.nn.parameter.Parameter(torch.ones([Nx, Ny, Nz], device="cuda").float() * 3)
        self.cache_uncert = np.zeros([Nx, Ny, Nz], dtype=np.float32)
        return self.uncert_grid


    def init_uncer_tensors(self, cp_rank, tensors_dim, scale):
        line_coef = []
        for i in range(len(tensors_dim)):    
            line_coef.append(
                torch.nn.Parameter( scale * torch.ones((1, cp_rank, tensors_dim[i], 1)).float() ) 
                )
        return torch.nn.ParameterList(line_coef) # num_modes * (1, num_ranks, x/y/z dim, 1)


    def init_tensors(self, cp_rank, tensors_dim, scale):
        line_coef = []
        for i in range(len(tensors_dim)):    
            line_coef.append(
                torch.nn.Parameter( scale * torch.randn((1, cp_rank, tensors_dim[i], 1)) )
                )
        return torch.nn.ParameterList(line_coef) # num_modes * (1, num_ranks, x/y/z dim, 1)

    
    def compute_feature(self, xyz_sampled, xyztensors):
        """
            @param xyz_sampled: (N,3) query points coordinate
            @param xyztensors:
            @param feature_tensor: the fourthe tensor 
            @return feature: (N,R)
        """
        coordinate_line = torch.stack( [xyz_sampled[..., i] for i in range(3)] )
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2) # so last dimension will be (x,y)=(0,index)

        # xyztensors[0] has shape (1,num_ranks,tensor_size,1)
        # coordinate_line[[0]] has shape (1,N,1,2)
        line_coef_point = F.grid_sample(xyztensors[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, xyz_sampled.shape[0])  
        line_coef_point = line_coef_point * F.grid_sample(xyztensors[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, xyz_sampled.shape[0])
        line_coef_point = line_coef_point * F.grid_sample(xyztensors[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, xyz_sampled.shape[0])
        
        return line_coef_point.T 
    

    def compute_uncert_grid(self, xyz_smapled):
        """
            @param xyz_sampled: (N,3) query points coordinate
        """

        uncert = torch.nn.functional.grid_sample(self.uncert_grid[None, None, ...], xyz_smapled[None, None, None, ...], 
                                                 align_corners=False)
        return uncert.squeeze()[..., None]


    def forward(self, xyz_sampled):
        """
            @param xyz_sampled: (N,3) query points coordinate. [0,1] for tcnn_encoding
            @return cat_feature: (N,f_dim*2) 
            @return uncertainty: (N,1) 
        """
        xyz_sampled = (xyz_sampled*2 - 1).to(torch.float32) # to [-1,1]

        coarse_feature = self.compute_feature(xyz_sampled, self.tensorxyz_coarse)
        coarse_feature = self.feature_tensor_coarse(coarse_feature)
        
        fine_feature  = self.compute_feature(xyz_sampled, self.tesnorxyz_fine)
        fine_feature = self.feature_tensor_fine(fine_feature)
    
        cat_feature = torch.cat((coarse_feature, fine_feature), dim=-1)

        # (N,1)
        if self.uncertainty_flag == 'tensor':
            uncertainty = self.compute_feature(xyz_sampled, self.xyz_uncert).sum(-1).unsqueeze(1)
        elif self.uncertainty_flag == 'grid':
            uncertainty = self.compute_uncert_grid(xyz_sampled)
        else:
            uncertainty = None
        
        return cat_feature, uncertainty 
    


        




def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512, tensors_dim=[ [256,256,256], [512,512,512]],
                cp_rank=4, tensor_f_dim=16, uncertainty=True):
    """
        @param tensors_dim: list. the first item for coarse xyz tensors, the second for fine
        @param cp_ranK: rank for tensor CP decomposition 
        @param tensor_f_dim: dimension of feature output from tensor encoder
        @param uncertainty: if tensor CP should learn uncertainty
    """
    
    # Dense grid encoding
    if 'dense' in encoding.lower():
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": level_dim,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims
    
    # Sparse grid encoding
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

        # # TODO: use pytorch implementation of hah grid?
        # from gridencoder import GridEncoder
        # embed = GridEncoder(input_dim=input_dim, num_levels=n_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=False)
        # out_dim = embed.output_dim

    # tensor RF
    elif 'tensor' in encoding.lower():
        embed = TensorCP(tensors_dim, cp_rank, tensor_f_dim, uncertainty)
        out_dim = tensor_f_dim*2


    # Spherical harmonics encoding
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", #Component type.
	            "n_bins": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Frequency", 
                "n_frequencies": n_frequencies
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Identity encodingk
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim



# dimension check 
if __name__ == "__main__":
    embed, out_dim = get_encoder(encoding='tensor')
    xyz_sampled = torch.rand((100,3))
    out = embed(xyz_sampled)

    for name, p in embed.named_parameters():
        print(f'name={name}, shape={p.shape}')

    print("End of debugging")



