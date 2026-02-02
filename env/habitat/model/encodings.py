import torch
import numpy as np
import tinycudann as tcnn
import torch.nn.functional as F

class HashUncertainty(torch.nn.Module):
    def __init__(self, input_dim=3,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512, uncertainty_res=None,
                cfg={}
                ):
        super(HashUncertainty, self).__init__()
        self.uncertainty_flag = cfg['grid'].get('uncertainty', 'ensemble')
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        encoding_config = {
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            }
        
        if self.uncertainty_flag == 'ensemble':
            self.embed_ensemble = torch.nn.ModuleList()
            base_seed = torch.initial_seed()
            custom_init = cfg['grid'].get('custom_init', False)
            init_gain = cfg['grid'].get('init_gain', 1.0)

            for i in range(cfg['grid'].get('ensemble_size', 5)):
                enc = tcnn.Encoding(
                    n_input_dims=input_dim,
                    encoding_config=encoding_config,
                    dtype=torch.float,
                    seed=base_seed + i 
                )
                if custom_init:
                    for param in enc.parameters():
                        if len(param.shape) == 1:
                            torch.nn.init.xavier_normal_(param.view(-1, level_dim), gain=init_gain)
                        else:
                            torch.nn.init.xavier_normal_(param, gain=init_gain)
                self.embed_ensemble.append(enc)
            self.n_output_dims = self.embed_ensemble[0].n_output_dims
            self.uncertainty_init = cfg['grid'].get('initial_uncert', 1.0e-6)
        else:
            self.embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config=encoding_config,
                dtype=torch.float
            )
            self.n_output_dims = self.embed.n_output_dims

        self.get_uncert_grid(uncertainty_res)


    def get_uncert_grid(self, xyz_dim):
        Nx, Ny, Nz = xyz_dim
        if self.uncertainty_flag == 'NARUTO':
            # Uncertainty initialize to 3
            self.xyz_uncert = torch.nn.parameter.Parameter(torch.ones([Nx, Ny, Nz], device="cuda").float() * 3)
        elif self.uncertainty_flag == 'ensemble':
            if self.uncertainty_init is None:
                self.initialized = False
                self.register_buffer('xyz_uncert', torch.zeros([Nx, Ny, Nz]).float())
            else:
                self.initialized = True
                self.register_buffer('xyz_uncert', torch.ones([Nx, Ny, Nz]).float() * self.uncertainty_init)
        else:
            print('Create Hash Grid with No Uncertainty')
      

    def compute_uncert_grid(self, xyz_sampled):
        """
            @param xyz_sampled: (N,3) query points coordinate in [-1, 1]

        """
        uncert = torch.nn.functional.grid_sample(self.xyz_uncert[None, None, ...], xyz_sampled[None, None, None, ...], 
                                                 align_corners=False)
        return uncert.squeeze()[..., None]
    

    def update_uncert_grid(self, xyz_norm, uncert_val, alpha=0.5):
        """
            for ensemble uncertainty 
            @param xyz_norm: (N,3) query points coordinate in [0, 1]
            @param uncert_val: (N,1) uncertainty value
            @param alpha: EMA update rate
        """
        grid_size = torch.tensor(self.xyz_uncert.shape).to(xyz_norm.device)
        indices = (xyz_norm * (grid_size - 1)).long()
        
        # Clip to be safe
        indices[..., 0] = torch.clamp(indices[..., 0], 0, grid_size[0]-1)
        indices[..., 1] = torch.clamp(indices[..., 1], 0, grid_size[1]-1)
        indices[..., 2] = torch.clamp(indices[..., 2], 0, grid_size[2]-1)
        
        ix = indices[:, 0]
        iy = indices[:, 1]
        iz = indices[:, 2]

        if not self.initialized:
            max_var = uncert_val.max().item()
            self.xyz_uncert.fill_(max_var)
            self.uncertainty_init = max_var
            self.initialized = True
        
        # EMA Update: U_t = alpha * U_obs + (1 - alpha) * U_{t-1}
        self.xyz_uncert[ix, iy, iz] = alpha * uncert_val.squeeze() + (1 - alpha) * self.xyz_uncert[ix, iy, iz]


    def get_uncert_map(self,):
        if self.uncertainty_flag == 'ensemble':
            xyz_uncert_cpu = self.xyz_uncert.cpu()
            if self.uncertainty_init is not None:
                xyz_uncert_cpu = torch.clamp(xyz_uncert_cpu, 0, self.uncertainty_init) 
            uncert_map = xyz_uncert_cpu.numpy().mean(1)[::-1,::-1]
            return uncert_map
        elif self.uncertainty_flag == 'NARUTO':
            uncert_map = self.xyz_uncert.detach().cpu().numpy().mean(1).T[::-1,::-1]
            return uncert_map
        else:
            raise Exception("Unsupported Uncertainty")  
        

    def forward(self, xyz_sampled):
        """
            @param xyz_sampled: (N,3) query points coordinate. [0,1] for tcnn_encoding
        """
        
        if self.uncertainty_flag == 'ensemble':
            embedded = []
            for embed in self.embed_ensemble:
                embedded.append(embed(xyz_sampled))
            embedded = torch.stack(embedded, dim=1) # (B, E, D)
            
            uncertainty = None # for ensemble, uncertainty is computed at final output
        else:
            embedded = self.embed(xyz_sampled)
            if self.uncertainty_flag == 'NARUTO':
                xyz_sampled_norm = (xyz_sampled*2 - 1).to(torch.float32) # to [-1,1]
                uncertainty = self.compute_uncert_grid(xyz_sampled_norm)
            else:
                uncertainty = None
        
        return embedded, uncertainty


    

def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512,
                uncertainty_res=[100,100,100],
                cfg={}):
    """
        @param uncertainty_flag: use what uncertainty
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
        embed = HashUncertainty(input_dim=input_dim, n_levels=n_levels, level_dim=level_dim,
                                base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                desired_resolution=desired_resolution,
                                uncertainty_res=uncertainty_res,
                                cfg=cfg)
        out_dim = embed.n_output_dims

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



