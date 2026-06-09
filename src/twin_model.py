import torch
import sys
import torch.nn as nn
from ultralytics.nn.modules import Detect
from copy import deepcopy

class ChannelAttention(nn.Module):
    """
    Computes a weight for each channel (The 'What').
    Structure: GlobalPool -> FC -> ReLU -> FC -> Sigmoid
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Reduce dimension by 'ratio' to save parameters
        hidden_planes = max(in_planes // ratio, 8) 
        
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class MultiScaleSpatialGate(nn.Module):
    """
    Multi-scale spatial gate using parallel convolutions at different scales.
    Captures shuttle features at multiple receptive fields (3x3, 5x5, 7x7).
    
    VERSION 4.0: Improved spatial awareness for better shuttle detection.
    """
    def __init__(self, in_channels):
        super().__init__()
        hidden = max(in_channels // 4, 16)
        
        # Multi-scale branches
        self.branch_3x3 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.branch_5x5 = nn.Conv2d(in_channels, hidden, 5, padding=2)
        self.branch_7x7 = nn.Conv2d(in_channels, hidden, 7, padding=3)
        
        # Fusion layer (3 branches -> 1 mask)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)
        b7 = self.branch_7x7(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([b3, b5, b7], dim=1)
        
        # Fuse to spatial mask
        return self.fuse(multi_scale)

class ContextGuidedDetect(nn.Module):
    """
    VERSION 4.0: ADAPTIVE RESIDUAL TWIN WITH MULTI-SCALE SPATIAL GATES
    - Upgrade: MultiScaleSpatialGate (3x3, 5x5, 7x7) for better spatial awareness
    - Fix: Correctly inspects first/last layers to handle channel reduction in YOLO heads.
    """
    def __init__(self, base_detect: Detect, nc_new=1):
        super().__init__()
        # Standard Setup
        self.nc = nc_new
        self.nl = base_detect.nl
        self.reg_max = base_detect.reg_max
        self.no = nc_new + self.reg_max * 4
        self.stride = base_detect.stride
        self.f = base_detect.f
        self.i = base_detect.i
        
        # 1. The Veteran (Frozen)
        self.context_head = deepcopy(base_detect)
        for p in self.context_head.parameters():
            p.requires_grad = False
            
        # 2. The Recruit (Trainable)
        self.shuttle_head = deepcopy(base_detect)
        self.shuttle_head.nc = nc_new
        self.shuttle_head.no = self.no
        
        # Re-init Recruit Layers
        for i in range(self.nl):
            old_conv = self.shuttle_head.cv3[i][-1]
            # Back to nc_new (1). Total output = cv2(64) + cv3(1) = 65.
            new_conv = nn.Conv2d(old_conv.in_channels, nc_new, 1, 1, 0).to(old_conv.weight.device)
            nn.init.constant_(new_conv.bias, 0)
            nn.init.normal_(new_conv.weight, std=0.01)
            self.shuttle_head.cv3[i][-1] = new_conv

        # 3. The New Bridge Modules
        self.channel_attns = nn.ModuleList()
        self.spatial_gates = nn.ModuleList()
        self.alphas = nn.ParameterList()
        self.projections = nn.ModuleList() 
        
        for i in range(self.nl):
            # --- CRITICAL FIX: Correct Channel Inspection ---
            # cv2[i] is a Sequential block.
            # ch_in: The input to the whole block (First Layer input)
            # ch_internal: The output of the whole block (Last Layer output)
            
            seq_block = base_detect.cv2[i]
            first_layer = seq_block[0]
            last_layer = seq_block[-1]
            
            # Get ch_in (Input to the Projection target)
            if hasattr(first_layer, 'conv'): # Ultralytics Conv
                ch_in = first_layer.conv.in_channels
            else: # Standard Conv2d
                ch_in = first_layer.in_channels
                
            # Get ch_internal (Input to Attention)
            if hasattr(last_layer, 'conv'):
                ch_internal = last_layer.conv.out_channels
            else:
                ch_internal = last_layer.out_channels
            
            print(f"DEBUG: Head {i} | Input: {ch_in} -> Internal: {ch_internal}")

            # A. Channel Attention (Operates on internal dim)
            self.channel_attns.append(ChannelAttention(ch_internal))
            
            # B. Spatial Gate (Operates on internal dim)
            # UPGRADE 2: Multi-Scale Spatial Gate (3x3, 5x5, 7x7)
            self.spatial_gates.append(MultiScaleSpatialGate(ch_internal))
            
            # C. Projection Layer (Internal -> Input)
            # Maps ch_internal back to ch_in so we can add them
            self.projections.append(nn.Conv2d(ch_internal, ch_in, 1, bias=False))
            
            # D. Learnable Alpha
            self.alphas.append(nn.Parameter(torch.tensor(0.5)))

    def forward(self, x):
        recruit_inputs = []
        
        for i, feat in enumerate(x):
            # 1. Ask Veteran (Downsamples Input -> Internal)
            with torch.no_grad():
                vet_feats = self.context_head.cv2[i](feat)
            
            # 2. Channel Attention
            channel_scale = self.channel_attns[i](vet_feats)
            refined_vet_feats = vet_feats * channel_scale
            
            # 3. Spatial Gate
            spatial_mask = self.spatial_gates[i](refined_vet_feats)
            highlight = refined_vet_feats * spatial_mask
            
            # 4. Project Highlight back to Input Size
            projected_highlight = self.projections[i](highlight)
            
            # 5. Residual Injection
            enhanced_feat = feat + (self.alphas[i] * projected_highlight)
            
            recruit_inputs.append(enhanced_feat)
            
        # 6. Run Recruit Head
        recruit_out = self.shuttle_head(recruit_inputs)
        
        # DEBUG: Print shapes on first run
        if not hasattr(self, 'debug_printed'):
            print(f"\n>>> DEBUG: ContextGuidedDetect Forward <<<")
            print(f"  self.no: {self.no}")
            print(f"  Batch size (from x[0]): {x[0].shape[0]}")
            for i, r_out in enumerate(recruit_out):
                print(f"  Head {i} Result Type: {type(r_out)}", flush=True)
                if isinstance(r_out, torch.Tensor):
                    print(f"    Shape: {r_out.shape}", flush=True)
                elif isinstance(r_out, (list, tuple)):
                    print(f"    List/Tuple length: {len(r_out)}", flush=True)
                    for j, sub_out in enumerate(r_out):
                        if isinstance(sub_out, torch.Tensor):
                            print(f"      Sub-item {j} Shape: {sub_out.shape}", flush=True)
                        else:
                            print(f"      Sub-item {j} Type: {type(sub_out)}", flush=True)
            self.debug_printed = True
            sys.stdout.flush()
            
        if self.training:
            return recruit_out
        elif getattr(self, 'return_twin', False):
            veteran_out = self.context_head(x)
            return recruit_out, veteran_out
        else:
            return recruit_out