import torch
import torch.nn.functional as F
import numpy as np

class Explainer:
    """
    Computes region-based importance for Identity Contribution Scores.
    """
    def __init__(self, factor_wrapper, safety_evaluator, device='cpu'):
        self.factor_wrapper = factor_wrapper
        self.safety_evaluater = safety_evaluator
        self.device = device
        
    def explain(self, img_C, factors_A, factors_B, grid_size=4):
        """
        Perturbs img_C in a grid and measures drop in ICS.
        Returns: Heatmap (H, W, num_factors) showing contribution to 'A'.
        """
        B, C, H, W = img_C.shape
        # Base ICS
        factors_C = self.factor_wrapper.extract_all(img_C)
        base_ics = self.safety_evaluater.compute_ics(factors_C, factors_A, factors_B)
        
        # Heatmap storage
        heatmap = {}
        for k in base_ics.keys():
            heatmap[k] = torch.zeros(B, grid_size, grid_size).to(self.device)
            
        step_h = H // grid_size
        step_w = W // grid_size
        
        for r in range(grid_size):
            for c in range(grid_size):
                # PERTURBATION: Occlude region
                img_perturbed = img_C.clone()
                y1, y2 = r*step_h, (r+1)*step_h
                x1, x2 = c*step_w, (c+1)*step_w
                img_perturbed[:, :, y1:y2, x1:x2] = 0 # Black out
                
                # Extract new factors
                factors_p = self.factor_wrapper.extract_all(img_perturbed)
                ics_p = self.safety_evaluater.compute_ics(factors_p, factors_A, factors_B)
                
                # Attrib = Drop in ICS_A (If occlusion drops ICS_A, this region was important for A)
                for k in base_ics.keys():
                    drop = base_ics[k]['A'] - ics_p[k]['A']
                    heatmap[k][:, r, c] = drop
                    
        return heatmap, base_ics
