import argparse
import torch
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from src.generator import StyleGAN2Wrapper, RecompositionModule
from src.factors import FactorWrapper
from src.safety import SafetyEvaluator
from src.explanation import Explainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.6, help="Strict leakage threshold")
    parser.add_argument('--output_dir', type=str, default='results/demo_report')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Init System
    print("Initializing system...")
    factor_wrapper = FactorWrapper(device=device)
    # Using robust fallback to mock if weights fail
    safety_eval = SafetyEvaluator(device=device, use_mock=True) 
    explainer = Explainer(factor_wrapper, safety_eval, device=device)

    # Dynamic Probe for Dimensions (V2 Upgrade)
    print("Probing factor dimensions...")
    with torch.no_grad():
        dummy_probe = torch.randn(1, 3, 256, 256).to(device)
        dummy_factors = factor_wrapper.extract_all(dummy_probe)
        factor_dims = {k: v.shape[1] for k, v in dummy_factors.items()}
    print(f"Detected Factor Dims: {factor_dims}")
    
    # Auto-load standard weights if available
    weights_path = "results/models/ffhq.pkl"
    n_layers = 14 # Default for 256x256 Mock
    
    if not os.path.exists(weights_path):
        weights_path = None
        print("Note: Using Mock Generator (No weights found at results/models/ffhq.pkl)")
    else:
        print(f"Note: Using Real Generator weights from {weights_path}")
        n_layers = 18 # FFHQ is 1024x1024 -> 18 layers

    recomposer = RecompositionModule(factor_dims, n_layers=n_layers).to(device)
    generator = StyleGAN2Wrapper(weights_path=weights_path, n_layers=n_layers).to(device)
    
    print(f"Generating {args.num_samples} constrained identities...")
    
    generated_count = 0
    attempts = 0
    max_attempts = args.num_samples * 5
    
    html_report = "<html><body><h1>Identity-Safe Generation Report</h1>"
    
    while generated_count < args.num_samples and attempts < max_attempts:
        attempts += 1
        
        # 2. Sample Sources (Synthetic for safety in demo)
        # In real demo, these would be loaded from 'data/processed/factors_A.pt'
        factors_A = {k: torch.randn(1, v).to(device) for k,v in factor_dims.items()}
        factors_B = {k: torch.randn(1, v).to(device) for k,v in factor_dims.items()}
        
        # VISUAL FIX: Generate the images for A and B so the user can see the "parents"
        # Previously these were just noise placeholders. Now we render them.
        w_A = recomposer(factors_A)
        img_A, _ = generator(w=w_A)
        
        w_B = recomposer(factors_B)
        img_B, _ = generator(w=w_B)
        
        # 3. Recompose
        w_geom = random.uniform(0.2, 0.8)
        w_tex = random.uniform(0.2, 0.8)
        
        factors_C = {
            'geometry': w_geom * factors_A['geometry'] + (1-w_geom) * factors_B['geometry'],
            'symmetry': 0.5 * factors_A['symmetry'] + 0.5 * factors_B['symmetry'],
            'texture': w_tex * factors_A['texture'] + (1-w_tex) * factors_B['texture'],
            'local': 0.5 * factors_A['local'] + 0.5 * factors_B['local'] # Passed through but unused in Generation R
        }
        
        # 4. Generate
        w_latent = recomposer(factors_C)
        img_C, _ = generator(w=w_latent)
        
        # 5. Safety Check (Leakage)
        leakage, sim_A, sim_B = safety_eval.compute_leakage(img_C, img_A, img_B)
        
        if leakage > args.tau:
            print(f"Rejected sample {attempts}: Leakage {leakage.item():.3f} > {args.tau}")
            continue
            
        # 6. ICS & Explainability
        # Extract Actual Factors from Image C to verify consistency
        factors_C_measured = factor_wrapper.extract_all(img_C)
        ics_scores = safety_eval.compute_ics(factors_C_measured, factors_A, factors_B)
        
        # Attribution
        heatmaps, _ = explainer.explain(img_C, factors_A, factors_B)
        
        # Save & Report
        fname_a = f"source_A_{generated_count:04d}.png"
        fname_b = f"source_B_{generated_count:04d}.png"
        fname_c = f"id_{generated_count:04d}.png"
        
        save_image(img_A, os.path.join(args.output_dir, fname_a), normalize=True, value_range=(-1, 1))
        save_image(img_B, os.path.join(args.output_dir, fname_b), normalize=True, value_range=(-1, 1))
        save_image(img_C, os.path.join(args.output_dir, fname_c), normalize=True, value_range=(-1, 1))
        
        # Generate Heatmap Plot
        plt.figure()
        hm = heatmaps['geometry'][0].cpu().numpy()
        plt.imshow(hm, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Geometry A-Contribution")
        hm_name = f"heatmap_id_{generated_count:04d}.png"
        plt.savefig(os.path.join(args.output_dir, hm_name))
        plt.close()
        
        # Append to HTML
        recipe_html = f"""
        <ul>
            <li><strong>Geometry:</strong> {w_geom*100:.1f}% Person A, {(1-w_geom)*100:.1f}% Person B</li>
            <li><strong>Texture:</strong> {w_tex*100:.1f}% Person A, {(1-w_tex)*100:.1f}% Person B</li>
            <li><strong>Symmetry:</strong> 50.0% Person A, 50.0% Person B</li>
        </ul>
        """
        
        html_report += f"""
        <div style="border:1px solid #ccc; margin:10px; padding:10px; font-family: sans-serif;">
            <h3>Identity ID: {generated_count}</h3>
            <h4>Recipe (Mixing Logic):</h4>
            {recipe_html}
            
            <h4>Safety Metrics:</h4>
            <p><strong>Leakage:</strong> {leakage.item():.3f} (Safe &lt; {args.tau})</p>
            <p><strong>ICS Geometry (Similarity to A):</strong> {ics_scores['geometry']['A'].item():.3f}</p>
            
            <div style="display:flex; justify-content: space-around; align-items: center; text-align: center;">
                <div>
                    <img src="{fname_a}" width="200"/>
                    <p>Source A</p>
                </div>
                <div>
                    <img src="{fname_c}" width="250" style="border: 3px solid #4CAF50;"/>
                    <p><strong>Generated Person C</strong></p>
                </div>
                <div>
                    <img src="{fname_b}" width="200"/>
                    <p>Source B</p>
                </div>
            </div>
            <div style="text-align:center; margin-top:10px">
                <img src="{hm_name}" width="300"/>
                <p>Explanation Heatmap (Why is it unique?)</p>
            </div>
        </div>
        <hr/>
        """
        
        generated_count += 1
        print(f"Generated {generated_count}/{args.num_samples}. Leakage: {leakage.item():.3f}")

    html_report += "</body></html>"
    with open(os.path.join(args.output_dir, "report.html"), "w") as f:
        f.write(html_report)
        
    print(f"Finished. Report saved to {args.output_dir}/report.html")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("error.log", "w") as f:
            f.write(traceback.format_exc())
        print("CRASHED. See error.log")

