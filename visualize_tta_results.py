"""
Generate publication-quality visualizations for TTA results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class TTAVisualizer:
    """Generate figures for TTA paper section"""
    
    def __init__(self, results_path='tta_results/all_results.json'):
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.output_dir = Path('tta_figures')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
    
    def plot_method_comparison(self):
        """Figure: Bar chart comparing all TTA methods"""
        methods = []
        dice_scores = []
        dice_stds = []
        colors = []
        
        for method, data in self.results.items():
            methods.append(method.replace('_', '\n'))
            dice_scores.append(data['dice_mean'])
            dice_stds.append(data['dice_std'])
            colors.append('red' if method == 'baseline' else 'steelblue')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(methods))
        bars = ax.bar(x, dice_scores, yerr=dice_stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Highlight best performer
        best_idx = np.argmax(dice_scores)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)
        
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Dice Coefficient (%)', fontsize=14, fontweight='bold')
        ax.set_title('Test-Time Adaptation: Performance Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([min(dice_scores) - 5, max(dice_scores) + 5])
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, dice_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tta_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tta_method_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Method comparison saved")
    
    def plot_hyperparameter_sensitivity(self):
        """Figure: Learning rate and steps sensitivity"""
        # Extract data for combined method
        lr_data = {}
        steps_data = {}
        
        for method, data in self.results.items():
            if 'TTA_BN_Entropy' in method:
                if 'LR' in method:
                    lr = data.get('lr', 0)
                    lr_data[lr] = data['dice_mean']
                if 'S' in method and 'LR1e-4' in method:
                    steps = data.get('adapt_steps', 0)
                    steps_data[steps] = data['dice_mean']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Learning rate sensitivity
        if lr_data:
            lrs = sorted(lr_data.keys())
            scores = [lr_data[lr] for lr in lrs]
            ax1.plot(lrs, scores, marker='o', markersize=10, linewidth=2, color='steelblue')
            ax1.set_xscale('log')
            ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Mean Dice (%)', fontsize=12, fontweight='bold')
            ax1.set_title('Learning Rate Sensitivity', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Adaptation steps sensitivity
        if steps_data:
            steps = sorted(steps_data.keys())
            scores = [steps_data[s] for s in steps]
            ax2.plot(steps, scores, marker='s', markersize=10, linewidth=2, color='coral')
            ax2.set_xlabel('Adaptation Steps', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Mean Dice (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Adaptation Steps Sensitivity', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tta_hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tta_hyperparameter_sensitivity.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Hyperparameter sensitivity saved")
    
    def plot_per_class_improvement(self):
        """Figure: Per-class Dice improvement with TTA"""
        baseline_dice = np.array(self.results['baseline']['dice_per_class'])
        
        # Find best TTA method
        best_method = max(self.results.items(), 
                         key=lambda x: x[1]['dice_mean'] if x[0] != 'baseline' else 0)
        tta_dice = np.array(best_method[1]['dice_per_class'])
        
        improvement = tta_dice - baseline_dice
        
        # Organ names (adjust based on your dataset)
        organs = ['Background', 'Aorta', 'Gallbladder', 'Kidney (L)', 'Kidney (R)',
                 'Liver', 'Pancreas', 'Spleen', 'Stomach', 'Class 9', 'Class 10',
                 'Class 11', 'Class 12', 'Class 13']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(organs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_dice, width, label='Baseline', 
                      alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, tta_dice, width, label=f'TTA ({best_method[0]})',
                      alpha=0.8, color='skyblue')
        
        ax.set_xlabel('Organ Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dice Coefficient (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance: Baseline vs TTA', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(organs, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tta_per_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tta_per_class_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class comparison saved")
        
        # Print improvement summary
        print(f"\n=== Per-Class Improvement (%) ===")
        for organ, imp in zip(organs, improvement):
            print(f"{organ:15s}: {imp:+.2f}%")
    
    def generate_latex_table(self):
        """Generate LaTeX table for paper"""
        latex = r"""\begin{table}[H]
\centering
\caption{Test-Time Adaptation Results}
\label{tab:tta_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Dice (\%)} & \textbf{IoU (\%)} & \textbf{HD95 (mm)} & \textbf{$\Delta$ Dice} \\
\midrule
"""
        
        baseline_dice = self.results['baseline']['dice_mean']
        
        for method, data in self.results.items():
            dice = data['dice_mean']
            dice_std = data['dice_std']
            iou = data['iou_mean']
            iou_std = data['iou_std']
            hd95 = data['hd95_mean']
            hd95_std = data['hd95_std']
            delta = dice - baseline_dice
            
            method_name = method.replace('_', ' ')
            latex += f"{method_name} & {dice:.2f}±{dice_std:.2f} & {iou:.2f}±{iou_std:.2f} & "
            latex += f"{hd95:.2f}±{hd95_std:.2f} & {delta:+.2f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save to file
        with open(self.output_dir / 'tta_latex_table.tex', 'w') as f:
            f.write(latex)
        
        print(f"✓ LaTeX table saved to {self.output_dir / 'tta_latex_table.tex'}")
        print("\nLaTeX Table Preview:")
        print(latex)
    
    def generate_all_figures(self):
        """Generate all publication figures"""
        print("\n" + "="*60)
        print("GENERATING TTA VISUALIZATIONS")
        print("="*60)
        
        self.plot_method_comparison()
        self.plot_hyperparameter_sensitivity()
        self.plot_per_class_improvement()
        self.generate_latex_table()
        
        print(f"\n✓ All figures saved to {self.output_dir}")
        print("="*60)


# Usage
if __name__ == "__main__":
    visualizer = TTAVisualizer()
    visualizer.generate_all_figures()
