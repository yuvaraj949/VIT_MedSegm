"""
Standalone TTA Results Plotter
Generates all figures from saved results - NO EXPERIMENTS NEEDED
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP warning

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TTAResultsPlotter:
    """Generate all TTA figures from saved results"""
    
    def __init__(self, results_path='tta_results/all_results.json'):
        print("\n" + "="*70)
        print("TTA RESULTS VISUALIZATION")
        print("="*70)
        
        # Load results
        print(f"\nLoading results from: {results_path}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"✓ Loaded {len(self.results)} experimental results")
        
        # Create output directory
        self.output_dir = Path('tta_figures')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14
    
    def plot_method_comparison(self):
        """Figure 1: Bar chart comparing all TTA methods"""
        print("\n[1/5] Generating method comparison plot...")
        
        methods = []
        dice_scores = []
        dice_stds = []
        colors = []
        
        for method, data in self.results.items():
            # Clean method name
            clean_name = method.replace('_', ' ').replace('TTA ', 'TTA\n')
            methods.append(clean_name)
            dice_scores.append(data['dice_mean'])
            dice_stds.append(data['dice_std'])
            
            # Color coding
            if 'baseline' in method.lower():
                colors.append('#e74c3c')  # Red for baseline
            elif 'BN Entropy' in method or 'Combined' in method:
                colors.append('#27ae60')  # Green for combined
            else:
                colors.append('#3498db')  # Blue for others
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(methods))
        bars = ax.bar(x, dice_scores, yerr=dice_stds, capsize=5, 
                      color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Highlight best performer
        best_idx = np.argmax(dice_scores)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Labels and styling
        ax.set_xlabel('Method', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Dice Coefficient (%)', fontsize=13, fontweight='bold')
        ax.set_title('Test-Time Adaptation: Performance Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([min(dice_scores) - 5, max(dice_scores) + 8])
        
        # Add value labels on bars
        for i, (bar, score, std) in enumerate(zip(bars, dice_scores, dice_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add improvement annotation for best
        baseline_dice = self.results['baseline']['dice_mean']
        best_dice = dice_scores[best_idx]
        improvement = best_dice - baseline_dice
        
        ax.annotate(f'Best: +{improvement:.2f}%', 
                   xy=(best_idx, best_dice), 
                   xytext=(best_idx, best_dice + 6),
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gold'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tta_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tta_method_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {self.output_dir / 'tta_method_comparison.png'}")
        print(f"  ✓ Best improvement: +{improvement:.2f}%")
    
    def plot_hyperparameter_sensitivity(self):
        """Figure 2: Learning rate and steps sensitivity"""
        print("\n[2/5] Generating hyperparameter sensitivity plots...")
        
        # Extract data for combined method
        lr_data = {}
        steps_data = {}
        
        for method, data in self.results.items():
            if 'BN Entropy' in method or 'Combined' in method:
                if 'lr' in data:
                    lr = data['lr']
                    lr_data[lr] = data['dice_mean']
                if 'adapt_steps' in data:
                    steps = data['adapt_steps']
                    # Only include if LR is default (1e-4)
                    if data.get('lr', 1e-4) == 1e-4:
                        steps_data[steps] = data['dice_mean']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Learning rate sensitivity
        if lr_data:
            lrs = sorted(lr_data.keys())
            scores = [lr_data[lr] for lr in lrs]
            
            axes[0].plot(lrs, scores, marker='o', markersize=12, linewidth=3, 
                        color='steelblue', label='Dice Score')
            axes[0].fill_between(lrs, [s-2 for s in scores], [s+2 for s in scores], 
                                alpha=0.2, color='steelblue')
            axes[0].set_xscale('log')
            axes[0].set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Mean Dice (%)', fontsize=12, fontweight='bold')
            axes[0].set_title('Learning Rate Sensitivity', fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].legend()
            
            # Mark optimal point
            best_lr = max(lr_data.items(), key=lambda x: x[1])
            axes[0].scatter([best_lr[0]], [best_lr[1]], s=200, c='red', 
                           marker='*', zorder=5, label='Optimal')
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'No LR sensitivity data', 
                        ha='center', va='center', transform=axes[0].transAxes)
        
        # Adaptation steps sensitivity
        if steps_data:
            steps = sorted(steps_data.keys())
            scores = [steps_data[s] for s in steps]
            
            axes[1].plot(steps, scores, marker='s', markersize=12, linewidth=3, 
                        color='coral', label='Dice Score')
            axes[1].fill_between(steps, [s-2 for s in scores], [s+2 for s in scores], 
                                alpha=0.2, color='coral')
            axes[1].set_xlabel('Adaptation Steps', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Mean Dice (%)', fontsize=12, fontweight='bold')
            axes[1].set_title('Adaptation Steps Sensitivity', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend()
            
            # Mark optimal point
            best_steps = max(steps_data.items(), key=lambda x: x[1])
            axes[1].scatter([best_steps[0]], [best_steps[1]], s=200, c='red', 
                           marker='*', zorder=5, label='Optimal')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No steps sensitivity data', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tta_hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tta_hyperparameter_sensitivity.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {self.output_dir / 'tta_hyperparameter_sensitivity.png'}")
    
    def plot_per_class_improvement(self):
        """Figure 3: Per-class Dice improvement with TTA"""
        print("\n[3/5] Generating per-class comparison...")
        
        baseline_dice = np.array(self.results['baseline']['dice_per_class'])
        
        # Find best TTA method
        best_method = max(
            [(k, v) for k, v in self.results.items() if k != 'baseline'],
            key=lambda x: x[1]['dice_mean']
        )
        tta_dice = np.array(best_method[1]['dice_per_class'])
        
        improvement = tta_dice - baseline_dice
        
        # Organ names
        organs = ['BG', 'Aorta', 'Gallbl.', 'Kid(L)', 'Kid(R)',
                 'Liver', 'Pancr.', 'Spleen', 'Stom.', 'C9', 'C10',
                 'C11', 'C12', 'C13']
        
        # Ensure arrays match
        n_organs = min(len(organs), len(baseline_dice), len(tta_dice))
        organs = organs[:n_organs]
        baseline_dice = baseline_dice[:n_organs]
        tta_dice = tta_dice[:n_organs]
        improvement = improvement[:n_organs]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(organs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_dice, width, label='Baseline', 
                      alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, tta_dice, width, label=f'TTA (Best)',
                      alpha=0.8, color='#27ae60', edgecolor='black', linewidth=1)
        
        # Add improvement labels
        for i, (imp, x_pos) in enumerate(zip(improvement, x)):
            if abs(imp) > 1.0:  # Only show significant improvements
                ax.text(x_pos, max(baseline_dice[i], tta_dice[i]) + 2, 
                       f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color='green' if imp > 0 else 'red')
        
        ax.set_xlabel('Organ Class', fontsize=13, fontweight='bold')
        ax.set_ylabel('Dice Coefficient (%)', fontsize=13, fontweight='bold')
        ax.set_title('Per-Class Performance: Baseline vs TTA', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(organs, rotation=45, ha='right')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, max(max(baseline_dice), max(tta_dice)) + 10])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tta_per_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'tta_per_class_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {self.output_dir / 'tta_per_class_comparison.png'}")
        
        # Print top improvements
        print("\n  Top 5 improvements:")
        sorted_improvements = sorted(enumerate(improvement), key=lambda x: x[1], reverse=True)
        for i, (idx, imp) in enumerate(sorted_improvements[:5]):
            if idx < len(organs):
                print(f"    {i+1}. {organs[idx]:10s}: +{imp:.2f}%")
    
    def generate_latex_table(self):
        """Figure 4: Generate LaTeX table"""
        print("\n[4/5] Generating LaTeX table...")
        
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
        
        # Sort by dice score
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['dice_mean'], 
                               reverse=True)
        
        for method, data in sorted_results:
            dice = data['dice_mean']
            dice_std = data.get('dice_std', 0)
            iou = data.get('iou_mean', 0)
            iou_std = data.get('iou_std', 0)
            hd95 = data.get('hd95_mean', 0)
            hd95_std = data.get('hd95_std', 0)
            delta = dice - baseline_dice
            
            # Clean method name for LaTeX
            method_name = method.replace('_', ' ')
            if len(method_name) > 30:
                method_name = method_name[:27] + '...'
            
            # Bold the best non-baseline
            if delta > 0 and delta == max([v['dice_mean'] - baseline_dice 
                                          for k, v in self.results.items() 
                                          if k != 'baseline']):
                method_name = f"\\textbf{{{method_name}}}"
                dice_str = f"\\textbf{{{dice:.2f}$\\pm${dice_std:.2f}}}"
                delta_str = f"\\textbf{{{delta:+.2f}}}"
            else:
                dice_str = f"{dice:.2f}$\\pm${dice_std:.2f}"
                delta_str = f"{delta:+.2f}" if delta != 0 else "--"
            
            latex += f"{method_name} & {dice_str} & {iou:.2f}$\\pm${iou_std:.2f} & "
            latex += f"{hd95:.2f}$\\pm${hd95_std:.2f} & {delta_str} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save to file
        with open(self.output_dir / 'tta_latex_table.tex', 'w') as f:
            f.write(latex)
        
        print(f"  ✓ Saved: {self.output_dir / 'tta_latex_table.tex'}")
    
    def generate_summary_report(self):
        """Figure 5: Generate text summary"""
        print("\n[5/5] Generating summary report...")
        
        baseline_dice = self.results['baseline']['dice_mean']
        
        # Find best method
        best_method = max(
            [(k, v) for k, v in self.results.items() if k != 'baseline'],
            key=lambda x: x[1]['dice_mean']
        )
        
        best_name = best_method[0]
        best_data = best_method[1]
        improvement = best_data['dice_mean'] - baseline_dice
        
        summary = f"""
    {'='*70}
    TTA EXPERIMENT SUMMARY REPORT
    {'='*70}

    BASELINE PERFORMANCE
    • Dice:  {self.results['baseline']['dice_mean']:.2f}% ± {self.results['baseline']['dice_std']:.2f}%
    • IoU:   {self.results['baseline'].get('iou_mean', 0):.2f}% ± {self.results['baseline'].get('iou_std', 0):.2f}%
    • HD95:  {self.results['baseline'].get('hd95_mean', 0):.2f} ± {self.results['baseline'].get('hd95_std', 0):.2f} mm

    BEST TTA METHOD: {best_name}
    • Dice:  {best_data['dice_mean']:.2f}% ± {best_data['dice_std']:.2f}%
    • IoU:   {best_data.get('iou_mean', 0):.2f}% ± {best_data.get('iou_std', 0):.2f}%
    • HD95:  {best_data.get('hd95_mean', 0):.2f} ± {best_data.get('hd95_std', 0):.2f} mm
    
    IMPROVEMENT
    • Dice:  +{improvement:.2f}% (absolute)
    • Relative: +{(improvement/baseline_dice)*100:.2f}%

    CONFIGURATION
    • Learning Rate: {best_data.get('lr', 'N/A')}
    • Adapt Steps: {best_data.get('adapt_steps', 'N/A')}
    • Method: {best_data.get('tta_method', 'N/A')}

    ALL METHODS RANKED BY DICE:
    """
        
        sorted_methods = sorted(self.results.items(), 
                            key=lambda x: x[1]['dice_mean'], 
                            reverse=True)
        
        for i, (method, data) in enumerate(sorted_methods, 1):
            dice = data['dice_mean']
            delta = dice - baseline_dice
            summary += f"  {i}. {method:40s} {dice:6.2f}% (Delta {delta:+.2f}%)\n"
        
        summary += f"\n{'='*70}\n"
        summary += "OUTPUTS GENERATED:\n"
        summary += f"  • {self.output_dir / 'tta_method_comparison.png'}\n"
        summary += f"  • {self.output_dir / 'tta_hyperparameter_sensitivity.png'}\n"
        summary += f"  • {self.output_dir / 'tta_per_class_comparison.png'}\n"
        summary += f"  • {self.output_dir / 'tta_latex_table.tex'}\n"
        summary += f"  • {self.output_dir / 'tta_summary.txt'}\n"
        summary += f"{'='*70}\n"
        
        # Save to file with UTF-8 encoding (FIXED)
        with open(self.output_dir / 'tta_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Print to console
        print(summary)
        
        print(f"  ✓ Saved: {self.output_dir / 'tta_summary.txt'}")

    
    def generate_all(self):
        """Generate all figures and reports"""
        try:
            self.plot_method_comparison()
            self.plot_hyperparameter_sensitivity()
            self.plot_per_class_improvement()
            self.generate_latex_table()
            self.generate_summary_report()
            
            print("\n" + "="*70)
            print("✓ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
            print("\nNext steps:")
            print("  1. Check tta_summary.txt for detailed results")
            print("  2. Use PNG files in your presentation")
            print("  3. Use PDF files in your paper (vector graphics)")
            print("  4. Copy tta_latex_table.tex into your paper")
            
        except Exception as e:
            print(f"\n❌ ERROR during visualization: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Check if results file exists
    results_file = 'tta_results/all_results.json'
    
    if not Path(results_file).exists():
        print(f"❌ ERROR: Results file not found: {results_file}")
        print("\nPlease run the experiments first:")
        print("  python run_tta_experiments.py")
        exit(1)
    
    # Create plotter and generate all visualizations
    plotter = TTAResultsPlotter(results_file)
    plotter.generate_all()
