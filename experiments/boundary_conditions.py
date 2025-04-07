# experiments/boundary_conditions.py

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd
import warnings
import seaborn as sns

# Ensure project root directory is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.triangle_visualizer import TriangleVisualizer
from utils.market_generator import MarketGenerator
from algorithms.market import Market


class BoundaryConditionsExperiment:
    """Study regulatory behavior under special distributions and boundary conditions"""

    def __init__(self):
        self.market_generator = MarketGenerator()

    def run_experiment(self, save_dir=None):
        """
        Run boundary conditions experiment

        Parameters:
        save_dir: Directory to save results

        Returns:
        Dictionary of experiment results
        """
        results = {}

        # 1. Extreme binary distribution (highly imbalanced)
        binary_results = self._analyze_extreme_binary(save_dir)
        results["extreme_binary"] = binary_results

        # 2. Degenerate distribution (single value point)
        degenerate_results = self._analyze_degenerate_markets(save_dir)
        results["degenerate"] = degenerate_results

        # 3. Multimodal distribution (market segmentation)
        multimodal_results = self._analyze_multimodal_markets(save_dir)
        results["multimodal"] = multimodal_results

        # 4. Extreme F conditions (F interval nearly degenerate or full domain)
        extreme_f_results = self._analyze_extreme_f_conditions(save_dir)
        results["extreme_f"] = extreme_f_results

        # 5. Special price points (such as sparse price distribution)
        sparse_price_results = self._analyze_sparse_price_points(save_dir)
        results["sparse_price"] = sparse_price_results

        return results

    def _analyze_extreme_binary(self, save_dir):
        """Analyze extreme binary distributions - very high or low p values"""
        print("\n===== Extreme Binary Distribution Analysis =====")

        results = []

        # Extreme p values
        p_values = [0.001, 0.01, 0.99, 0.999]

        # Fixed parameters
        low, high = 1, 10

        # F value selection
        F_values = [
            [low, high],  # Full range
            [5, 5],  # Single point F
            [low + 0.001, high - 0.001]  # Almost full range
        ]

        for p in p_values:
            print(f"\n--- p = {p:.6f} ---")

            # Generate market
            values, masses = self.market_generator.binary(p=p, low=low, high=high)

            # Create market and visualizer
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio, avoid division by zero
                ratio = passive_area / active_area if active_area > 0 else float('inf')
                ratio_str = f"{ratio:.6f}" if ratio != float('inf') else "Inf"

                print(f"F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.6f}, Active: {active_area:.6f}, "
                      f"Ratio: {ratio_str}")

                results.append({
                    "p": p,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio if ratio != float('inf') else 1000
                })

            # Create and save triangle visualization
            if save_dir:
                save_path = Path(save_dir) / f"extreme_binary_p{p:.6f}.png"
                fig, ax = plt.subplots(figsize=(10, 8))
                visualizer.draw_triangles(F_values[0], ax=ax, fixed_axes=True)
                ax.set_title(f"Extreme Binary Distribution (p={p:.6f})")
                plt.savefig(save_path, dpi=300)
                plt.close()

        return results

    def _analyze_degenerate_markets(self, save_dir):
        """Analyze degenerate distributions - market with single value point or very few value points"""
        print("\n===== Degenerate Distribution Analysis =====")

        results = []

        # Test scenarios
        scenarios = [
            {"name": "Single Value Point", "values": [5.0], "masses": [1.0]},
            {"name": "Two Close Value Points", "values": [4.99, 5.01], "masses": [0.5, 0.5]},
            {"name": "Extremely Uneven Distribution", "values": [1.0, 10.0], "masses": [0.9999, 0.0001]}
        ]

        # F value selection
        F_sets = [
            [[5, 5]],  # Exactly the single value
            [[4.99, 5.01]],  # Exactly covers two value points
            [[1, 10]]  # Full range
        ]

        for i, scenario in enumerate(scenarios):
            name = scenario["name"]
            values = scenario["values"]
            masses = scenario["masses"]
            F_values = F_sets[i] if i < len(F_sets) else [[min(values), max(values)]]

            print(f"\n--- {name} ---")
            print(f"  Values: {values}")
            print(f"  Masses: {masses}")

            # Create market and visualizer
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio, avoid division by zero
                ratio = passive_area / active_area if active_area > 0 else float('inf')
                ratio_str = f"{ratio:.6f}" if ratio != float('inf') else "Inf"

                print(f"F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.6f}, Active: {active_area:.6f}, "
                      f"Ratio: {ratio_str}")

                results.append({
                    "scenario": name,
                    "values": values,
                    "masses": masses,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio if ratio != float('inf') else 1000
                })

            # Create and save triangle visualization
            if save_dir:
                save_path = Path(save_dir) / f"degenerate_market_{i + 1}.png"
                fig, ax = plt.subplots(figsize=(10, 8))
                visualizer.draw_triangles(F_values[0], ax=ax, fixed_axes=True)
                ax.set_title(f"Degenerate Market: {name}")
                plt.savefig(save_path, dpi=300)
                plt.close()

        return results

    def _analyze_multimodal_markets(self, save_dir):
        """Analyze multimodal distributions - such as mixture of normal distributions"""
        print("\n===== Multimodal Distribution Analysis =====")

        results = []

        # Test scenarios - parameters for mixed distributions
        scenarios = [
            # Bimodal, two equal peaks
            {"name": "Symmetric Bimodal",
             "means": [3, 7],
             "stds": [0.5, 0.5],
             "weights": [0.5, 0.5]},

            # Bimodal, one dominant peak
            {"name": "Asymmetric Bimodal",
             "means": [3, 7],
             "stds": [0.5, 0.5],
             "weights": [0.8, 0.2]},

            # Trimodal, uniform distribution
            {"name": "Trimodal Pattern",
             "means": [2, 5, 8],
             "stds": [0.5, 0.5, 0.5],
             "weights": [0.33, 0.34, 0.33]}
        ]

        # Fixed parameters
        n_points = 7  # Points per component
        low, high = 1, 10

        # F value selection strategy
        F_strategies = [
            lambda means, range_min, range_max: [means[0], means[0]],  # Single point F at first peak
            lambda means, range_min, range_max: [means[-1], means[-1]],  # Single point F at last peak
            lambda means, range_min, range_max: [means[0], means[-1]],  # Range between peaks
            lambda means, range_min, range_max: [range_min, range_max]  # Full range
        ]

        for scenario in scenarios:
            name = scenario["name"]
            means = scenario["means"]
            stds = scenario["stds"]
            weights = scenario["weights"]

            print(f"\n--- {name} ---")
            print(f"  Means: {means}")
            print(f"  Weights: {weights}")

            # Generate mixture distribution
            values, masses = self._generate_mixture_distribution(
                means, stds, weights, n_points, low, high)

            # Create market and visualizer
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            # Calculate F values
            F_values = [F_func(means, low, high) for F_func in F_strategies]

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio, avoid division by zero
                ratio = passive_area / active_area if active_area > 0 else float('inf')
                ratio_str = f"{ratio:.6f}" if ratio != float('inf') else "Inf"

                print(f"F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.6f}, Active: {active_area:.6f}, "
                      f"Ratio: {ratio_str}")

                results.append({
                    "scenario": name,
                    "means": means,
                    "weights": weights,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio if ratio != float('inf') else 1000
                })

            # Create and save triangle visualization
            if save_dir:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()

                for i, (F, ax) in enumerate(zip(F_values, axes)):
                    visualizer.draw_triangles(F, ax=ax, fixed_axes=True)
                    F_desc = "Single Point" if F[0] == F[1] else "Range"
                    ax.set_title(f"F={F} ({F_desc})")

                fig.suptitle(f"Multimodal Distribution: {name}", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                save_path = Path(save_dir) / f"multimodal_{name.replace(' ', '_')}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()

                # Also save distribution plot
                self._plot_distribution(values, masses, name, save_dir)

        return results

    def _generate_mixture_distribution(self, means, stds, weights, n_points, low, high):
        """Generate mixture distribution"""
        from scipy.stats import truncnorm

        values = []
        masses = []

        for mean, std, weight in zip(means, stds, weights):
            # For each component, generate truncated normal distribution
            a, b = (low - mean) / std, (high - mean) / std
            component_values = truncnorm.ppf(
                np.linspace(0.01, 0.99, n_points), a, b, loc=mean, scale=std)
            component_masses = np.ones(n_points) * weight / n_points

            values.extend(component_values)
            masses.extend(component_masses)

        # Sort and merge possible duplicate values
        sorted_indices = np.argsort(values)
        values = np.array(values)[sorted_indices]
        masses = np.array(masses)[sorted_indices]

        # Merge very close values
        epsilon = 1e-6
        i = 0
        while i < len(values) - 1:
            if abs(values[i + 1] - values[i]) < epsilon:
                masses[i] += masses[i + 1]
                values = np.delete(values, i + 1)
                masses = np.delete(masses, i + 1)
            else:
                i += 1

        return values.tolist(), masses.tolist()

    def _plot_distribution(self, values, masses, name, save_dir):
        """Plot distribution"""
        plt.figure(figsize=(10, 6))
        plt.bar(values, masses, width=0.2)
        plt.title(f"Distribution: {name}")
        plt.xlabel("Value")
        plt.ylabel("Mass")
        plt.grid(alpha=0.3)

        save_path = Path(save_dir) / f"distribution_{name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _analyze_extreme_f_conditions(self, save_dir):
        """Analyze extreme F conditions"""
        print("\n===== Extreme F Conditions Analysis =====")

        results = []

        # Uniform distribution as baseline
        values, masses = self.market_generator.uniform(n=5, low=1, high=10)
        visualizer = TriangleVisualizer(np.array(masses), np.array(values))

        # Extreme F conditions
        extreme_F_values = [
            # Special single points
            [1, 1],  # Lowest value point
            [10, 10],  # Highest value point
            [5, 5],  # Middle value point

            # Almost single point
            [4.999, 5.001],  # Nearly single point

            # Very narrow range
            [4.9, 5.1],  # Very narrow range

            # Extremely wide range
            [1.001, 9.999],  # Almost full range

            # Standard range (reference)
            [1, 10]  # Full range
        ]

        print("\n--- Extreme F Conditions with Uniform Distribution ---")

        for F in extreme_F_values:
            is_feasible = visualizer.check_F_feasibility(F)
            features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

            passive_area = features["passive_intermediary"]["area"]
            active_area = features["active_intermediary"]["area"]

            # Calculate ratio, avoid division by zero
            ratio = passive_area / active_area if active_area > 0 else float('inf')
            ratio_str = f"{ratio:.6f}" if ratio != float('inf') else "Inf"

            F_width = F[1] - F[0]
            F_type = "Single Point" if F_width == 0 else \
                "Almost Single" if F_width < 0.01 else \
                    "Very Narrow" if F_width < 0.5 else \
                        "Almost Full Range" if F_width > 8.9 else \
                            "Standard Range"

            print(f"F={F} ({F_type}), Feasible: {is_feasible}, "
                  f"Passive: {passive_area:.6f}, Active: {active_area:.6f}, "
                  f"Ratio: {ratio_str}")

            results.append({
                "F": F,
                "F_type": F_type,
                "F_width": F_width,
                "is_feasible": is_feasible,
                "passive_area": passive_area,
                "active_area": active_area,
                "area_ratio": ratio if ratio != float('inf') else 1000
            })

        # Create and save F width vs triangle area plot
        if save_dir:
            self._plot_f_width_impact(results, save_dir)

        return results

    def _plot_f_width_impact(self, results, save_dir):
        """Plot impact of F width on triangle areas"""
        # Prepare data
        F_widths = [r["F_width"] for r in results]
        passive_areas = [r["passive_area"] for r in results]
        active_areas = [r["active_area"] for r in results]
        is_feasible = [r["is_feasible"] for r in results]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Feasible F
        feasible_indices = [i for i, f in enumerate(is_feasible) if f]
        if feasible_indices:
            ax.scatter([F_widths[i] for i in feasible_indices],
                       [passive_areas[i] for i in feasible_indices],
                       c='r', marker='o', s=100, label="Passive (Feasible)")
            ax.scatter([F_widths[i] for i in feasible_indices],
                       [active_areas[i] for i in feasible_indices],
                       c='b', marker='o', s=100, label="Active (Feasible)")

        # Infeasible F
        infeasible_indices = [i for i, f in enumerate(is_feasible) if not f]
        if infeasible_indices:
            ax.scatter([F_widths[i] for i in infeasible_indices],
                       [passive_areas[i] for i in infeasible_indices],
                       c='r', marker='x', s=100, label="Passive (Not Feasible)")
            ax.scatter([F_widths[i] for i in infeasible_indices],
                       [active_areas[i] for i in infeasible_indices],
                       c='b', marker='x', s=100, label="Active (Not Feasible)")

        # Add labels
        for i, r in enumerate(results):
            ax.annotate(f"{r['F']}", (F_widths[i], max(passive_areas[i], active_areas[i]) + 0.1),
                        ha='center')

        # Use log scale for better display of width variations
        ax.set_xscale('symlog', linthresh=0.1)

        ax.set_title("Impact of F Width on Triangle Areas")
        ax.set_xlabel("F Width (log scale)")
        ax.set_ylabel("Triangle Area")
        ax.legend()
        ax.grid(alpha=0.3)

        save_path = Path(save_dir) / "f_width_impact.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _analyze_sparse_price_points(self, save_dir):
        """Analyze impact of sparse price points on triangle characteristics"""
        print("\n===== Sparse Price Points Analysis =====")

        results = []

        # Test scenarios
        scenarios = [
            {"name": "Uniform 2 Points", "n": 2},  # Only two value points
            {"name": "Uniform 3 Points", "n": 3},  # Only three value points
            {"name": "Uniform 5 Points", "n": 5},  # Standard 5 points
            {"name": "Uniform 10 Points", "n": 10}  # High precision 10 points
        ]

        # Fixed parameters
        low, high = 1, 10

        # F value selection
        F_values = [
            [5, 5],  # Single point
            [3, 7],  # Medium range
            [1, 10]  # Full range
        ]

        for scenario in scenarios:
            name = scenario["name"]
            n_points = scenario["n"]

            print(f"\n--- {name} (n={n_points}) ---")

            # Generate uniform distribution
            values, masses = self.market_generator.uniform(n=n_points, low=low, high=high)

            # Create market and visualizer
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio, avoid division by zero
                ratio = passive_area / active_area if active_area > 0 else float('inf')
                ratio_str = f"{ratio:.6f}" if ratio != float('inf') else "Inf"

                print(f"F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.6f}, Active: {active_area:.6f}, "
                      f"Ratio: {ratio_str}")

                results.append({
                    "scenario": name,
                    "n_points": n_points,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio if ratio != float('inf') else 1000
                })

            # Create and save triangle visualization
            if save_dir:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                for i, (F, ax) in enumerate(zip(F_values, axes)):
                    visualizer.draw_triangles(F, ax=ax, fixed_axes=True)
                    ax.set_title(f"F={F}")

                fig.suptitle(f"Sparse Price Points: {name} (n={n_points})", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                save_path = Path(save_dir) / f"sparse_points_{n_points}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()

                # Also save distribution plot
                self._plot_sparse_distribution(values, masses, name, save_dir)

        # Create and save point count vs triangle area plot
        if save_dir:
            self._plot_point_density_impact(results, save_dir)

        return results

    def _plot_sparse_distribution(self, values, masses, name, save_dir):
        """Plot sparse distribution"""
        plt.figure(figsize=(10, 6))

        # Compatible with different matplotlib versions
        try:
            plt.stem(values, masses, use_line_collection=True)
        except TypeError:
            plt.stem(values, masses)  # Older version compatibility

        plt.title(f"Sparse Distribution: {name}")
        plt.xlabel("Value")
        plt.ylabel("Mass")
        plt.grid(alpha=0.3)
        plt.ylim(0, max(masses) * 1.1)

        save_path = Path(save_dir) / f"sparse_distribution_{name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_point_density_impact(self, results, save_dir):
        """Plot impact of point density on triangle areas"""
        # Group by F value
        F_grouped_results = {}
        for r in results:
            F_str = str(r["F"])
            if F_str not in F_grouped_results:
                F_grouped_results[F_str] = []
            F_grouped_results[F_str].append(r)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        for F_str, F_results in F_grouped_results.items():
            n_points = [r["n_points"] for r in F_results]
            passive_areas = [r["passive_area"] for r in F_results]
            active_areas = [r["active_area"] for r in F_results]

            # Connect lines
            ax.plot(n_points, passive_areas, 'ro-', label=f"Passive (F={F_str})")
            ax.plot(n_points, active_areas, 'bo-', label=f"Active (F={F_str})")

        ax.set_title("Impact of Price Point Density on Triangle Areas")
        ax.set_xlabel("Number of Price Points")
        ax.set_ylabel("Triangle Area")
        ax.legend()
        ax.grid(alpha=0.3)

        # Set x-axis to integer ticks
        ax.set_xticks(sorted(list(set([r["n_points"] for r in results]))))

        save_path = Path(save_dir) / "point_density_impact.png"
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    # run_boundary_conditions.py

    from pathlib import Path
    from experiments.boundary_conditions import BoundaryConditionsExperiment

    # Create results directory
    result_dir = Path("results/boundary_conditions")
    result_dir.mkdir(exist_ok=True, parents=True)

    # Run boundary conditions experiment
    experiment = BoundaryConditionsExperiment()
    results = experiment.run_experiment(save_dir=result_dir)

    print("Boundary conditions experiment completed! Results saved in:", result_dir)