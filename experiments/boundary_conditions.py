# experiments/boundary_conditions.py (修改版)

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 确保项目根目录在路径中
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

        # 1. Extreme binary distribution
        binary_results = self._analyze_extreme_binary(save_dir)
        results["extreme_binary"] = binary_results

        # 2. Degenerate distribution
        degenerate_results = self._analyze_degenerate_markets(save_dir)
        results["degenerate"] = degenerate_results

        # 3. Extreme F conditions
        extreme_f_results = self._analyze_extreme_f_conditions(save_dir)
        results["extreme_f"] = extreme_f_results

        # 4. Sparse price points
        sparse_results = self._analyze_sparse_price_points(save_dir)
        results["sparse_price"] = sparse_results

        return results

    def _analyze_extreme_binary(self, save_dir):
        """Analyze extreme binary distributions with highly imbalanced probabilities"""
        print("\n===== Extreme Binary Distribution Analysis =====")

        results = []
        # Extreme p values
        p_values = [0.001, 0.01, 0.99, 0.999]
        low, high = 1, 10
        F_values = [
            [5, 5],  # Single point F
            [1, 10]  # Full range F
        ]

        for p in p_values:
            print(f"\n--- p = {p:.3f} ---")
            values, masses = self.market_generator.binary(p=p, low=low, high=high)
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            # Calculate uniform optimal price
            market = Market(values, masses)
            optimal_prices = market.optimal_price()
            uniform_price = optimal_prices[0] if optimal_prices else None
            print(f"  Uniform optimal price: {uniform_price}")

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio safely
                ratio = passive_area / active_area if active_area > 0 else 999.99
                if passive_area < 1e-6 and active_area < 1e-6:
                    ratio = 0  # Both areas effectively zero

                print(f"  F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio:.4f}" + (" (approx)" if not is_feasible else ""))

                results.append({
                    "p": p,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio if ratio < 999 else 999.99
                })

            # Create and save triangle visualization
            if save_dir:
                save_path = Path(save_dir) / f"extreme_binary_p{p:.3f}.svg"
                fig, ax = plt.subplots(figsize=(8, 6))
                visualizer.draw_triangles(F=[1, 10], ax=ax)
                ax.set_title(f"Extreme Binary Distribution (p={p:.3f})")
                plt.savefig(save_path, dpi=300)
                plt.close()

        return results

    def _analyze_degenerate_markets(self, save_dir):
        """Analyze degenerate distributions with single value point or very few values"""
        print("\n===== Degenerate Market Analysis =====")

        results = []
        # Test scenarios
        scenarios = [
            {"name": "Single Value Point", "values": [5.0], "masses": [1.0]},
            {"name": "Two Close Points", "values": [4.99, 5.01], "masses": [0.5, 0.5]},
            {"name": "Extreme Skew", "values": [1.0, 10.0], "masses": [0.999, 0.001]}
        ]

        for scenario in scenarios:
            name = scenario["name"]
            values = scenario["values"]
            masses = scenario["masses"]
            print(f"\n--- {name} ---")
            print(f"  Values: {values}")
            print(f"  Masses: {masses}")

            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            # Test single point F and full range F
            F_values = [
                [5, 5],  # Single point F
                [min(values), max(values)]  # Full range F
            ]

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio safely
                ratio = passive_area / active_area if active_area > 0 else 999.99
                if passive_area < 1e-6 and active_area < 1e-6:
                    ratio = 0  # Both areas effectively zero

                print(f"  F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio:.4f}" + (" (approx)" if not is_feasible else ""))

                results.append({
                    "scenario": name,
                    "values": values,
                    "masses": masses,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": min(ratio, 999.99)
                })

            # Create and save triangle visualization
            if save_dir:
                save_path = Path(save_dir) / f"degenerate_{name.replace(' ', '_')}.svg"
                fig, ax = plt.subplots(figsize=(8, 6))
                visualizer.draw_triangles(F=F_values[1], ax=ax)
                ax.set_title(f"Degenerate Market: {name}")
                plt.savefig(save_path, dpi=300)
                plt.close()

        return results

    def _analyze_extreme_f_conditions(self, save_dir):
        """Analyze extreme F conditions"""
        print("\n===== Extreme F Conditions Analysis =====")

        results = []
        # Uniform distribution as baseline
        values, masses = self.market_generator.uniform(n=5, low=1, high=10)
        visualizer = TriangleVisualizer(np.array(masses), np.array(values))

        # Extreme F values
        extreme_F_values = [
            [1, 1],  # Lowest value point
            [10, 10],  # Highest value point
            [5, 5],  # Middle value point
            [4.9, 5.1],  # Very narrow range
            [1, 10]  # Full range
        ]

        print("\n--- Uniform Distribution with Extreme F Values ---")

        # Calculate uniform optimal price
        market = Market(values, masses)
        optimal_prices = market.optimal_price()
        uniform_price = optimal_prices[0] if optimal_prices else None
        print(f"  Uniform optimal price: {uniform_price}")

        for F in extreme_F_values:
            is_feasible = visualizer.check_F_feasibility(F)
            features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

            passive_area = features["passive_intermediary"]["area"]
            active_area = features["active_intermediary"]["area"]

            # Calculate ratio safely
            ratio = passive_area / active_area if active_area > 0 else 999.99
            if passive_area < 1e-6 and active_area < 1e-6:
                ratio = 0  # Both areas effectively zero

            F_width = F[1] - F[0]
            F_type = "Single Point" if F_width == 0 else \
                "Narrow Range" if F_width < 1 else \
                    "Full Range"

            print(f"  F={F} ({F_type}), Feasible: {is_feasible}, "
                  f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                  f"Ratio: {ratio:.4f}" + (" (approx)" if not is_feasible else ""))

            results.append({
                "F": F,
                "F_type": F_type,
                "F_width": F_width,
                "is_feasible": is_feasible,
                "passive_area": passive_area,
                "active_area": active_area,
                "area_ratio": min(ratio, 999.99)
            })

        # Create and save visualization
        if save_dir:
            # Plot F width vs triangle area
            plt.figure(figsize=(10, 6))
            F_widths = [r["F_width"] for r in results]
            passive_areas = [r["passive_area"] for r in results]
            active_areas = [r["active_area"] for r in results]

            plt.plot(F_widths, passive_areas, 'ro-', label="Passive Intermediary")
            plt.plot(F_widths, active_areas, 'bo-', label="Active Intermediary")

            plt.title("Impact of F Width on Triangle Areas")
            plt.xlabel("F Width")
            plt.ylabel("Area")
            plt.legend()
            plt.grid(alpha=0.3)

            # Use symlog scale for better visibility
            plt.xscale('symlog', linthresh=0.1)

            # Add F value annotations
            for i, r in enumerate(results):
                plt.annotate(f"F={r['F']}",
                             (F_widths[i], max(passive_areas[i], active_areas[i]) + 0.02),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center')

            save_path = Path(save_dir) / "f_width_impact.svg"
            plt.savefig(save_path, dpi=300)
            plt.close()

        return results

    def _analyze_sparse_price_points(self, save_dir):
        """Analyze impact of sparse price points on triangle characteristics"""
        print("\n===== Sparse Price Points Analysis =====")

        results = []
        # Test scenarios
        n_values = [2, 3, 5, 10]  # Price point counts
        low, high = 1, 10

        F_values = [
            [5, 5],  # Single point
            [1, 10]  # Full range
        ]

        for n in n_values:
            print(f"\n--- Uniform Distribution (n={n}) ---")
            # Generate uniform distribution
            values, masses = self.market_generator.uniform(n=n, low=low, high=high)
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            # Calculate uniform optimal price
            market = Market(values, masses)
            optimal_prices = market.optimal_price()
            uniform_price = optimal_prices[0] if optimal_prices else None
            print(f"  Uniform optimal price: {uniform_price}")

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # Calculate ratio safely
                ratio = passive_area / active_area if active_area > 0 else 999.99
                if passive_area < 1e-6 and active_area < 1e-6:
                    ratio = 0  # Both areas effectively zero

                print(f"  F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio:.4f}" + (" (approx)" if not is_feasible else ""))

                results.append({
                    "n_points": n,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": min(ratio, 999.99)
                })

            # Create and save visualization
            if save_dir:
                save_path = Path(save_dir) / f"sparse_points_n{n}.svg"
                fig, ax = plt.subplots(figsize=(8, 6))
                visualizer.draw_triangles(F=[1, 10], ax=ax)
                ax.set_title(f"Sparse Price Points (n={n})")
                plt.savefig(save_path, dpi=300)
                plt.close()

        # Create summary plot
        if save_dir and results:
            # Plot impact of price point density
            plt.figure(figsize=(10, 6))

            # Group data by F value
            single_point_data = [(r["n_points"], r["passive_area"], r["active_area"])
                                 for r in results if str(r["F"]) == "[5, 5]"]
            full_range_data = [(r["n_points"], r["passive_area"], r["active_area"])
                               for r in results if str(r["F"]) == "[1, 10]"]

            # Sort by n_points
            single_point_data.sort(key=lambda x: x[0])
            full_range_data.sort(key=lambda x: x[0])

            # Plot single point F data
            if single_point_data:
                n_values = [d[0] for d in single_point_data]
                p_areas = [d[1] for d in single_point_data]
                a_areas = [d[2] for d in single_point_data]
                plt.plot(n_values, p_areas, 'rs-', label="Single Point F - Passive")
                plt.plot(n_values, a_areas, 'bs-', label="Single Point F - Active")

            # Plot full range F data
            if full_range_data:
                n_values = [d[0] for d in full_range_data]
                p_areas = [d[1] for d in full_range_data]
                a_areas = [d[2] for d in full_range_data]
                plt.plot(n_values, p_areas, 'ro--', label="Full Range F - Passive")
                plt.plot(n_values, a_areas, 'bo--', label="Full Range F - Active")

            plt.title("Impact of Price Point Density on Triangle Areas")
            plt.xlabel("Number of Price Points (n)")
            plt.ylabel("Triangle Area")
            plt.xticks(sorted(list(set(r["n_points"] for r in results))))
            plt.legend()
            plt.grid(alpha=0.3)

            save_path = Path(save_dir) / "sparse_points_summary.svg"
            plt.savefig(save_path, dpi=300)
            plt.close()

        return results

    def generate_summary(self, results):
        """Generate experiment summary"""
        summary = {
            "extreme_binary": {
                "total_scenarios": len(results.get("extreme_binary", [])),
                "feasible_count": sum(1 for r in results.get("extreme_binary", []) if r["is_feasible"]),
                "avg_passive_area": np.mean([r["passive_area"] for r in results.get("extreme_binary", [])]),
                "avg_active_area": np.mean([r["active_area"] for r in results.get("extreme_binary", [])])
            },
            "degenerate": {
                "total_scenarios": len(results.get("degenerate", [])),
                "feasible_count": sum(1 for r in results.get("degenerate", []) if r["is_feasible"]),
                "scenarios": [r["scenario"] for r in results.get("degenerate", [])]
            },
            "extreme_f": {
                "total_scenarios": len(results.get("extreme_f", [])),
                "feasible_count": sum(1 for r in results.get("extreme_f", []) if r["is_feasible"]),
                "f_types": [r["F_type"] for r in results.get("extreme_f", [])],
                "avg_ratio_by_type": {
                    "Single Point": np.mean([r["area_ratio"] for r in results.get("extreme_f", [])
                                             if r["F_type"] == "Single Point" and r["area_ratio"] < 999]),
                    "Narrow Range": np.mean([r["area_ratio"] for r in results.get("extreme_f", [])
                                             if r["F_type"] == "Narrow Range" and r["area_ratio"] < 999]),
                    "Full Range": np.mean([r["area_ratio"] for r in results.get("extreme_f", [])
                                           if r["F_type"] == "Full Range" and r["area_ratio"] < 999])
                }
            },
            "sparse_price": {
                "total_scenarios": len(results.get("sparse_price", [])),
                "feasible_count": sum(1 for r in results.get("sparse_price", []) if r["is_feasible"]),
                "n_values": sorted(set(r["n_points"] for r in results.get("sparse_price", [])))
            }
        }
        return summary


if __name__ == "__main__":
    # Run boundary conditions experiment
    from pathlib import Path
    import time

    # Setup parameters
    save_dir = Path("results/boundary_conditions")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialize and run experiment
    start_time = time.time()
    experiment = BoundaryConditionsExperiment()
    results = experiment.run_experiment(save_dir=save_dir)

    # Generate summary
    summary = experiment.generate_summary(results)

    # Output timing and summary
    end_time = time.time()
    print(f"\nExperiment completed! Time: {end_time - start_time:.2f} seconds")
    print(f"Results saved in: {save_dir}")

    print("\n--- Experiment Summary ---")
    print(
        f"Extreme binary distribution: {summary['extreme_binary']['feasible_count']}/{summary['extreme_binary']['total_scenarios']} feasible")
    print(
        f"Degenerate markets: {summary['degenerate']['feasible_count']}/{summary['degenerate']['total_scenarios']} feasible")
    print(
        f"Extreme F conditions: {summary['extreme_f']['feasible_count']}/{summary['extreme_f']['total_scenarios']} feasible")
    print(
        f"Sparse price points: {summary['sparse_price']['feasible_count']}/{summary['sparse_price']['total_scenarios']} feasible")