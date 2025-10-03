import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from .scoring import ScoreVerifier
from .stats import StatisticsCalculator
from typing import Any

warnings.filterwarnings("ignore")


class ScoreVisualizationAnalyzer:
    """Comprehensive visualization tools for score comparison analysis."""

    def __init__(self, figsize=(15, 10), style="whitegrid"):
        """Initialize the visualization analyzer with default settings."""
        plt.style.use("default")
        sns.set_style(style)
        self.figsize = figsize
        self.colors = {
            "manual": "#2E86AB",
            "automatic": "#A23B72",
            "exact_match": "#2ECC71",
            "close_match": "#F39C12",
            "significant_diff": "#E74C3C",
            "not_scored": "#95A5A6",
        }

    def create_comprehensive_analysis(
        self,
        result_df: pd.DataFrame,
        save_plots: bool = False,
        output_dir: str = "score_analysis_plots",
    ) -> None:
        """Create a comprehensive set of visualizations for score analysis."""
        # Prepare data
        valid_scores_df = self._prepare_valid_scores_data(result_df)

        if valid_scores_df.empty:
            print("No valid score comparisons found for visualization.")
            return

        # Create plots
        fig = plt.figure(figsize=(20, 24))

        # 1. Score Distribution Comparison (Violin Plot)
        plt.subplot(4, 3, 1)
        self._create_score_violin_plot(valid_scores_df)

        # 2. Score Scatter Plot with Confidence
        plt.subplot(4, 3, 2)
        self._create_score_scatter_plot(valid_scores_df)

        # 3. Score Difference Distribution
        plt.subplot(4, 3, 3)
        self._create_difference_distribution(valid_scores_df)

        # 4. Match Type Distribution
        plt.subplot(4, 3, 4)
        self._create_match_type_distribution(result_df)

        # 5. Confidence Score Distribution
        plt.subplot(4, 3, 5)
        self._create_confidence_distribution(result_df)

        # 6. Score Differences by Match Type
        plt.subplot(4, 3, 6)
        self._create_difference_by_match_type(valid_scores_df)

        # 7. Publication Match Type Distribution
        plt.subplot(4, 3, 7)
        self._create_publication_match_distribution(result_df)

        # 8. Correlation Heatmap
        plt.subplot(4, 3, 8)
        self._create_correlation_heatmap(valid_scores_df)

        # 9. Score Range Analysis
        plt.subplot(4, 3, 9)
        self._create_score_range_analysis(valid_scores_df)

        # 10. Accuracy by Score Range
        plt.subplot(4, 3, 10)
        self._create_accuracy_by_score_range(valid_scores_df)

        # 11. Gene-wise Performance (if applicable)
        plt.subplot(4, 3, 11)
        self._create_gene_performance_summary(result_df)

        # 12. Error Analysis
        plt.subplot(4, 3, 12)
        self._create_error_analysis(valid_scores_df)

        plt.tight_layout()

        if save_plots:
            import os

            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                f"{output_dir}/comprehensive_score_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Comprehensive analysis saved to {output_dir}/comprehensive_score_analysis.png"
            )

        plt.show()

        # Create additional detailed plots
        self._create_detailed_violin_comparison(valid_scores_df, save_plots, output_dir)
        self._create_advanced_scatter_analysis(valid_scores_df, save_plots, output_dir)
        self._create_performance_dashboard(result_df, save_plots, output_dir)

    def _prepare_valid_scores_data(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for visualization by filtering valid score comparisons."""
        # Note: NOT SCORED cases are now excluded entirely from processing
        valid_scores = result_df[
            (result_df["manual_case_id"] != "NO MATCH")
            & (result_df["manual_case_id"] != "NO CASE MATCH")
            & (result_df["manual_case_id"] != "PUBLICATION_MATCH_ONLY")
            & pd.notna(result_df["manual_score"])
            & pd.notna(result_df["automatic_score"])
        ].copy()

        # Ensure numeric data types and remove any remaining invalid values
        if not valid_scores.empty:
            valid_scores["manual_score"] = pd.to_numeric(
                valid_scores["manual_score"], errors="coerce"
            )
            valid_scores["automatic_score"] = pd.to_numeric(
                valid_scores["automatic_score"], errors="coerce"
            )
            valid_scores["case_confidence"] = pd.to_numeric(
                valid_scores["case_confidence"], errors="coerce"
            )

            # Remove rows with NaN values after conversion
            valid_scores = valid_scores.dropna(
                subset=["manual_score", "automatic_score"]
            )

        return valid_scores

    def _create_score_violin_plot(self, valid_scores_df: pd.DataFrame) -> None:
        """Create violin plot comparing manual vs automatic score distributions."""
        # Prepare data for violin plot with proper data cleaning
        manual_scores = valid_scores_df["manual_score"].dropna().astype(float)
        auto_scores = valid_scores_df["automatic_score"].dropna().astype(float)

        if len(manual_scores) == 0 or len(auto_scores) == 0:
            plt.text(
                0.5,
                0.5,
                "No valid score data for violin plot",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(
                "Score Distribution Comparison\n(Violin Plot)",
                fontsize=12,
                fontweight="bold",
            )
            return

        # Create properly formatted dataframe
        data_for_violin = []
        data_for_violin.extend(
            [("Manual", float(score)) for score in manual_scores if not pd.isna(score)]
        )
        data_for_violin.extend(
            [("Automatic", float(score)) for score in auto_scores if not pd.isna(score)]
        )

        violin_df = pd.DataFrame(data_for_violin, columns=["Score_Type", "Score"])

        # Ensure Score_Type is string and Score is numeric
        violin_df["Score_Type"] = violin_df["Score_Type"].astype(str)
        violin_df["Score"] = pd.to_numeric(violin_df["Score"], errors="coerce")

        # Remove any remaining NaN values
        violin_df = violin_df.dropna()

        if len(violin_df) == 0:
            plt.text(
                0.5,
                0.5,
                "No valid data for violin plot after cleaning",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(
                "Score Distribution Comparison\n(Violin Plot)",
                fontsize=12,
                fontweight="bold",
            )
            return

        try:
            sns.violinplot(
                data=violin_df,
                x="Score_Type",
                y="Score",
                palette=[self.colors["manual"], self.colors["automatic"]],
            )
            plt.title(
                "Score Distribution Comparison\n(Violin Plot)",
                fontsize=12,
                fontweight="bold",
            )
            plt.ylabel("Score Value")
            plt.grid(True, alpha=0.3)

            # Add mean lines
            manual_mean = manual_scores.mean()
            auto_mean = auto_scores.mean()
            plt.axhline(
                y=manual_mean,
                color=self.colors["manual"],
                linestyle="--",
                alpha=0.7,
                linewidth=1,
            )
            plt.axhline(
                y=auto_mean,
                color=self.colors["automatic"],
                linestyle="--",
                alpha=0.7,
                linewidth=1,
            )

            # Add statistics text
            plt.text(
                0.02,
                0.98,
                f"Manual μ: {manual_mean:.2f}\nAuto μ: {auto_mean:.2f}",
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except Exception as e:
            plt.text(
                0.5,
                0.5,
                f"Error creating violin plot:\n{str(e)}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(
                "Score Distribution Comparison\n(Violin Plot)",
                fontsize=12,
                fontweight="bold",
            )

    def _create_score_scatter_plot(self, valid_scores_df: pd.DataFrame) -> None:
        """Create scatter plot of manual vs automatic scores with confidence coloring."""
        scatter = plt.scatter(
            valid_scores_df["manual_score"],
            valid_scores_df["automatic_score"],
            c=valid_scores_df["case_confidence"],
            cmap="viridis",
            alpha=0.6,
            s=50,
        )

        # Add perfect correlation line
        min_score = min(
            valid_scores_df["manual_score"].min(),
            valid_scores_df["automatic_score"].min(),
        )
        max_score = max(
            valid_scores_df["manual_score"].max(),
            valid_scores_df["automatic_score"].max(),
        )
        plt.plot(
            [min_score, max_score],
            [min_score, max_score],
            "r--",
            alpha=0.7,
            linewidth=2,
            label="Perfect Match",
        )

        plt.xlabel("Manual Score")
        plt.ylabel("Automatic Score")
        plt.title(
            "Manual vs Automatic Scores\n(Colored by Match Confidence)",
            fontsize=12,
            fontweight="bold",
        )
        plt.colorbar(scatter, label="Match Confidence")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Calculate and display correlation
        correlation = np.corrcoef(
            valid_scores_df["manual_score"], valid_scores_df["automatic_score"]
        )[0, 1]
        plt.text(
            0.02,
            0.98,
            f"Correlation: {correlation:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _create_difference_distribution(self, valid_scores_df: pd.DataFrame) -> None:
        """Create histogram of score differences."""
        differences = valid_scores_df["difference"]

        plt.hist(
            differences,
            bins=30,
            alpha=0.7,
            color=self.colors["automatic"],
            edgecolor="black",
            linewidth=0.5,
        )
        plt.axvline(
            x=0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Perfect Match",
        )
        plt.axvline(
            x=differences.mean(),
            color="orange",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {differences.mean():.3f}",
        )

        plt.xlabel("Score Difference (Auto - Manual)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Score Differences", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Mean: {differences.mean():.3f}\nStd: {differences.std():.3f}\nMedian: {differences.median():.3f}"
        plt.text(
            0.98,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _create_match_type_distribution(self, result_df: pd.DataFrame) -> None:
        """Create bar chart of match type distribution."""
        match_counts = result_df["match_type"].value_counts()

        colors = []
        for match_type in match_counts.index:
            if "EXACT" in match_type:
                colors.append(self.colors["exact_match"])
            elif "CLOSE" in match_type:
                colors.append(self.colors["close_match"])
            elif "SIGNIFICANT" in match_type:
                colors.append(self.colors["significant_diff"])
            else:
                colors.append(self.colors["not_scored"])

        bars = plt.bar(
            range(len(match_counts)), match_counts.values, color=colors, alpha=0.8
        )
        plt.xticks(
            range(len(match_counts)), match_counts.index, rotation=45, ha="right"
        )
        plt.ylabel("Count")
        plt.title("Score Match Type Distribution", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, value in zip(bars, match_counts.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _create_confidence_distribution(self, result_df: pd.DataFrame) -> None:
        """Create histogram of case confidence scores."""
        confidences = result_df[result_df["case_confidence"] > 0]["case_confidence"]

        if confidences.empty:
            plt.text(
                0.5,
                0.5,
                "No confidence data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Case Confidence Distribution", fontsize=12, fontweight="bold")
            return

        plt.hist(
            confidences,
            bins=20,
            alpha=0.7,
            color=self.colors["manual"],
            edgecolor="black",
            linewidth=0.5,
        )
        plt.axvline(
            x=confidences.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {confidences.mean():.3f}",
        )

        plt.xlabel("Case Match Confidence")
        plt.ylabel("Frequency")
        plt.title("Case Confidence Distribution", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _create_difference_by_match_type(self, valid_scores_df: pd.DataFrame) -> None:
        """Create box plot of score differences grouped by match type."""
        match_types = valid_scores_df["match_type"].unique()

        if len(match_types) > 1:
            sns.boxplot(data=valid_scores_df, x="match_type", y="absolute_difference")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Absolute Score Difference")
            plt.title("Score Differences by Match Type", fontsize=12, fontweight="bold")
            plt.grid(True, alpha=0.3, axis="y")
        else:
            plt.text(
                0.5,
                0.5,
                f"Single match type: {match_types[0]}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Score Differences by Match Type", fontsize=12, fontweight="bold")

    def _create_publication_match_distribution(self, result_df: pd.DataFrame) -> None:
        """Create pie chart of publication match types."""
        pub_match_counts = result_df["publication_match_type"].value_counts()

        plt.pie(
            pub_match_counts.values,
            labels=pub_match_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Set3.colors,
        )
        plt.title("Publication Match Type Distribution", fontsize=12, fontweight="bold")

    def _create_correlation_heatmap(self, valid_scores_df: pd.DataFrame) -> None:
        """Create correlation heatmap of numerical variables."""
        numerical_cols = [
            "manual_score",
            "automatic_score",
            "difference",
            "absolute_difference",
            "case_confidence",
        ]
        correlation_data = valid_scores_df[numerical_cols].corr()

        sns.heatmap(
            correlation_data,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation Matrix", fontsize=12, fontweight="bold")

    def _create_score_range_analysis(self, valid_scores_df: pd.DataFrame) -> None:
        """Analyze accuracy across different score ranges."""
        # Define score ranges
        score_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        range_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]

        accuracies = []
        counts = []

        for (low, high), label in zip(score_ranges, range_labels):
            range_data = valid_scores_df[
                (valid_scores_df["manual_score"] >= low)
                & (valid_scores_df["manual_score"] < high)
            ]

            if len(range_data) > 0:
                accuracy = (range_data["absolute_difference"] <= 0.5).mean()
                accuracies.append(accuracy)
                counts.append(len(range_data))
            else:
                accuracies.append(0)
                counts.append(0)

        bars = plt.bar(range_labels, accuracies, alpha=0.8, color=self.colors["manual"])
        plt.ylabel("Accuracy (within 0.5 points)")
        plt.xlabel("Manual Score Range")
        plt.title("Accuracy by Score Range", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")

        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _create_accuracy_by_score_range(self, valid_scores_df: pd.DataFrame) -> None:
        """Create detailed accuracy analysis by score ranges."""
        # Bin scores into ranges
        valid_scores_df["score_bin"] = pd.cut(
            valid_scores_df["manual_score"], bins=5, precision=1
        )

        bin_stats = (
            valid_scores_df.groupby("score_bin")
            .agg(
                {
                    "absolute_difference": ["mean", "std", "count"],
                    "scores_match": "mean",
                }
            )
            .round(3)
        )

        if not bin_stats.empty:
            x_pos = range(len(bin_stats))
            plt.bar(
                x_pos,
                bin_stats[("absolute_difference", "mean")],
                alpha=0.8,
                color=self.colors["automatic"],
                yerr=bin_stats[("absolute_difference", "std")],
                capsize=5,
            )

            plt.xticks(x_pos, [str(idx) for idx in bin_stats.index], rotation=45)
            plt.ylabel("Mean Absolute Difference")
            plt.xlabel("Manual Score Range")
            plt.title("Error by Score Range", fontsize=12, fontweight="bold")
            plt.grid(True, alpha=0.3, axis="y")
        else:
            plt.text(
                0.5,
                0.5,
                "Insufficient data for binning",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Error by Score Range", fontsize=12, fontweight="bold")

    def _create_gene_performance_summary(self, result_df: pd.DataFrame) -> None:
        """Create summary of performance by gene (if applicable)."""
        valid_genes = result_df[
            (result_df["manual_gene"] != "")
            & (result_df["automatic_gene"] != "")
            & (result_df["manual_case_id"] != "NO MATCH")
        ]

        if len(valid_genes) == 0:
            plt.text(
                0.5,
                0.5,
                "No gene matching data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Gene Matching Performance", fontsize=12, fontweight="bold")
            return

        gene_matches = (
            valid_genes["manual_gene"] == valid_genes["automatic_gene"]
        ).sum()
        total_cases = len(valid_genes)

        labels = ["Gene Match", "Gene Mismatch"]
        sizes = [gene_matches, total_cases - gene_matches]
        colors = [self.colors["exact_match"], self.colors["significant_diff"]]

        plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        plt.title(
            f"Gene Matching Performance\n({gene_matches}/{total_cases} matches)",
            fontsize=12,
            fontweight="bold",
        )

    def _create_error_analysis(self, valid_scores_df: pd.DataFrame) -> None:
        """Create error analysis visualization."""
        errors = valid_scores_df["difference"]

        # Calculate error metrics
        mae = errors.abs().mean()
        rmse = np.sqrt((errors**2).mean())

        # Create error distribution with statistics
        plt.hist(
            errors,
            bins=25,
            alpha=0.7,
            color=self.colors["automatic"],
            edgecolor="black",
            linewidth=0.5,
        )

        plt.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        plt.axvline(
            x=errors.mean(), color="orange", linestyle="-", linewidth=2, alpha=0.7
        )

        plt.xlabel("Error (Auto - Manual)")
        plt.ylabel("Frequency")
        plt.title("Error Analysis", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # Add error statistics
        stats_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nBias: {errors.mean():.3f}"
        plt.text(
            0.98,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _create_detailed_violin_comparison(
        self,
        valid_scores_df: pd.DataFrame,
        save_plots: bool = False,
        output_dir: str = "",
    ) -> None:
        """Create detailed violin plot comparison with additional statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Clean and prepare data
        manual_scores = valid_scores_df["manual_score"].dropna().astype(float)
        auto_scores = valid_scores_df["automatic_score"].dropna().astype(float)

        if len(manual_scores) == 0 or len(auto_scores) == 0:
            ax1.text(
                0.5,
                0.5,
                "No valid score data for detailed violin plot",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title(
                "Detailed Score Distribution Comparison", fontsize=14, fontweight="bold"
            )
            ax2.text(
                0.5,
                0.5,
                "No valid score data for histogram",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Score Distribution Overlay", fontsize=14, fontweight="bold")
            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    f"{output_dir}/detailed_violin_comparison.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()
            return

        # Create properly formatted dataframe
        data_for_violin = []
        data_for_violin.extend(
            [("Manual", float(score)) for score in manual_scores if not pd.isna(score)]
        )
        data_for_violin.extend(
            [("Automatic", float(score)) for score in auto_scores if not pd.isna(score)]
        )

        violin_df = pd.DataFrame(data_for_violin, columns=["Score_Type", "Score"])

        # Ensure proper data types
        violin_df["Score_Type"] = violin_df["Score_Type"].astype(str)
        violin_df["Score"] = pd.to_numeric(violin_df["Score"], errors="coerce")
        violin_df = violin_df.dropna()

        if len(violin_df) > 0:
            try:
                # Enhanced violin plot
                sns.violinplot(
                    data=violin_df,
                    x="Score_Type",
                    y="Score",
                    ax=ax1,
                    palette=[self.colors["manual"], self.colors["automatic"]],
                )

                # Add box plot overlay
                sns.boxplot(
                    data=violin_df,
                    x="Score_Type",
                    y="Score",
                    ax=ax1,
                    width=0.2,
                    boxprops=dict(alpha=0.7),
                )

                ax1.set_title(
                    "Detailed Score Distribution Comparison",
                    fontsize=14,
                    fontweight="bold",
                )
                ax1.grid(True, alpha=0.3)
            except Exception as e:
                ax1.text(
                    0.5,
                    0.5,
                    f"Error creating violin plot:\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title(
                    "Detailed Score Distribution Comparison",
                    fontsize=14,
                    fontweight="bold",
                )
        else:
            ax1.text(
                0.5,
                0.5,
                "No valid data after cleaning",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title(
                "Detailed Score Distribution Comparison", fontsize=14, fontweight="bold"
            )

        # Statistical comparison - histogram overlay
        try:
            ax2.hist(
                manual_scores,
                bins=20,
                alpha=0.6,
                label="Manual",
                color=self.colors["manual"],
                density=True,
            )
            ax2.hist(
                auto_scores,
                bins=20,
                alpha=0.6,
                label="Automatic",
                color=self.colors["automatic"],
                density=True,
            )

            ax2.set_xlabel("Score Value")
            ax2.set_ylabel("Density")
            ax2.set_title("Score Distribution Overlay", fontsize=14, fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(
                0.5,
                0.5,
                f"Error creating histogram:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Score Distribution Overlay", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_plots:
            plt.savefig(
                f"{output_dir}/detailed_violin_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Detailed violin comparison saved to {output_dir}/detailed_violin_comparison.png"
            )

        plt.show()

    def _create_advanced_scatter_analysis(
        self,
        valid_scores_df: pd.DataFrame,
        save_plots: bool = False,
        output_dir: str = "",
    ) -> None:
        """Create advanced scatter plot analysis with multiple views."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Basic scatter with confidence
        scatter1 = ax1.scatter(
            valid_scores_df["manual_score"],
            valid_scores_df["automatic_score"],
            c=valid_scores_df["case_confidence"],
            cmap="viridis",
            alpha=0.7,
            s=60,
        )

        # Perfect correlation line
        min_score = min(
            valid_scores_df["manual_score"].min(),
            valid_scores_df["automatic_score"].min(),
        )
        max_score = max(
            valid_scores_df["manual_score"].max(),
            valid_scores_df["automatic_score"].max(),
        )
        ax1.plot(
            [min_score, max_score],
            [min_score, max_score],
            "r--",
            alpha=0.7,
            linewidth=2,
        )

        ax1.set_xlabel("Manual Score")
        ax1.set_ylabel("Automatic Score")
        ax1.set_title("Scores with Match Confidence")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label="Confidence")

        # 2. Scatter with error bars (if multiple measurements exist)
        ax2.scatter(
            valid_scores_df["manual_score"],
            valid_scores_df["automatic_score"],
            alpha=0.6,
            s=50,
            color=self.colors["automatic"],
        )
        ax2.plot(
            [min_score, max_score],
            [min_score, max_score],
            "r--",
            alpha=0.7,
            linewidth=2,
        )
        ax2.set_xlabel("Manual Score")
        ax2.set_ylabel("Automatic Score")
        ax2.set_title("Basic Score Comparison")
        ax2.grid(True, alpha=0.3)

        # 3. Residuals plot
        residuals = valid_scores_df["difference"]
        ax3.scatter(
            valid_scores_df["manual_score"],
            residuals,
            alpha=0.6,
            s=50,
            color=self.colors["manual"],
        )
        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax3.set_xlabel("Manual Score")
        ax3.set_ylabel("Residuals (Auto - Manual)")
        ax3.set_title("Residuals vs Manual Score")
        ax3.grid(True, alpha=0.3)

        # 4. Bland-Altman plot
        mean_scores = (
            valid_scores_df["manual_score"] + valid_scores_df["automatic_score"]
        ) / 2
        diff_scores = valid_scores_df["difference"]

        ax4.scatter(
            mean_scores, diff_scores, alpha=0.6, s=50, color=self.colors["automatic"]
        )
        ax4.axhline(
            y=diff_scores.mean(),
            color="red",
            linestyle="-",
            alpha=0.7,
            label=f"Mean diff: {diff_scores.mean():.3f}",
        )
        ax4.axhline(
            y=diff_scores.mean() + 1.96 * diff_scores.std(),
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax4.axhline(
            y=diff_scores.mean() - 1.96 * diff_scores.std(),
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax4.set_xlabel("Mean Score")
        ax4.set_ylabel("Difference (Auto - Manual)")
        ax4.set_title("Bland-Altman Plot")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(
                f"{output_dir}/advanced_scatter_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Advanced scatter analysis saved to {output_dir}/advanced_scatter_analysis.png"
            )

        plt.show()

    def _create_performance_dashboard(
        self, result_df: pd.DataFrame, save_plots: bool = False, output_dir: str = ""
    ) -> None:
        """Create a performance dashboard with key metrics."""
        fig = plt.figure(figsize=(20, 12))

        # Calculate key metrics
        total_cases = len(result_df)
        matched_cases = len(
            result_df[
                (result_df["manual_case_id"] != "NO MATCH")
                & (result_df["manual_case_id"] != "NO CASE MATCH")
                & (result_df["manual_case_id"] != "PUBLICATION_MATCH_ONLY")
            ]
        )

        valid_scores = result_df[
            pd.notna(result_df["manual_score"]) & pd.notna(result_df["automatic_score"])
        ]

        exact_matches = len(valid_scores[valid_scores["scores_match"] == True])

        # 1. Key Performance Indicators
        ax1 = plt.subplot(3, 4, 1)
        kpis = [
            ("Total Cases", total_cases),
            ("Matched Cases", matched_cases),
            ("Valid Scores", len(valid_scores)),
            ("Exact Matches", exact_matches),
        ]

        y_pos = range(len(kpis))
        values = [kpi[1] for kpi in kpis]
        labels = [kpi[0] for kpi in kpis]

        bars = plt.barh(
            y_pos,
            values,
            color=[
                self.colors["manual"],
                self.colors["automatic"],
                self.colors["close_match"],
                self.colors["exact_match"],
            ],
        )
        plt.yticks(y_pos, labels)
        plt.xlabel("Count")
        plt.title("Key Performance Indicators", fontweight="bold")

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
                fontsize=10,
            )

        # 2. Accuracy Rates
        plt.subplot(3, 4, 2)
        if matched_cases > 0 and len(valid_scores) > 0:
            matching_rate = matched_cases / total_cases
            accuracy_rate = (
                exact_matches / len(valid_scores) if len(valid_scores) > 0 else 0
            )

            rates = [matching_rate, accuracy_rate]
            rate_labels = ["Matching\nRate", "Accuracy\nRate"]

            plt.bar(
                rate_labels,
                rates,
                color=[self.colors["manual"], self.colors["exact_match"]],
                alpha=0.8,
            )
            plt.ylabel("Rate")
            plt.title("Performance Rates", fontweight="bold")
            plt.ylim(0, 1)

            # Add percentage labels
            for i, rate in enumerate(rates):
                plt.text(
                    i,
                    rate + 0.02,
                    f"{rate:.1%}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # 3. Score Distribution Summary
        plt.subplot(3, 4, 3)
        if not valid_scores.empty:
            # Ensure data is properly numeric for boxplot
            try:
                manual_scores_clean = pd.to_numeric(
                    valid_scores["manual_score"], errors="coerce"
                ).dropna()
                auto_scores_clean = pd.to_numeric(
                    valid_scores["automatic_score"], errors="coerce"
                ).dropna()

                if len(manual_scores_clean) > 0 and len(auto_scores_clean) > 0:
                    plt.boxplot(
                        [manual_scores_clean.values, auto_scores_clean.values],
                        labels=["Manual", "Automatic"],
                    )
                    plt.ylabel("Score Value")
                    plt.title("Score Distribution Summary", fontweight="bold")
                    plt.grid(True, alpha=0.3, axis="y")
                else:
                    plt.text(
                        0.5,
                        0.5,
                        "No valid numeric data for boxplot",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title("Score Distribution Summary", fontweight="bold")
            except Exception as e:
                plt.text(
                    0.5,
                    0.5,
                    f"Error creating boxplot:\n{str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title("Score Distribution Summary", fontweight="bold")
        else:
            plt.text(
                0.5,
                0.5,
                "No valid data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Score Distribution Summary", fontweight="bold")

        # 4. Error Distribution
        plt.subplot(3, 4, 4)
        if not valid_scores.empty:
            errors = valid_scores["absolute_difference"]
            plt.hist(
                errors,
                bins=15,
                alpha=0.7,
                color=self.colors["automatic"],
                edgecolor="black",
            )
            plt.axvline(
                x=errors.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {errors.mean():.3f}",
            )
            plt.xlabel("Absolute Error")
            plt.ylabel("Frequency")
            plt.title("Error Distribution", fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Add more dashboard components...
        # 5-12: Additional performance metrics and visualizations

        plt.tight_layout()

        if save_plots:
            plt.savefig(
                f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches="tight"
            )
            print(
                f"Performance dashboard saved to {output_dir}/performance_dashboard.png"
            )

        plt.show()


def visualize_score_comparison(
    result_df: pd.DataFrame,
    save_plots: bool = False,
    output_dir: str = "score_analysis_plots",
) -> None:
    """
    visualizations for score comparison analysis.

    Args:
        result_df: Results DataFrame from compare_manual_vs_automatic_scores
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots (created if doesn't exist)
    """
    if result_df.empty:
        print("No data available for visualization.")
        return

    # Initialize the visualization analyzer
    analyzer = ScoreVisualizationAnalyzer()

    # Create analysis
    print("Creating comprehensive score comparison visualizations...")
    analyzer.create_comprehensive_analysis(result_df, save_plots, output_dir)

    print("\nVisualization complete!")
    if save_plots:
        print(f"All plots saved to directory: {output_dir}")


def print_verification_summary(
    result_df: pd.DataFrame, require_gene_match: bool = True
) -> None:
    """Print a summary of verification results."""
    print("\n" + "=" * 80)
    print("SCORE VERIFICATION SUMMARY")
    if require_gene_match:
        print("(Gene matching REQUIRED - only cases with matching genes are included)")
    else:
        print("(Gene matching NOT required - legacy behavior)")
    print("(NOT SCORED cases are excluded entirely from comparison)")
    print("=" * 80)

    stats = StatisticsCalculator.calculate_summary_stats(result_df, require_gene_match)

    print(f"Total cases processed: {stats['total_cases']}")
    print(f"Publications with matches: {stats['matched_publications']}")
    print(f"Cases with matches: {stats['matched_cases']}")
    print(f"Exact score matches: {stats['exact_score_matches']}")
    print(f"Overall matching rate: {stats['matching_rate']:.1%}")
    print(f"Exact match rate (of matched): {stats['exact_match_rate']:.1%}")

    # Publication match type breakdown
    if "publication_match_type" in result_df.columns:
        print("\nPublication Match Types:")
        pub_match_types = result_df["publication_match_type"].value_counts()
        for match_type, count in pub_match_types.items():
            print(f"  - {match_type}: {count}")

    # Score match type breakdown
    if "match_type" in result_df.columns:
        valid_score_matches = result_df[
            (result_df["manual_case_id"] != "NO MATCH")
            & (result_df["manual_case_id"] != "NO CASE MATCH")
            & (result_df["manual_case_id"] != "PUBLICATION_MATCH_ONLY")
        ]
        if not valid_score_matches.empty:
            print("\nScore Match Types:")
            score_match_types = valid_score_matches["match_type"].value_counts()
            for match_type, count in score_match_types.items():
                print(f"  - {match_type}: {count}")

    # Score differences
    if "avg_absolute_difference" in stats:
        print(f"\nScore Differences:")
        print(
            f"  - Average absolute difference: {stats['avg_absolute_difference']:.3f}"
        )
        print(
            f"  - Median absolute difference: {stats['median_absolute_difference']:.3f}"
        )
        print(f"  - Standard deviation: {stats['std_absolute_difference']:.3f}")
        print(
            f"  - Min/Max difference: {stats['min_absolute_difference']:.3f} / {stats['max_absolute_difference']:.3f}"
        )

    # Case confidence distribution
    if "case_confidence" in result_df.columns:
        matched_with_confidence = result_df[
            (result_df["manual_case_id"] != "NO MATCH")
            & (result_df["manual_case_id"] != "NO CASE MATCH")
            & (result_df["manual_case_id"] != "PUBLICATION_MATCH_ONLY")
        ]
        if not matched_with_confidence.empty:
            confidences = matched_with_confidence["case_confidence"]
            print("\nCase Matching Confidence:")
            print(f"  - Average confidence: {confidences.mean():.3f}")
            print(f"  - High confidence (≥0.8): {len(confidences[confidences >= 0.8])}")
            print(
                f"  - Medium confidence (≥0.5): {len(confidences[confidences >= 0.5])}"
            )
            print(f"  - Low confidence (<0.5): {len(confidences[confidences < 0.5])}")

    print("\n" + "=" * 80)


def compare_manual_vs_automatic_scores(
    manual_df: pd.DataFrame,
    auto_extractions: Any,
    show_summary: bool = True,
    save_to_file: str = None,
    require_gene_match: bool = True,
    publication_match_threshold: float = 0.8,
    case_match_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Workflow to compare manual vs automatic scores with matching functionality.

    Args:
        manual_df: DataFrame with manual dataset
        auto_extractions: Auto extraction data (various formats supported)
        show_summary: Whether to print summary statistics
        save_to_file: Optional filename to save results
        require_gene_match: If True, only compare cases with matching genes
        publication_match_threshold: Minimum confidence for publication matching
        case_match_threshold: Minimum confidence for case matching

    Returns:
        DataFrame with detailed comparison results
    """

    # Validate manual dataset columns
    required_columns = ["publication", "pmid", "final_score", "gene", "id"]
    missing_columns = [col for col in required_columns if col not in manual_df.columns]

    if missing_columns:
        raise ValueError(
            f"Manual dataset is missing required columns: {missing_columns}"
        )

    # Initialize the enhanced verifier with custom thresholds
    verifier = ScoreVerifier(
        require_gene_match=require_gene_match,
        publication_match_threshold=publication_match_threshold,
        case_match_threshold=case_match_threshold,
    )

    # Run the verification
    result_df = verifier.verify_scores(manual_df, auto_extractions)

    # Show summary if requested
    if show_summary:
        print_verification_summary(result_df, require_gene_match)

    # Save to file if requested
    if save_to_file:
        result_df.to_csv(save_to_file, index=False)
        print(f"\nResults saved to: {save_to_file}")

    return result_df


def compare_manual_vs_automatic_scores_with_viz(
    manual_df: pd.DataFrame,
    auto_extractions: Any,
    show_summary: bool = True,
    save_to_file: str = None,
    require_gene_match: bool = True,
    publication_match_threshold: float = 0.8,
    case_match_threshold: float = 0.4,
    create_visualizations: bool = True,
    save_plots: bool = False,
    plot_output_dir: str = "score_analysis_plots",
) -> pd.DataFrame:
    """
    Enhanced workflow with visualization capabilities.

    Args:
        manual_df: DataFrame with manual dataset
        auto_extractions: Auto extraction data (various formats supported)
        show_summary: Whether to print summary statistics
        save_to_file: Optional filename to save results
        require_gene_match: If True, only compare cases with matching genes
        publication_match_threshold: Minimum confidence for publication matching
        case_match_threshold: Minimum confidence for case matching
        create_visualizations: Whether to create visualization plots
        save_plots: Whether to save plots to files
        plot_output_dir: Directory to save plots

    Returns:
        DataFrame with detailed comparison results
    """

    # Run the basic comparison
    result_df = compare_manual_vs_automatic_scores(
        manual_df,
        auto_extractions,
        show_summary,
        save_to_file,
        require_gene_match,
        publication_match_threshold,
        case_match_threshold,
    )

    # Create visualizations if requested
    if create_visualizations:
        print("\n" + "=" * 60)
        print("CREATING SCORE COMPARISON VISUALIZATIONS")
        print("=" * 60)
        visualize_score_comparison(result_df, save_plots, plot_output_dir)

    return result_df
