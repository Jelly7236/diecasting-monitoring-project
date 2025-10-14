# viz/shap_plots.py
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from shiny import render, reactive


def register_shap_plots(output, shap_values_state, X_input_state, y_test_state, models, explainers, input):
    """
    SHAP Force Plot / Summary Plot / Permutation Importance 등록
    """

    # -----------------------
    # 1. Force Plot (개별 샘플)
    # -----------------------
    @output
    @render.plot
    def shap_force_plot():
        if input.btn_predict() == 0:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, '예측을 실행하면 SHAP Force Plot이 표시됩니다',
                    ha='center', va='center', fontsize=12, color='#6c757d',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
            ax.axis('off')
            plt.tight_layout()
            return fig

        shap_values = shap_values_state.get()
        X = X_input_state.get()

        if shap_values is None or X is None:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, "SHAP 값을 계산할 수 없습니다",
                    ha="center", va="center", fontsize=12, color='#dc3545')
            ax.axis("off")
            plt.tight_layout()
            return fig

        explainer = explainers.get(input.mold_code())
        if explainer is None:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, f"금형 코드 '{input.mold_code()}'에 대한 Explainer가 없습니다",
                    ha="center", va="center", fontsize=12, color='#dc3545')
            ax.axis("off")
            plt.tight_layout()
            return fig

        try:
            base_value = explainer.expected_value
            if hasattr(base_value, "__len__"):
                base_value = base_value[1]

            if hasattr(shap_values, "values"):
                vals = shap_values.values[0]
                if vals.ndim == 2 and vals.shape[1] == 2:
                    vals = vals[:, 1]
            else:
                vals = shap_values[0]

            # 한글 변수명 + 소수점 반올림
            from shared import feature_name_map_kor
            X_korean = X.copy()
            X_korean.columns = [feature_name_map_kor.get(col, col) for col in X.columns]
            X_rounded = X_korean.iloc[0, :].round(4)  # ✅ 소수점 4자리

            X_labels = X_rounded.copy()
            X_labels.index = [
                f"{col}\n" for col, val in zip(X_labels.index, X_labels.values)
            ]
            plt.figure(figsize=(14, 3))
            shap.plots.force(base_value, vals, X_labels, matplotlib=True, show=False)
            plt.title("SHAP Force Plot (개별 샘플 분석)", fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout(pad=1.5)
            return plt.gcf()
        
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, f"SHAP Plot 생성 오류:\n{str(e)[:100]}",
                    ha="center", va="center", fontsize=10, color='#dc3545')
            ax.axis("off")
            plt.tight_layout()
            return fig

    # # -----------------------
    # # 2. Summary Plot (전체 샘플)
    # # -----------------------
    # @output
    # @render.plot
    # def shap_summary_plot():
    #     if input.btn_predict() == 0:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         ax.text(0.5, 0.5, '예측을 실행하면 SHAP Summary Plot이 표시됩니다',
    #                 ha='center', va='center', fontsize=12, color='#6c757d',
    #                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    #         ax.axis('off')
    #         plt.tight_layout()
    #         return fig

    #     shap_values = shap_values_state.get()
    #     X = X_input_state.get()

    #     if shap_values is None or X is None:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         ax.text(0.5, 0.5, "SHAP 값을 계산할 수 없습니다",
    #                 ha="center", va="center", fontsize=12, color='#dc3545')
    #         ax.axis("off")
    #         plt.tight_layout()
    #         return fig

    #     try:
    #         if shap_values.values.ndim == 3:
    #             vals = shap_values.values[:, :, 1]
    #         else:
    #             vals = shap_values.values

    #         from shared import feature_name_map_kor
    #         X_korean = X.copy()
    #         X_korean.columns = [feature_name_map_kor.get(col, col) for col in X.columns]

    #         plt.figure(figsize=(11, max(len(X.columns) * 0.4, 8)))
    #         shap.summary_plot(vals, X_korean, show=False)
    #         plt.title("SHAP Summary Plot (전체 변수 영향도)", fontsize=13, fontweight='bold', pad=15)
    #         plt.yticks(fontsize=9)
    #         plt.xticks(fontsize=9)
    #         plt.tight_layout(pad=1.5)
    #         return plt.gcf()
        
    #     except Exception as e:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         ax.text(0.5, 0.5, f"Summary Plot 생성 오류:\n{str(e)[:100]}",
    #                 ha="center", va="center", fontsize=10, color='#dc3545')
    #         ax.axis("off")
    #         plt.tight_layout()
    #         return fig

    # # -----------------------
    # # 3. Permutation Importance
    # # -----------------------
    # @output
    # @render.plot
    # def permutation_importance_plot():
    #     if input.btn_predict() == 0:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         ax.text(0.5, 0.5, '예측을 실행하면 Permutation Importance가 표시됩니다',
    #                 ha='center', va='center', fontsize=12, color='#6c757d',
    #                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    #         ax.axis('off')
    #         plt.tight_layout()
    #         return fig

    #     mold_code = input.mold_code()
    #     model = models.get(mold_code)
    #     X = X_input_state.get()
    #     y = y_test_state.get()

    #     if model is None or X is None or y is None:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         ax.text(0.5, 0.5, "모델 또는 데이터를 찾을 수 없습니다",
    #                 ha="center", va="center", fontsize=12, color='#dc3545')
    #         ax.axis("off")
    #         plt.tight_layout()
    #         return fig

    #     try:
    #         result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    #         sorted_idx = result.importances_mean.argsort()

    #         from shared import feature_name_map_kor
    #         feature_names_korean = [feature_name_map_kor.get(col, col) for col in X.columns]

    #         plt.figure(figsize=(11, max(len(X.columns) * 0.4, 8)))
    #         plt.barh([feature_names_korean[i] for i in sorted_idx],
    #                  result.importances_mean[sorted_idx],
    #                  color='#0d6efd', alpha=0.7)
    #         plt.xlabel("Permutation Importance", fontsize=11, fontweight='bold')
    #         plt.title(f"Permutation Importance (금형: {mold_code})",
    #                   fontsize=13, fontweight='bold', pad=15)
    #         plt.yticks(fontsize=9)
    #         plt.xticks(fontsize=9)
    #         plt.grid(axis='x', alpha=0.3)
    #         plt.tight_layout(pad=1.5)
    #         return plt.gcf()
        
    #     except Exception as e:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         ax.text(0.5, 0.5, f"⚠️ Permutation Importance 계산 오류:\n{str(e)[:100]}",
    #                 ha="center", va="center", fontsize=10, color='#dc3545')
    #         ax.axis("off")
    #         plt.tight_layout()
    #         return fig