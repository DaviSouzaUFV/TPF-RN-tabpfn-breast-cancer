# =============================================================================
# TPF-02 - Aprimoramento do modelo TabPFN
# Modificação: Redução de dimensionalidade (PCA) + validação cruzada robusta
#              (Stratified K-Fold) para avaliar generalização e complexidade
# Cole este script em células do Colab (já dividido em blocos com "# %%")
# =============================================================================

# %% [1] Instalação (rode apenas se necessário)
# !pip install tabpfn scikit-learn pandas numpy matplotlib -q

# %% [2] Imports
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from tabpfn import TabPFNClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [3] Carregar dados (idêntico ao TPF-01)
X, y = load_breast_cancer(return_X_y=True)
print(f"Amostras: {X.shape[0]}, Atributos: {X.shape[1]}")

# %% [4] BASELINE - reprodução exata do TPF-01 (split único 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf_baseline = TabPFNClassifier()
t0 = time.time()
clf_baseline.fit(X_train, y_train)
preds = clf_baseline.predict(X_test)
probs = clf_baseline.predict_proba(X_test)[:, 1]
t_baseline = time.time() - t0

acc_baseline = accuracy_score(y_test, preds)
auc_baseline = roc_auc_score(y_test, probs)
f1_baseline = f1_score(y_test, preds)

print("\n=== BASELINE (TPF-01, 30 atributos, split único) ===")
print(f"Acurácia: {acc_baseline:.4f} | AUC: {auc_baseline:.4f} | F1: {f1_baseline:.4f}")
print(f"Tempo fit+predict: {t_baseline:.2f}s")
print("Matriz de confusão:\n", confusion_matrix(y_test, preds))

# %% [5] Determinar nº de componentes PCA para reter ~95% da variância
scaler_full = StandardScaler().fit(X)
X_scaled_full = scaler_full.transform(X)
pca_probe = PCA(random_state=RANDOM_STATE).fit(X_scaled_full)
cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
n_components = int(np.argmax(cum_var >= 0.95) + 1)
print(f"\nComponentes PCA necessários p/ reter 95% da variância: {n_components} "
      f"(redução de {X.shape[1]} -> {n_components} atributos, "
      f"{100*(1 - n_components/X.shape[1]):.1f}% menos dimensões)")

# Gráfico de variância explicada acumulada (usar no relatório)
plt.figure(figsize=(5, 3.5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", ms=3)
plt.axhline(0.95, color="red", linestyle="--", label="95% variância")
plt.axvline(n_components, color="gray", linestyle=":")
plt.xlabel("Nº de componentes principais")
plt.ylabel("Variância explicada acumulada")
plt.title("PCA - Variância explicada")
plt.legend()
plt.tight_layout()
plt.savefig("pca_variancia.png", dpi=150)
plt.show()

# %% [6] Validação cruzada robusta: COM PCA vs SEM PCA (mesmo protocolo)
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def run_cv(X_data, use_pca, n_components=None):
    accs, aucs, f1s, times = [], [], [], []
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_data, y)):
        X_tr, X_te = X_data[tr_idx], X_data[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # padronizar (fit somente no treino, para evitar vazamento de dados)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        if use_pca:
            pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_tr_s)
            X_tr_s, X_te_s = pca.transform(X_tr_s), pca.transform(X_te_s)

        clf = TabPFNClassifier()
        t0 = time.time()
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        probs = clf.predict_proba(X_te_s)[:, 1]
        elapsed = time.time() - t0

        accs.append(accuracy_score(y_te, preds))
        aucs.append(roc_auc_score(y_te, probs))
        f1s.append(f1_score(y_te, preds))
        times.append(elapsed)
        print(f"  Fold {fold_i+1}/{N_FOLDS} - acc={accs[-1]:.4f} auc={aucs[-1]:.4f} t={elapsed:.2f}s")

    return {"accuracy": accs, "auc": aucs, "f1": f1s, "time": times}

print(f"\n=== CV 10-fold SEM PCA (30 atributos) ===")
results_full = run_cv(X, use_pca=False)

print(f"\n=== CV 10-fold COM PCA ({n_components} atributos) ===")
results_pca = run_cv(X, use_pca=True, n_components=n_components)

# %% [7] Resumo comparativo
def summarize(name, res):
    return {
        "Configuração": name,
        "Acurácia (média±dp)": f"{np.mean(res['accuracy']):.4f} ± {np.std(res['accuracy']):.4f}",
        "AUC (média±dp)": f"{np.mean(res['auc']):.4f} ± {np.std(res['auc']):.4f}",
        "F1 (média±dp)": f"{np.mean(res['f1']):.4f} ± {np.std(res['f1']):.4f}",
        "Tempo médio/fold (s)": f"{np.mean(res['time']):.2f}",
    }

summary_df = pd.DataFrame([
    {"Configuração": "TPF-01 (split único, 30 attrs)", "Acurácia (média±dp)": f"{acc_baseline:.4f}",
     "AUC (média±dp)": f"{auc_baseline:.4f}", "F1 (média±dp)": f"{f1_baseline:.4f}",
     "Tempo médio/fold (s)": f"{t_baseline:.2f}"},
    summarize("TPF-02 CV 10-fold, 30 attrs (sem PCA)", results_full),
    summarize(f"TPF-02 CV 10-fold, {n_components} attrs (PCA 95%)", results_pca),
])
print("\n=== TABELA RESUMO (usar no relatório) ===")
print(summary_df.to_string(index=False))
summary_df.to_csv("tpf02_resultados_resumo.csv", index=False)

# %% [8] Gráfico comparativo (barras com erro) - usar no relatório
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

labels = ["30 attrs\n(sem PCA)", f"{n_components} attrs\n(PCA 95%)"]
acc_means = [np.mean(results_full["accuracy"]), np.mean(results_pca["accuracy"])]
acc_stds = [np.std(results_full["accuracy"]), np.std(results_pca["accuracy"])]
axes[0].bar(labels, acc_means, yerr=acc_stds, capsize=5, color=["#4C72B0", "#DD8452"])
axes[0].axhline(acc_baseline, color="green", linestyle="--", label="TPF-01 (split único)")
axes[0].set_ylabel("Acurácia")
axes[0].set_title("Acurácia (10-fold CV)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.9, 1.0)

time_means = [np.mean(results_full["time"]), np.mean(results_pca["time"])]
time_stds = [np.std(results_full["time"]), np.std(results_pca["time"])]
axes[1].bar(labels, time_means, yerr=time_stds, capsize=5, color=["#4C72B0", "#DD8452"])
axes[1].set_ylabel("Tempo (s) por fold")
axes[1].set_title("Custo computacional (fit+predict)")

plt.tight_layout()
plt.savefig("tpf02_comparacao.png", dpi=150)
plt.show()

# %% [9] "Curva de aprendizado" - variação da acurácia por fold (mostra estabilidade/generalização)
plt.figure(figsize=(6, 3.8))
plt.plot(range(1, N_FOLDS + 1), results_full["accuracy"], marker="o", label="30 attrs (sem PCA)")
plt.plot(range(1, N_FOLDS + 1), results_pca["accuracy"], marker="s", label=f"{n_components} attrs (PCA)")
plt.axhline(acc_baseline, color="green", linestyle="--", alpha=0.6, label="TPF-01 (split único)")
plt.xlabel("Fold")
plt.ylabel("Acurácia")
plt.title("Estabilidade da acurácia entre folds (generalização)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("tpf02_estabilidade_folds.png", dpi=150)
plt.show()

print("\nArquivos gerados: pca_variancia.png, tpf02_comparacao.png, "
      "tpf02_estabilidade_folds.png, tpf02_resultados_resumo.csv")
print("Baixe esses arquivos do Colab (aba de arquivos) para usar no relatório.")
