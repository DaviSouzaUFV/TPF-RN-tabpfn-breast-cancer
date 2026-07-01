# TPF-02 — Aprimoramento do TabPFN em Classificação Tabular

Reprodução (TPF-01) e aprimoramento (TPF-02) do artigo *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second* (Hollmann et al., 2022/2023), aplicado ao **Breast Cancer Wisconsin Dataset**.

## Resumo

- **TPF-01**: reprodução do TabPFN em modo *inference-only* (in-context learning), com *split* único 80/20, obtendo 98,25% de acurácia e AUC 1,00.
- **TPF-02**: duas modificações sobre o ambiente experimental original:
  1. **Redução de dimensionalidade via PCA** (30 → 10 atributos, retendo 95% da variância).
  2. **Validação cruzada estratificada (10-fold)** em vez de um único *split*, para uma estimativa mais robusta de generalização.

## Resultados

| Configuração | Acurácia | AUC | F1-score | Tempo/execução (s) |
|---|---|---|---|---|
| TPF-01 (split único, 30 attrs) | 0,9825 | 0,9967 | 0,9861 | 2,03 |
| TPF-02 (10-fold CV, 30 attrs) | 0,9807 ± 0,0123 | 0,9971 ± 0,0054 | 0,9847 ± 0,0098 | 2,30 |
| TPF-02 (10-fold CV, PCA 10 attrs) | 0,9701 ± 0,0208 | 0,9956 ± 0,0055 | 0,9762 ± 0,0171 | 2,12 |

Ver o relatório completo (`relatorio/TPF02_Relatorio.pdf`) para discussão detalhada.

## Estrutura do repositório

```
.
├── notebooks/
│   └── tpf02_experimento.py     # Script completo (baseline + PCA + 10-fold CV)
├── resultados/
│   └── tpf02_resultados_resumo.csv
├── relatorio/
│   ├── TPF02_Relatorio.pdf
│   └── latex/                   # Fonte LaTeX (formato NeurIPS)
│       ├── main.tex
│       └── neurips_2024.sty
└── README.md
```

## Como executar

1. Abra o Google Colab e habilite GPU (`Ambiente de execução > Alterar tipo de ambiente de execução > GPU`).
2. Instale as dependências:
   ```bash
   pip install tabpfn scikit-learn pandas numpy matplotlib
   ```
3. Autentique-se com a HuggingFace Hub e aceite os termos do modelo TabPFN em <https://huggingface.co/Prior-Labs/tabpfn_3> (necessário apenas na primeira execução):
   ```bash
   hf auth login
   ```
4. Execute o script `notebooks/tpf02_experimento.py` (pode ser colado célula por célula em um notebook Colab — já está dividido em blocos `# %%`).
5. O script gera automaticamente:
   - `tpf02_resultados_resumo.csv` — tabela comparativa dos resultados
   - `pca_variancia.png`, `tpf02_comparacao.png`, `tpf02_estabilidade_folds.png` — figuras usadas no relatório

## Dataset

Breast Cancer Wisconsin Dataset (`sklearn.datasets.load_breast_cancer`): 569 amostras, 30 atributos numéricos, classificação binária (benigno/maligno).

## Referências

- Hollmann, N., Müller, S., Eggensperger, K., Hutter, F. *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*. ICLR, 2023. [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
- Prior Labs. [TabPFN Official Repository](https://github.com/PriorLabs/TabPFN)
