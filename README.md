Visão Geral

Este projeto implementa, manualmente e sem dependência de bibliotecas externas como NumPy, Pandas ou Scikit-learn, um sistema completo de detecção de multicolinearidade em datasets tabulares.

O código implementa:

Correlação de Pearson,
Correlação de Spearman,
Regressão Linear via Equação Normal,
Inversão de Matrizes com Gauss-Jordan,
Variance Inflation Factor (VIF),
Tolerância,
Estrutura simples de Dataset CSV,

O objetivo do projeto é:

Demonstrar os fundamentos matemáticos da multicolinearidade.
Implementar álgebra linear e estatística “from scratch”.
Servir como biblioteca educacional e de aprendizado profundo em:
Estatística
Econometria
Machine Learning
Álgebra Linear Computacional

| Classe                      | Responsabilidade                  |
| --------------------------- | --------------------------------- |
| `Dataset`                   | Leitura e validação dos dados     |
| `Statistics`                | Estatística básica                |
| `Matrix`                    | Álgebra linear                    |
| `LinearRegression`          | Regressão Linear                  |
| `MulticollinearityDetector` | Diagnóstico de multicolinearidade |



