# Previsão de Preços de Imóveis

## Descrição
Este repositório contém um projeto de previsão de preços de imóveis utilizando técnicas de aprendizado de máquina. O objetivo é desenvolver e avaliar modelos preditivos para estimar o preço de imóveis com base em características como área, número de quartos, banheiros, localização e outras variáveis relevantes. O projeto utiliza o dataset "Housing Price Prediction" disponível no Kaggle e inclui uma análise exploratória de dados (EDA) seguida pela aplicação de modelos de regressão.

**Dataset**: [Housing Price Prediction no Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction/data)

## Estrutura do Projeto
O projeto está organizado da seguinte forma:
- **`dataset/Housing.csv`**: Arquivo com os dados brutos utilizados no projeto.
- **`housing_price_prediction.ipynb`**: Notebook Jupyter contendo o código completo, desde a análise exploratória até a avaliação dos modelos.
- **`api.py`**
- **`predict.py`**
- **`train_housing_price_prediction.py`**
- **`README.md`**: Este arquivo com informações sobre o projeto.

## Dataset
O dataset contém 545 registros e 13 colunas, divididas em variáveis numéricas e categóricas:
- **Numéricas**: `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- **Categóricas**: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

Não há valores nulos no dataset, o que facilita o pré-processamento.

### Estatísticas Básicas
- **Preço (`price`)**: Varia de 1.750.000 a 13.300.000, com média de aproximadamente 4.766.729.
- **Área (`area`)**: Entre 1.650 e 16.200 pés quadrados, com média de 5.150.
- **Quartos (`bedrooms`)**: De 1 a 6, com média de 2,97.
- **Banheiros (`bathrooms`)**: De 1 a 4, com média de 1,29.

## Metodologia
1. **Análise Exploratória de Dados (EDA)**:
   - Verificação de tipos de dados, valores nulos e estatísticas descritivas.
   - Separação entre variáveis numéricas e categóricas.
   - Visualização inicial com `df.head()` e `df.tail()`.

2. **Pré-processamento**:
   - Conversão de variáveis categóricas (`yes/no`) em valores binários (0 e 1) usando `LabelEncoder`.
   - Codificação da variável `furnishingstatus` (furnished, semi-furnished, unfurnished) em valores numéricos.
   - Normalização das variáveis independentes (`X`) e dependente (`y`) para o intervalo [0, 1] com `MinMaxScaler`.

3. **Divisão dos Dados**:
   - 80% para treino e 20% para teste, com embaralhamento (`train_test_split`, `random_state=42`).

4. **Modelos Avaliados**:
   - **Regressão Linear** (`LinearRegression`)
   - **Random Forest** (`RandomForestRegressor`)

5. **Métricas de Avaliação**:
   - Erro Quadrático Médio (MSE)
   - Erro Absoluto Médio (MAE)
   - Coeficiente de Determinação (R²)

## Resultados
Os modelos foram treinados e avaliados com os seguintes resultados no conjunto de teste:

| Modelo            | MSE       | MAE       | R²       |
|-------------------|-----------|-----------|----------|
| Regressão Linear  | 0.01328   | 0.08482   | 0.6495   |
| Random Forest     | 0.01459   | 0.08918   | 0.6149   |

- **Regressão Linear** obteve o melhor desempenho com um R² de 0.6495, indicando que aproximadamente 65% da variabilidade nos preços foi explicada pelo modelo.
- **Random Forest** apresentou um R² ligeiramente inferior (0.6149), mas ainda assim competitivo.

## Conclusão

O projeto demonstrou que é possível prever preços de imóveis com razoável precisão utilizando técnicas simples de regressão. A **Regressão Linear** se destacou como o modelo mais eficaz neste cenário, superando o Random Forest em todas as métricas avaliadas. A análise também reforça a importância do pré-processamento adequado dos dados, especialmente quando se trabalha com variáveis categóricas e escalas diferentes.

## Próximos Passos

Algumas melhorias e expansões que podem ser exploradas no futuro:

- Aplicação de **engenharia de atributos (feature engineering)** para criação de novas variáveis mais informativas.
- Testar outros modelos de regressão como **XGBoost**, **Gradient Boosting** ou **SVR**.
- 
## Como Usar a API
O projeto inclui uma API construída com **FastAPI** que permite realizar previsões de preços de imóveis em tempo real. A API utiliza os modelos treinados (`Regressão Linear` ou `Random Forest`) e retorna o preço estimado com base nos dados fornecidos.

### Pré-requisitos
- Python 3.10+
- Instale as dependências listadas no arquivo `requirements.txt`:
  ```bash
  pip install -r requirements.txt
- Faça um ambiente e rode ele
  ```bash
  source venv/bin/activate  
### Executando a API
1. Certifique-se de que os modelos e scalers estão salvos no diretório model/ (gerados durante o treinamento).
2. Certifique-se de que os modelos e scalers estão salvos no diretório model/ (gerados durante o treinamento).
```bash
 uvicorn api:app --host 127.0.0.1 --port 8000 --reload
 ```
 ou 
```bash
 python api.py
 ```
### Fazendo uma Requisição
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "area": 50,
    "bedrooms": 1,
    "bathrooms": 1,
    "stories": 1,
    "mainroad": "no",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "no",
    "parking": 1,
    "prefarea": "yes",
    "furnishingstatus": "furnished",
    "model_type": "rf"
}'
 ```

 ### Acessar documentação(swagger)
 http://127.0.0.1:8000/docs


#### Usando Docker
Você pode containerizar a API com Docker para uma execução consistente em qualquer ambiente.

1. Certifique-se de ter o Docker instalado.
2. Construa a imagem Docker:
   ```bash
   docker build -t housing-price-api .
2. Execute o container:
3.  ```bash
    docker run -d -p 8000:8000 housing-price-api