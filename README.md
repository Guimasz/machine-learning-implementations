# Machine Learning Implementations

Este repositório contém implementações de algoritmos de Machine Learning desenvolvidas como parte dos estudos baseados no curso de Machine Learning de Stanford.

## Linear Regression Model

A implementação atual contém um modelo básico de regressão linear utilizando o algoritmo de gradient descent. Esta implementação demonstra como:

- Preparar dados de entrada e saída
- Inicializar parâmetros do modelo (weight e bias)
- Definir funções de previsão e custo
- Calcular gradientes para atualização dos parâmetros
- Treinar um modelo usando gradient descent
- Fazer previsões com o modelo treinado

### Características do código:
- Implementação de regressão linear simples (y = w*x + b)
- Função de custo utilizando erro quadrático médio (MSE)
- Atualização de parâmetros com gradient descent
- Monitoramento de progresso durante o treinamento

## Detalhes da Implementação

O modelo implementado no arquivo `Main.py` busca encontrar os parâmetros de uma função linear `y = wx + b` que melhor se ajustam aos dados de treinamento. No exemplo atual, o algoritmo trabalha com um conjunto de dados simples onde a relação verdadeira é `y = 2x + 1`.

### Componentes principais:
- **Inicialização**: Os parâmetros w (weight) e b (bias) são inicializados com 0
- **Previsão**: Usa a fórmula linear `y = wx + b`
- **Função de Custo**: Média do erro quadrático (MSE)
- **Gradientes**: Derivadas parciais da função de custo em relação a w e b
- **Atualização**: Utiliza gradient descent com taxa de aprendizado de 0.01

## Instruções de Execução

Para executar o modelo de regressão linear:

1. Certifique-se de ter Python 3.x instalado
2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```
   ou instale diretamente:
   ```bash
   pip install numpy
   ```
3. Execute o script principal:
   ```bash
   python Main.py
   ```

### Saída esperada:
O programa mostrará o progresso do treinamento a cada 100 epochs (sessões de treino), exibindo:
- O número da epoch
- O valor da função de custo
- Os valores atuais de w e b

Ao final, o modelo fará uma previsão para x=12, que deve se aproximar de y=25 (já que a relação implícita nos dados de treinamento é y=2x+1).

## Modificando o Modelo

Você pode experimentar com o modelo alterando:
- Os dados de entrada e saída (arrays X e Y)
- A taxa de aprendizado (learning_rate)
- O número de épocas (epochs)
- O valor de teste para previsão no final do script

