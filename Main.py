import numpy as np


#Array de dados no eixo x (entradas) e eixo y (resultados corretos para as entradas)

X = np.array([1,2,3,4], dtype=float)
Y = np.array([3,5,7,9], dtype=float)

#Inicializando parâmetros
w = 0.0 # weight (coeficiente angular)
b = 0.0 # bias (coeficiente linear/intercepto)
learning_rate = 0.01 #Valor que determina o quanto o valor de custo vai atualizar a cada aprendizado
epochs = 2000 # Número de iterações de aprendizado

#Função de previsão
def predict (x, w, b):
    return w * x +b 

# Função de custo (determina a média entre os erros entre a previsão e o resultado esperado para aqueles parâmetros)
def compute_cost(x, y, w, b):
    m = len(x)
    return (1/m) * np.sum( (predict(x, w, b) - y) **2)

#Função que calcula em que direcao o modelo deve se mover para diminuir o custo

def compute_gradients(x, y, w, b):
    m = len(x)
    y_pred = predict(x, w, b) #lista de previsões
    dw = (2/m)* np.sum((y_pred-y)*x) #derivada da funcao de custo  d/dw 
    db = (2/m) * np.sum(y_pred - y) #derivada da funcao de custo d/db
    
    return dw,db #gradientes 

# Loop de treinamento do modelo utilizando "gradient descent"
for epoch in range(epochs):
    dw, db = compute_gradients(X, Y, w, b) #Usa os X e Y do programa
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if epoch % 100 == 0:
        cost = compute_cost(X, Y, w, b)
        print(f"Treino número {epoch}: Custo = {cost:.4f}, w = {w:.4f}, b = {b:.4f} ")
    
print("Modelo foi treinado corretamente ✅")    

#Teste o modelo alterando o valor de x, para prever o seu y
x = 12

print(f"Predição para x = {x}:", predict(x, w, b)) #O modelo deve chegar a conclusao que y = 2x+1
    