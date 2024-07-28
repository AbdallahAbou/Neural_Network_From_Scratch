from src.models.neural_network import NeuralNetwork
from src.data.data_loader import sine

def train():
    X, y = sine.create_data()
    model = NeuralNetwork()
    
    for epoch in range(10000):
        loss = model.forward(X, y)
        model.backward(y)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == "__main__":
    train()