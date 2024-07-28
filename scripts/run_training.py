from src.models.neural_network import NeuralNetwork
from src.data.data_loader import load_data

def train():
    X, y = load_data()
    model = NeuralNetwork()
    
    for epoch in range(10000):
        loss = model.forward(X, y)
        model.backward(X, y)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == "__main__":
    train()