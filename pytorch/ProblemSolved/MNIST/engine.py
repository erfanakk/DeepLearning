import torch
from helper_functions import accuracy_fn
device = "cuda" if torch.cuda.is_available() else "cpu"





def train_step(model: torch.nn.modules,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.modules,
                accuracy_fn,
                device: torch.device= device):
    global step , clss
    model.train()
    train_acc , train_loss = 0,0
    
    for batch , (X,y) in enumerate(dataloader):
        X, y = X.to(device) , y.to(device)
        #X = X.view((X.shape[0] , -1))

        y_pred = model.forward(X)
        
        loss = loss_fn(y_pred,y)
        train_loss += loss
        acc =accuracy_fn(y_true=y,
                y_pred=y_pred.argmax(dim=1))
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_acc /= len(dataloader)
    train_loss /= len(dataloader)
    print(print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%"))



def test_step(model: torch.nn.modules,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.modules,
                accuracy_fn,
                device: torch.device= device):

    model.eval()
    test_acc , test_loss = 0,0
    with torch.inference_mode():
        for batch , (X,y) in enumerate(dataloader):
            X, y = X.to(device) , y.to(device)

            y_pred = model.forward(X)
            loss = loss_fn(y_pred,y)
            test_loss += loss

            test_acc += accuracy_fn(y_true=y,
                    y_pred=y_pred.argmax(dim=1))


        test_acc /= len(dataloader)
        test_loss /= len(dataloader)
        print(print(f"test loss: {test_loss:.5f} | test accuracy: {test_acc:.2f}%"))
