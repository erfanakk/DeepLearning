import torch
import utils, engine, model_builder, data_setup
from helper_functions import accuracy_fn
import torch.nn as nn


NUM_EPOCHS = 1
NUM_CLASS = 10
IN_CHANNELS = 1
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"


train_dataloader, test_dataloader, class_names = data_setup.creat_dataset()


model = model_builder.CNN_simple(IN_CHANNELS, NUM_CLASS).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}\n---------")
    if (epoch+1) % 5 == 0:
        checkpoint = {
        'state_dic' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
        }
        utils.save_checkpoint(stete=checkpoint)

    engine.train_step(model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    engine.test_step(model=model, dataloader=test_dataloader,loss_fn=loss_fn, accuracy_fn=accuracy_fn)

# utils.load_checkpoint(torch.load('cnn_simple.pth.tar' , map_location=torch.device('cpu')), model=model, optimizer=optimizer)
# engine.test_step(model=model, dataloader=test_dataloader,loss_fn=loss_fn, accuracy_fn=accuracy_fn)