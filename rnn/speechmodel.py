class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  
        self.i2o = nn.Linear(input_size + hidden_size, output_size)   
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



def train(input_tensor, target_tensor, rnn, optimizer, criterion):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden)
        l = criterion(output, target_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_tensor.size(0)




