import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformer import Encoder
from sequence_gen import seq_gen_train, seq_gen_test
import random


# 训练和测试函数
def train(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for i, (seq, target, mask) in enumerate(dataloader):
        seq, target, mask = seq.to(device), target.to(device), mask 
        optimizer.zero_grad()
        src_mask = None

        output = model(seq, src_mask)
        # seq_array = seq.cpu().numpy()

        mask_expanded = mask.unsqueeze(-1)
        output = output * mask_expanded.to(device)
        target = target * mask_expanded.to(device)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()


        target_max = torch.argmax(target, dim=-1)
        preds = torch.argmax(output, dim=-1)
        # print(target_max)
        # print(preds)
        print("##################################")
        print(target_max)
        print(preds)
        target_max = torch.argmax(target, dim=-1)
        temp = (preds == target_max) * mask.to(device)
        correct += (temp).sum().item()


        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(dataloader) + i)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset) * target.size(1) * 0.5)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)

    return avg_loss, accuracy

def test(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for seq, target, mask in dataloader:
            seq, target, mask = seq.to(device), target.to(device), mask 
            src_mask = None
            output = model(seq, src_mask)
            mask_expanded = mask.unsqueeze(-1)

            # output_temp = output 
            output = output * mask_expanded.to(device)
            target = target * mask_expanded.to(device)


            loss = criterion(output, target)
            total_loss += loss.item()


            preds = torch.argmax(output, dim=-1)
 
            target_max = torch.argmax(target, dim=-1)
            temp = (preds == target_max) * mask.to(device)
            correct += (temp).sum().item()
        #    print("##################################")
        #     print(target_max)
        #     print(preds) 
   


    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset) * target.size(1) * 0.5)
    writer.add_scalar('Test/Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)

    return avg_loss, accuracy

# # # 自定义数据集类
# class RandomDataset(Dataset):
#     def __init__(self, num_samples, num_classes=100):
#         self.num_samples = num_samples
#         self.num_classes = num_classes
#         self.sequences = [self._generate_sequence() for _ in range(num_samples)]
    
#     def _generate_sequence(self):
#         seq = [random.randint(10, 90) for _ in range(9)]
#         seq_one_hot = one_hot_encode(seq, self.num_classes)
#         return seq, seq_one_hot
    
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         seq, seq_one_hot = self.sequences[idx]
#         return torch.tensor(seq, dtype=torch.long), torch.tensor(seq_one_hot, dtype=torch.float)


def one_hot_encode(sequence, num_classes):
    one_hot_encoded = np.zeros((len(sequence), num_classes))
    for idx, val in enumerate(sequence):
        one_hot_encoded[idx, val - 1] = 1
    return one_hot_encoded

# 自定义数据集类
class TrainDataset(Dataset):
    def __init__(self, num_samples, num_classes=100):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sequences= self._generate_sequence(num_samples)
        # print(self.sequences)

    def _generate_sequence(self, num_samples):
        seq = seq_gen_train(num_samples)
        return seq
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq, seq_one_hot, masks = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(seq_one_hot, dtype=torch.float), torch.tensor(masks, dtype=torch.float)
    
class TestDataset(Dataset):
    def __init__(self, num_samples, num_classes=100):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sequences= self._generate_sequence(num_samples)
        # print(self.sequences)

    def _generate_sequence(self, num_samples):
        seq = seq_gen_test(num_samples)
        return seq
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq, seq_one_hot, masks = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(seq_one_hot, dtype=torch.float), torch.tensor(masks, dtype=torch.float)

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_voc_size = 10000
    max_len = 9
    d_model = 512
    ffn_hidden = 128
    n_head = 4
    n_layers = 4
    drop_prob = 0.1
    num_epochs = 4000
    batch_size = 100
    learning_rate = 2e-5

    model = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TrainDataset(num_samples=3000)
    test_dataset = TestDataset(num_samples=3000)
    # train_size = int(1 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter()
    best_accuracy = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        test_loss, test_accuracy = test(model, test_loader, criterion, device, writer, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    writer.close()

if __name__ == "__main__":
    main()