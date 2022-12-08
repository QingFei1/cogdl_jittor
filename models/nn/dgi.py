import os 
os.environ['nvcc_path']="/usr/local/cuda-11.4/bin/nvcc"
import numpy as np
import jittor
jittor.flags.use_cuda = 1
# jittor.flags.use_device = 2
from jittor import nn,Module
from cogdl_jittor.models import BaseModel
from cogdl_jittor.utils import get_activation, spmm
from cogdl_jittor.datasets.planetoid_data import CoraDataset

# Borrowed from https://github.com/PetarV-/DGI
class GCN(Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else get_activation(act)

        if bias:
            self.bias = jittor.array(out_ft,dtype=jittor.float32)
            self.bias.data[0]= 0.0
        else:
            self.register_parameter("bias", None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            jittor.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def execute(self, graph, seq, sparse=False):
        seq_fts = self.fc(seq)
        if len(seq_fts.shape) > 2:
            if sparse:
                out = jittor.unsqueeze(spmm(graph, jittor.squeeze(seq_fts, 0)), 0)
            else:
                out = jittor.bmm(graph, seq_fts)
        else:
            if sparse:
                if list(seq_fts.shape)[0]==1:
                    seq_fts = jittor.squeeze(seq_fts, 0)
                out = spmm(graph, seq_fts)
            else:
                out = jittor.mm(graph, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class DGIModel(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--activation", type=str, default="prelu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.activation)

    def __init__(self, in_feats, hidden_size, activation):
        super(DGIModel, self).__init__()
        self.gcn = GCN(in_feats, hidden_size, activation)
        self.sparse = True

    def execute(self, graph):
        graph.sym_norm()
        x = graph.x
        logits = self.gcn(graph, x, self.sparse)
        return logits

    # Detach the return variables
    def embed(self, data):
        h_1 = self.gcn(data, data.x, self.sparse)
        return h_1.detach()



def train(model, dataset):
    graph = dataset[0]

    optimizer = nn.AdamW(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    val_mask = graph.val_mask
    labels = graph.y
    
    for epoch in range(50):
        model.train()
        output = model(graph)
        loss = loss_function(output[train_mask], labels[train_mask])
        optimizer.step(loss)
        
        model.eval()
        with jittor.no_grad():
            pred = model(graph)
            pred = pred.argmax(1)[0]
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        print(f"Epoch:{epoch}, loss:{loss:.3f}, train_acc:{train_acc:.3f}, val_acc:{val_acc:.3f}, test_acc:{test_acc:.3f}")

if __name__ == "__main__":
    dataset = CoraDataset()
    model = DGIModel(in_feats=dataset.num_features, hidden_size=512, activation="prelu")

    train(model, dataset)