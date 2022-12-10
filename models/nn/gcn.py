
import jittor as jt
jt.flags.use_cuda = 1
jt.flags.use_device = 1
from jittor import nn, Module, init
from cogdl_jittor.models import BaseModel
from jittor import optim
from jittor.contrib import slice_var_index

from tqdm import tqdm

from cogdl_jittor.layers import GCNLayer
from cogdl_jittor.datasets.planetoid_data import CoraDataset


class GCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--norm", type=str, default=None)
        parser.add_argument("--activation", type=str, default="relu")
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.num_layers,
            args.dropout,
            args.activation,
            args.residual,
            args.norm,
        )


    def __init__(self, in_feats, hidden_size, out_feats, dropout=0.5):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.conv1 = GCNLayer(in_feats, hidden_size, dropout=dropout, activation="relu")
        self.conv2 = GCNLayer(hidden_size, out_feats)

    def execute(self, graph):
        graph.sym_norm()
        x = graph.x
        out = self.conv1(graph, x)
        out = self.conv2(graph, out)
        return out


def train(model, dataset):
    graph = dataset[0]

    optimizer = nn.AdamW(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    val_mask = graph.val_mask
    labels = graph.y
    
    for epoch in range(200):
        model.train()
        output = model(graph)
        loss = loss_function(output[train_mask], labels[train_mask])
        optimizer.step(loss)
        
        model.eval()
        with jt.no_grad():
            pred = model(graph)
            pred = pred.argmax(1)[0]
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        print(f"Epoch:{epoch}, loss:{loss:.3f}, train_acc:{train_acc:.3f}, val_acc:{val_acc:.3f}, test_acc:{test_acc:.3f}")

if __name__ == "__main__":
    dataset = CoraDataset()
    model = GCN(in_feats=dataset.num_features, hidden_size=64, out_feats=dataset.num_classes, dropout=0.5)

    train(model, dataset)
