import jittor as jt
jt.flags.use_cuda = 1
from cogdl_jittor import experiment
print("use_cuda", jt.flags.use_cuda)
experiment(dataset="cora", model="gcn") 
experiment(dataset="pubmed", model="graphsage") 
experiment(dataset="cora", model="dgi") 
experiment(dataset="cora", model="grand")
experiment(dataset="cora", model="gcnii") 
experiment(dataset="cora", model="mvgrl") 
experiment(dataset="cora", model="grand") 

from cogdl_jittor.data.jittor.dataloader import Dataloader
#-----------------Attention Bug----------
# experiment(dataset="cora", model="gat") #能运行，几个epoch后loss增加
# experiment(dataset="cora", model="gat",nhead=1) #nhead=1报错
# experiment(dataset="cora", model="drgat")   #loss会出现nan