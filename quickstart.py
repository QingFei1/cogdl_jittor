import jittor as jt
jt.flags.use_cuda = 1
from cogdl_jittor import experiment
experiment(dataset="cora", model="gcn") 
experiment(dataset="cora", model="graphsage") 
experiment(dataset="cora", model="dgi") 
experiment(dataset="cora", model="grand")
experiment(dataset="cora", model="gcnii") 
experiment(dataset="cora", model="mvgrl") 
experiment(dataset="cora", model="grand") 



#-----------------Attention Bug----------
# experiment(dataset="cora", model="gat",nhead=1) #nhead=1报错
# experiment(dataset="cora", model="drgat")   #loss会出现nan