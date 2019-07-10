import torch
import numpy as np
import pickle
y = torch.load('shufflenetv2_x0.5-f707e7126e.pth')
px = torch.load('save_1.pth')
l = {'arch':'shuffl-en','state_dict':y}
l1 = {'arch':'shuffl-en','state_dict':{}}
# for k,v in pr['state_dict'].items():
#    print(k,v.size())
#for k,v in px['state_dict'].items():
#   print(k,v)
for k,v in l['state_dict'].items():
  #print(k)
  if (len(list(v.size()))==4):
      vx = v.detach().numpy()
      vx = np.stack([vx]*v.size(2),axis=2)/float(v.size(2))
      k = 'module.'+k
      z = torch.Tensor(vx)
      l1['state_dict'].setdefault(k,z)##setdefault更新添加新元素
  elif ('fc' in k):
      continue
  elif ('running_mean' in k) or ('running_var' in k):
      continue
  else:
      k = 'module.'+k
      l1['state_dict'].setdefault(k, v)
torch.save(l1,'shufflenet-imagenet-pretrained.pth')
for k,v in l1['state_dict'].items():
       print(k,v.size())
