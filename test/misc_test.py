# from __future__ import print_function
# import torch
# import os
# import numpy as np
# from collections import OrderedDict
# import itertools
#
# pred_list = [0,1,2,3,4,5]
# data = set(itertools.combinations(pred_list, 2))
# it = (0,1)
# print(type(it))
# if it in data:
#     print("yes")
#
# def addone(x):
#     return x + 1
#
# l = OrderedDict()
# l['a'] = addone
# y = l['a'](3)
# print(y)
#
# x = torch.rand(5, 3)
# print(x)
#
# path = '/home/User/Documents/file.txt'
#
# # Above specified path
# # will be splited into
# # (head, tail) pair as
# # ('/home/User/Documents', 'file.txt')
#
# # Get the base name
# # of the specified path
# basename = os.path.basename(path)
#
# # Print the basename name
# print(basename)
import torch
import torch.nn.functional as F

# data = torch.ones(4, 4)
# # pad(left, right, top, bottom)
# new_data = F.pad(input=data, pad=[1, 0, 0, 0], mode='constant', value=0)
# new_new_data = F.pad(input=new_data, pad=[0, 0, 0, 1], mode='constant', value=0)
# print(new_data)
# print(new_new_data)
input = [1,2,3,4,5]
print(len(input[0:3]))