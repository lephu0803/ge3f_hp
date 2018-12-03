import glob
import os
import mpkit

a = mpkit.

pretrained_path = glob.glob('/home/ge3f/Documents/GE3F/HP_project/ColorDb/model/*')
lastest_model = max(pretrained_path, key=os.path.getctime)
print(lastest_model)