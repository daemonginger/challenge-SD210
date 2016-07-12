import numpy as np
import pandas as pd
import sys

y_pred_valid = np.zeros(129715)
for i in range(1,int(sys.argv[1])+1):
	if(i != 3 and i != 8):
		y_pred_valid += np.loadtxt('../sub_partial/'+sys.argv[2]+'_'+sys.argv[3]+'_'+str(i))

y_pred_valid = y_pred_valid/(int(sys.argv[1])-2)
np.savetxt('../sub/'+sys.argv[2]+'_'+sys.argv[3],y_pred_valid)

