import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import seaborn as sns
from PIL import Image
#%matplotlib inline

class Visualizer():
    def __init__(self,controller):
        print("##INIT ",self.__class__.__name__)
    def show_loss(self):
        import warnings;warnings.filterwarnings('ignore')
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=1) # put ticks at regular intervals
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        #plt.show()
        plt.title(title)
        plt.savefig(path)
        plt.close()

    def show_attention(self):
        import warnings;warnings.filterwarnings('ignore')

        if not os.path.exists(path):
            os.makedirs(path)

        title=title.replace(" @@","").replace("@@ ","")
        fname=title+'.pdf'

        if len(list(fname))>255:
            title=str(len(list(fname)))+"too long txt"
            fname=os.path.join(path,title+'.pdf')
        else:
            fname=os.path.join(path,fname)

        plt.rcParams['font.family'] = 'IPAPGothic'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(attentions,aspect='auto', cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        #import pdb; pdb.set_trace()
        if label_x is not None and isinstance(label_x[0],str):
            ax.set_xticklabels([''] + label_x)
            ax.xaxis.set_ticks_position("top")
        else:
            ax.tick_params(labelbottom="off",bottom="off") # x軸の削除

        if label_y is not None and isinstance(label_y[0],str):
            ax.set_yticklabels([''] + label_y)
        else:
            ax.tick_params(labelleft="off",left="off") # y軸の削除
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.tight_layout()
        plt.savefig(fname)
        plt.close()

    def show_spectrogram(self):
        if not os.path.exists(path):
            os.makedirs(path)
        sns.reset_orig()
        plt.figure(figsize=(14, 6))
        plt.imshow(spec)
        if text:
            plt.title(text, fontsize='10')
        plt.colorbar(shrink=0.5, orientation='horizontal')
        plt.ylabel('mels')
        plt.xlabel('frames')
        title=text.replace(" @@","").replace("@@ ","")

        fname=os.path.join(path,title+".pdf")
        plt.savefig(fname)
        plt.close()
