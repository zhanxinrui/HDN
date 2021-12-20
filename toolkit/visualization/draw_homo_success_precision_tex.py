import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import os
from .draw_utils import COLOR, LINE_STYLE, COLOR2, LINE_STYLE2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
font1 = {
    'family':'sans-serif',
    # 'sans-serif':['Helvetica']
    # 'family': 'TImes New Roman',
    'weight': 'normal',
    'size': 20

}

def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
                           norm_precision_ret=None, centroid_precision_ret=None, robustness_ret= None,
                           bold_name=None, axis=[0, 1]):
    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold',font1)
    plt.ylabel('Success rate',font1)
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots on %s}' % (name),fontdict={"size":20})
    else:
        plt.title(r'\textbf{Success plots - %s}' % (attr),fontdict={"size":20})
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name

        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                 color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2, fontsize=14)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    if not os.path.exists('./figs/%s'%name):
        os.makedirs('./figs/%s/'%name)
    if attr == 'ALL':
        fig.savefig('./figs/%s/%s-overlap.pdf'%(name, name),format='pdf', bbox_inches='tight')
    else:
        fig.savefig('./figs/%s/%s-overlap-%s.pdf'%(name, name, attr),format='pdf',bbox_inches='tight')
    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold', font1)
        plt.ylabel('Precision',font1)
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of on %s}' % (name), fontdict={"size":20})
        else:
            plt.title(r'\textbf{Precision plots of - %s}' % (attr),fontdict={"size":20})
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(np.mean(value, axis=0))
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2, fontsize=14)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
        if not os.path.exists('./figs/%s'%name):
            os.makedirs('./figs/%s/'%name)
        if attr == 'ALL':
            fig.savefig('./figs/%s/%s-precision.pdf'%(name, name),format='pdf',bbox_inches='tight')
        else:
            fig.savefig('./figs/%s/%s-precision-%s.pdf'%(name, name, attr),format='pdf',bbox_inches='tight')
    #Fig. 6 in paper
    if robustness_ret:
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(1)
        plt.xlabel('Trajectories Length',font1)
        plt.ylabel('Trajectories Ratio',font1)
        if attr == 'ALL':
            plt.title(r'\textbf{Robustness plots on %s}' % (name),fontdict={"size":20})
            # plt.title('Success plots of OPE on %s' % (name))
        else:
            plt.title(r'\textbf{Robustness plots - %s}' % (attr),fontdict={"size":20})
            # plt.title('Success plots of OPE - %s' % (attr))
        plt.axis([0, 100]+axis)
        robustness = {}
        robustness_5 = {}
        # robustness_494 = {}
        robustness_494_501 = {}
        thresholds = np.arange(0, 502, 1)
        indexes_to_keep = []


        for tracker_name in robustness_ret.keys():
            value = [v for k, v in robustness_ret[tracker_name].items() if k in videos]
            robustness[tracker_name] = np.mean(value, axis=0)[40]
            rob_mean = np.mean(value, axis=0)
            robustness[tracker_name] = rob_mean[20]
            robustness_5[tracker_name] = rob_mean[5]
            robustness_494_501[tracker_name] = rob_mean[501] - rob_mean[494]

        for idx, (tracker_name, auc) in \
                enumerate(sorted(robustness.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
            else:
                label = "[%.3f] " % (auc) + tracker_name
            value = [v for k, v in robustness_ret[tracker_name].items() if k in videos]
            y = np.mean(value, axis=0)
            y_to_use = [y[nb] for nb in indexes_to_keep]
            plt.plot(thresholds, y,
                     color=COLOR[idx%10], linestyle=LINE_STYLE2[idx%10],label=label, linewidth=2)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='lower center', labelspacing=0.2, fontsize=14)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.yticks(np.arange(ymin, ymax, 0.1))
        plt.grid(False)
        ax.set_aspect('auto')
        if not os.path.exists('./figs/%s'%name):
            os.makedirs('./figs/%s/'%name)
        if attr == 'ALL':
            fig.savefig('./figs/%s/%s-robustness.pdf'%(name, name),format='pdf', bbox_inches='tight')
        else:
            fig.savefig('./figs/%s/%s-robustness-%s.pdf'%(name, name, attr),format='pdf',bbox_inches='tight')


        # # next plot the group bar
        # Histogram  Fig. 6 in Paper.
        # labels = ['[0, 5]','[495, 501]']
        # fig,ax = plt.subplots()
        # x = np.arange(len(labels))
        # width = 0.1
        # tracker_names = ['Ours','LISRD','GOP-ESM','LDES']
        # # tracker_names = ['Ours']
        # robustness_0_5_l = [ robustness_5[tracker_names[i]] for i in range(len(tracker_names))]
        # robustness_495_501_l = [robustness_494_501[tracker_names[i]] for i in range(len(tracker_names))]
        # print('roubustness_5', robustness_0_5_l, robustness_495_501_l)
        # rects = []
        # for r_idx in range(len(tracker_names)):
        #     rects.append(ax.bar(x-2*width + width*r_idx, [robustness_0_5_l[r_idx], robustness_495_501_l[r_idx]], width, label=tracker_names[r_idx]))
        #
        # # ax.set_ylabel('ratio', )
        # plt.xlabel('Trajectories length range',font1)
        # plt.ylabel('Ratio',font1)
        # ax.set_title(r'\textbf{Ratio by trajectory length range}', fontdict={"size":20})
        # ax.set_xticks(x)
        # ax.set_xticklabels(labels, fontdict={"size":14})
        # ax.set_yticks(np.arange(0,1, 0.1))
        # # plt.yticks(np.arange(0,1, 0.1))
        # ax.legend(labelspacing=0.2, fontsize=14)
        #
        # ax.tick_params(bottom=False)
        #
        # for r_idx in range(len(tracker_names)):
        #     ax.bar_label(rects[r_idx], padding=3)
        #     ax.bar_label(rects[r_idx], padding=3)
        #
        # fig.tight_layout()
        #
        # plt.show()
        # if not os.path.exists('./figs/%s'%name):
        #     os.makedirs('./figs/%s/'%name)
        # if attr == 'ALL':
        #     fig.savefig('./figs/%s/%s-robustness-bar.pdf'%(name, name),format='pdf', bbox_inches='tight')
        # else:
        #     fig.savefig('./figs/%s/%s-robustness-bar-%s.pdf'%(name, name, attr),format='pdf',bbox_inches='tight')


    if centroid_precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold',font1)
        plt.ylabel('Centroid precision',font1)
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots on %s}' % (name),fontdict={"size":20})
        else:
            plt.title(r'\textbf{Precision plots - %s}' % (attr),fontdict={"size":20})
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in centroid_precision_ret.keys():
            value = [v for k, v in centroid_precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(np.mean(value, axis=0))
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in centroid_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2, fontsize=14)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
        if not os.path.exists('./figs/%s'%name):
            os.makedirs('./figs/%s/'%name)
        if attr == 'ALL':
            fig.savefig('./figs/%s/%s-cent-precision.pdf'%(name, name),format='pdf',bbox_inches='tight')
        else:
            fig.savefig('./figs/%s/%s-cent-precision-%s.pdf'%(name, name, attr),format='pdf',bbox_inches='tight')

