import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from .draw_utils import COLOR, LINE_STYLE
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
                           norm_precision_ret=None, centroid_precision_ret=None, bold_name=None, axis=[0, 1]):
    # success plot
    fig, ax = plt.subplots(figsize=(13,8),dpi=300)
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if attr == 'ALL':
        plt.title('Success plots of OPE on %s' % (name))
    else:
        plt.title('Success plots of OPE - %s' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        label = "[%.3f] " % (auc) + tracker_name

        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                 color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    plt.show()

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots(figsize=(13,8),dpi=300)
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title('Corners Precision plots of OPE on %s' % (name))
        else:
            plt.title('Corners Precision plots of OPE - %s' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(np.mean(value, axis=0))
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = "[%.3f] %s" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()

    if centroid_precision_ret:
        # norm precision plot
        fig, ax = plt.subplots(figsize=(13,8),dpi=300)
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold')
        plt.ylabel('centroid Precision')
        if attr == 'ALL':
            plt.title('Precision plots of OPE on %s' % (name))
        else:
            plt.title('Precision plots of OPE - %s' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in centroid_precision_ret.keys():
            value = [v for k, v in centroid_precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(np.mean(value, axis=0))
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = "[%.3f] %s" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in centroid_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()


    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots(figsize=(13,8),dpi=300)
        ax.grid(b=True)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title('Normalized Precision plots of OPE on %s' % (name))
        else:
            plt.title('Normalized Precision plots of OPE - %s' % (attr))
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value)
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = "[%.3f] %s" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx%10], linestyle=LINE_STYLE[idx%10],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
