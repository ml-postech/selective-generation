import os, sys
import numpy as np
from scipy import stats
import math

import torch as tc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def logfactorial_rec(n, mem={}):
    mem[1] = np.log(1)
    if n not in mem:
        mem[n] = np.log(n) + logfactorial(n-1, mem=mem)
    return mem[n]


def logfactorial(n, mem={}):
    if n == 0:
        return np.log(1)
    trace = []
    mem[1] = np.log(1)
    i = n
    while i not in mem:
        trace.append(i)
        i -= 1
    while trace:
        mem[i+1] = np.log(trace.pop()) + mem[i]
        i += 1        
    return mem[n]


def logcomb(n, k, mem={}):
    assert(n >= k)
    assert(k >= 0)
    
    if (n, k) not in mem:
        lognCk = logfactorial(n, mem=mem) - logfactorial(k, mem=mem) - logfactorial(n-k, mem=mem)
        mem[(n, k)] = lognCk
    return mem[(n, k)]

    # if (n, k) not in mem:
    #     nCk = np.sum([np.log(n - i) for i in range(k)]) - np.sum([np.log(i) for i in range(1, k+1)])
    #     mem[(n, k)] = nCk
    # return mem[(n, k)]


def clopper_pearson(k, n, alpha, two_side=False, use_R=False, lower_bound=False):
    if two_side:
        if use_R:
            lo = stats.qbeta(alpha/2, int(k), int(n-k+1))[0]
            hi = stats.qbeta(1 - alpha/2, int(k+1), int(n-k))[0]
        else:
            lo = stats.beta.ppf(alpha/2, k, n-k+1)
            hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
        
            lo = 0.0 if math.isnan(lo) else lo
            hi = 1.0 if math.isnan(hi) else hi
    
        return lo, hi
    else:
        if use_R:
            hi = stats.qbeta(1 - alpha, int(k+1), int(n-k))[0]
        else:
            if lower_bound:
                lo = stats.beta.ppf(alpha, k, n-k+1)
                lo = 0.0 if math.isnan(lo) else lo

                return lo
            else:
                hi = stats.beta.ppf(1 - alpha, k+1, n-k)
                hi = 1.0 if math.isnan(hi) else hi
    
                return hi
        return hi

def clopper_pearson_worst(k, n, alpha, lower_bound=False):
    return clopper_pearson(k, n, alpha, lower_bound=lower_bound)



def plot_rel_diag(n_bins, conf_t, conf_e, n_cnt, ece, fn, fontsize=15):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    bins = np.linspace(0.0, 1.0, n_bins)
    bin_center = (bins[:-1] + bins[1:])/2.0
    conf_e, conf_t = conf_e[n_cnt>0], conf_t[n_cnt>0] 
    plt.figure(1)
    plt.clf()
    fig, ax1 = plt.subplots()
    ## acc-conf plot
    h1 = ax1.plot(conf_e, conf_t, 'ro--', label='estimated')
    h2 = ax1.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k-', label='ideal')
    ## example rate
    ax2 = ax1.twinx()
    h3 = ax2.bar(bin_center, n_cnt/np.sum(n_cnt), width=(bin_center[1]-bin_center[0])*0.75, color='b', edgecolor='k', alpha=0.5, label='ratio')
    ## beautify
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))
    ax1.grid('on')
    ax1.set_xlabel('confidence', fontsize=fontsize)
    ax1.set_ylabel('accuracy', fontsize=fontsize)
    ax2.set_ylabel('example ratio', fontsize=fontsize)
    plt.title('ECE = %.2f%%'%(ece*100.0), fontsize=fontsize)
    plt.legend(handles=[h1[0], h2[0], h3], loc='upper left', fontsize=fontsize)
    fig.tight_layout()
    ## save
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()


def plot_acc_rank(corr, log_conf, fn, fontsize=15, ratio=0.01):

    ## sort
    corr = corr[np.argsort(log_conf, kind='stable')][::-1]  # conduct a stable sorting to properly handle tie
    
    n = len(corr)

    ranking = [float(i) for i in range(1, n+1)]
    corr_mean = [corr[:i].mean() for i in range(1, n+1)]

    n_trim = round(n*ratio)
    ranking = ranking[:n_trim]
    corr_mean = corr_mean[:n_trim]

    ## plot
    plt.figure(1)
    plt.clf()
    plt.plot(ranking, corr_mean, 'r--')

    # beautify
    plt.grid('on')
    plt.ylim((0.0, 1.0))
    plt.xlabel('ranking', fontsize=fontsize)
    plt.ylabel('average accuracy', fontsize=fontsize)
    
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()

    
def plot_acc_conf(corr, conf, fn, fontsize=15):

    conf_rng = np.arange(0.0, 1.0, 0.01)
    corr_mean = np.array([corr[conf>=c].mean() for c in conf_rng])
    n_cnt = np.array([np.sum(conf>=c) for c in conf_rng])

    ## plot
    plt.figure(1)
    plt.clf()
    fig, ax1 = plt.subplots()

    ## #example 
    ax2 = ax1.twinx()
    bin_center = conf_rng
    h2 = ax2.bar(bin_center, n_cnt, width=(bin_center[1]-bin_center[0]), color='b', edgecolor=None, alpha=0.3, label='#examples')

    ## curve
    h1 = ax1.plot(conf_rng, corr_mean, 'r--', label='conditional accuracy')

    # beautify
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax2.set_xlim((0, 1))

    ax1.grid('on')
    ax1.set_xlabel('confidence threshold', fontsize=fontsize)
    ax1.set_ylabel('conditional accuracy', fontsize=fontsize)
    ax2.set_ylabel('#examples', fontsize=fontsize)
    plt.legend(handles=[h2, h1[0]], fontsize=fontsize, loc='lower left')
    
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()
    
    
def ECE(ph, yh, y, n_bins=15, overconf=False, is_correct=None, rel_diag_fn=None):
    assert(len(ph) == len(y))
    n = len(y)
    bins = np.linspace(0.0, 1.0, n_bins)
    conf_e = np.zeros(len(bins)-1)
    conf_t = np.zeros(len(bins)-1)
    n_cnt = np.zeros(len(bins)-1)

    if is_correct is None:
        is_correct = lambda a, b: a == b
    
    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        idx = (ph>=l)&(ph<=u) if i==(n_bins-2) else (ph>=l)&(ph<u)
        if np.sum(idx) == 0:
            continue
        ph_i, yh_i, y_i = ph[idx], yh[idx], y[idx]
        ## compute (estimated) true confidence
        conf_t[i] = np.mean(np.array([is_correct(y_ii, yh_ii) for y_ii, yh_ii in zip(y_i, yh_i)]).astype(np.float32))
        ## compute estimated confidence
        conf_e[i] = np.mean(ph_i)
        ## count the examples in the bin
        n_cnt[i] = np.sum(idx).astype(np.float32)
        
    ## expected calibration error
    ece = np.sum(np.abs(conf_e - conf_t)*n_cnt/n)
    if overconf:
        ece_oc = np.sum(np.maximum(0.0, conf_e - conf_t)*n_cnt/n)
        
    ## plot a reliability diagram
    if rel_diag_fn is not None:
        plot_rel_diag(n_bins, conf_t, conf_e, n_cnt, ece, rel_diag_fn)

    if overconf:
        return ece, ece_oc
    else:
        return ece
    

def compute_ece(mdl, ld, device):
    ## compute calibration error
    y_list, yh_list, ph_list = [], [], []
    for x, y in ld:
        x, y = to_device(x, device), to_device(y, device)
        with tc.no_grad():
            out = mdl(x)
        ph = out['ph_cal'] if 'ph_cal' in out else out['ph_top']
        yh = out['yh_cal'] if 'yh_cal' in out else out['yh_top']
        y_list.append(y.cpu())
        yh_list.append(yh.cpu())
        ph_list.append(ph.cpu())
    y_list, yh_list, ph_list = tc.cat(y_list), tc.cat(yh_list), tc.cat(ph_list)
    ece = ECE(ph_list.numpy(), yh_list.numpy(), y_list.numpy())
    return ece
