import os, sys
import numpy as np
from scipy import stats
import math

import torch as tc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from learning.util import to_device
import pickle


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



def box_plot(fdr_path, n_failed_path, method, model, eps):
    # Load the pickle file
    # with open('camera-ready/snapshots/NIPS24-nli-nq_gpt3.5-gpt3.5-GreedyGen-SGPlot-EXP-1_FDR_zu-10000_ze-3676_epS-0.25', 'rb') as file:
    # with open('camera-ready/snapshots/NIPS24-nli-nq_gpt3.5-gpt3.5-GreedyGen-SGPlot-EXP-1_FDR_zu-10000_ze-2764_epS-0.25', 'rb') as file:
    # with open('nq_alpaca7B-FDR_zu-10000_ze-4425_epS-0.25', 'rb') as file:
    # with open('nq_alpaca7B-FDR_zu-10000_ze-5900_epS-0.25', 'rb') as file:
    with open(fdr_path, 'rb') as file:
        data = pickle.load(file)
    # Replace the keys in data dictionary
    
    math_key = {
        'SGen_PL-H-Semi': r'$\mathtt{SGen}_\mathtt{PL}^\mathtt{H-Semi}$',
        'SGen_PFL-H-Semi': r'$\mathtt{SGen}_\mathtt{PFL}^\mathtt{H-Semi}$',
        'SGen_NoMS-Semi': r'$\mathtt{SGen}^\mathtt{Semi}_\mathtt{NoMS}$',
        'SGen_NoMS-Semi-Sup': r'$\mathtt{SGen}^\mathtt{Semi-Sup}_\mathtt{NoMS}$',
        'SGen_EM': r'$\mathtt{SGen}_\mathtt{EM}$',
        'SGen-Sup': r'$\mathtt{SGen}^\mathtt{Sup}$',
        'SGen-Semi': r'$\mathtt{SGen}^\mathtt{Semi}$'
    }
    # data = {
    #     key.replace('SGen_PL-H-Semi(f_M1)', r'$\mathtt{SGen}_\mathtt{PL}^\mathtt{H-Semi}$'+'(f_M1)')
    #         .replace('SGen_PFL-H-Semi(f_M1)', r'$\mathtt{SGen}_\mathtt{PFL}^\mathtt{H-Semi}$'+'(f_M1)')
    #         .replace('SGen_NoMS-Semi(f_M1)', r'$\mathtt{SGen}^\mathtt{Semi}_\mathtt{NoMS}$'+'(f_M1)')
    #         .replace('SGen_NoMS-Semi-Sup(f_M1)', r'$\mathtt{SGen}^\mathtt{Semi-Sup}_\mathtt{NoMS}$'+'(f_M1)')
    #         .replace('SGen_EM(f_M1)', r'$\mathtt{SGen}_\mathtt{EM}$'+'(f_M1)')
    #         .replace('SGen-Sup(f_M1)', r'$\mathtt{SGen}^\mathtt{Sup}$'+'(f_M1)'): value 
    #     for key, value in data.items()
    # }
    # data = {
    #     key.replace('SGen_PL-H-Semi(f_M2)', r'$\mathtt{SGen}_\mathtt{PL}^\mathtt{H-Semi}$'+'(f_M2)')
    #         .replace('SGen_PFL-H-Semi(f_M2)', r'$\mathtt{SGen}_\mathtt{PFL}^\mathtt{H-Semi}$'+'(f_M2)')
    #         .replace('SGen_NoMS-Semi(f_M2)', r'$\mathtt{SGen}^\mathtt{Semi}_\mathtt{NoMS}$'+'(f_M2)')
    #         .replace('SGen_NoMS-Semi-Sup(f_M2)', r'$\mathtt{SGen}^\mathtt{Semi-Sup}_\mathtt{NoMS}$'+'(f_M2)')
    #         .replace('SGen_EM(f_M2)', r'$\mathtt{SGen}_\mathtt{EM}$'+'(f_M2)')
    #         .replace('SGen-Sup(f_M2)', r'$\mathtt{SGen}^\mathtt{Sup}$'+'(f_M2)'): value 
    #     for key, value in data.items()
    # }
    # data = {key.replace('SGen-Semi', r'$\mathtt{SGen}^\mathtt{Semi}$'): value for key, value in data.items()}

    print(data.keys())
    failed = {}
    with open(n_failed_path, 'rb') as file:
        data_failed = pickle.load(file)
    for key in data.keys():
        print(key, sum(data_failed[key]))
        if sum(data_failed[key]) >= 80:
            failed[key] = True
        else:
            failed[key] = False


    fontsize = 15
    # Filter the keys to separate (f_M1) and (f_M2)
    keys_M1 = [key for key in data.keys() if '(f_M1)' in key]
    keys_M2 = [key for key in data.keys() if '(f_M2)' in key]
    keys_M1.append('SGen-Semi')
    keys_M2.append('SGen-Semi')
    # keys_M2.append('CSGen-MS')

    # Filter the keys to separate SG-EM, SG-EL, CSGen-Sup and others for (f_M1)
    keys_SL_M1 = [key for key in keys_M1 if 'SGen-Sup' in key or 'SGen_NoMS-Semi-Sup' in key]
    keys_SSL_M1 = [key for key in keys_M1 if key not in keys_SL_M1]

    # Filter the keys to separate SG-EM, SG-EL, CSGen-Sup and others for (f_M2)
    keys_SL_M2 = [key for key in keys_M2 if 'SGen-Sup' in key or 'SGen_NoMS-Semi-Sup' in key]
    keys_SSL_M2 = [key for key in keys_M2 if key not in keys_SL_M2]

    # Extract the values for (f_M1) and (f_M2)
    # print(keys_M1)
    # print(data.keys)
    values_M1 = [data[key] for key in keys_M1]
    values_M2 = [data[key] for key in keys_M2]
    failed_values_M1 = [failed[key] for key in keys_M1]
    failed_values_M2 = [failed[key] for key in keys_M2]

    # Calculate the mean and standard deviation for (f_M1) and (f_M2)
    means_M1 = [np.mean(val) for val in values_M1]
    stds_M1 = [np.std(val) for val in values_M1]
    means_M2 = [np.mean(val) for val in values_M2]
    stds_M2 = [np.std(val) for val in values_M2]

    keys_M1 = [key[:-6] if '(f_M1)' in key else key for key in keys_M1]
    keys_M2 = [key[:-6] if '(f_M2)' in key else key for key in keys_M2]
    keys_SL_M1 = [key[:-6] if '(f_M1)' in key else key for key in keys_SL_M1]
    keys_SSL_M1 = [key[:-6] if '(f_M1)' in key else key for key in keys_SSL_M1]
    keys_SL_M2 = [key[:-6] if '(f_M2)' in key else key for key in keys_SL_M2]
    keys_SSL_M2 = [key[:-6] if '(f_M2)' in key else key for key in keys_SSL_M2]

    whis = (0.02*100, (1-0.02)*100)

    if method == 'SSL':
        # Create the box plot for (f_M1) - SSL
        plt.figure()
        box = plt.boxplot([values_M1[keys_M1.index(key)] for key in keys_SSL_M1], labels=[math_key[key] for key in keys_SSL_M1], whis=whis)
        for i, key in enumerate(keys_SSL_M1):
            box_color = 'g' if failed_values_M1[keys_M1.index(key)] else 'r'
            plt.setp(box['boxes'][i], color=box_color)
            plt.setp(box['medians'][i], color=box_color)
        plt.xlabel('Method')
        plt.ylabel('FDR-E')
        # plt.title('Box Plot for (f_M1) - SSL')
        # plt.ylim(0, 0.4)  # Set the y-axis limit to show values up to 0.4
        plt.axhline(y=eps, xmin=-0.5, xmax=len(keys_SSL_M1) - 0.5, color='k', linestyle='dashed', label='$\epsilon_S = %.2f$' % eps, zorder=4)  # Add a horizontal dashed line at y=0.25
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=fontsize, ncol=3)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        # plt.ylim(0, 0.4)  # Set the y-axis limit to show values up to 0.75
        plt.tight_layout()
        plt.savefig(f'snapshots/box_plot/{model}_f_M1_SSL.png')
        plt.savefig(f'snapshots/box_plot/{model}_f_M1_SSL.pdf')

        # Create the box plot for (f_M2) - SSL
        plt.figure()
        box = plt.boxplot([values_M2[keys_M2.index(key)] for key in keys_SSL_M2], labels=[math_key[key] for key in keys_SSL_M2], whis=whis)
        for i, key in enumerate(keys_SSL_M2):
            box_color = 'g' if failed_values_M2[keys_M2.index(key)] else 'r'
            plt.setp(box['boxes'][i], color=box_color)
            plt.setp(box['medians'][i], color=box_color)
        plt.xlabel('Method')
        plt.ylabel('FDR-E')
        # plt.title('Box Plot for (f_M2) - SSL')
        plt.axhline(y=eps, xmin=-0.5, xmax=len(keys_SSL_M2) - 0.5, color='k', linestyle='dashed', label='$\epsilon_S = %.2f$' % eps, zorder=4)  # Add a horizontal dashed line at y=0.25
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=fontsize, ncol=3)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        # plt.ylim(0, 0.4)  # Set the y-axis limit to show values up to 0.75
        plt.tight_layout()
        plt.savefig(f'snapshots/box_plot/{model}_f_M2_SSL.png')
        plt.savefig(f'snapshots/box_plot/{model}_f_M2_SSL.pdf')



    if method == 'SL':
        # Create the box plot for (f_M1) - SL
        plt.figure()
        box = plt.boxplot([values_M1[keys_M1.index(key)] for key in keys_SL_M1], labels=[math_key[key] for key in keys_SL_M1], whis=whis)
        for i, key in enumerate(keys_SL_M1):
            box_color = 'g' if failed_values_M1[keys_M1.index(key)] else 'r'
            plt.setp(box['boxes'][i], color=box_color)
            plt.setp(box['medians'][i], color=box_color)
        plt.xlabel('Method')
        plt.ylabel('FDR-E')
        # plt.title('Box Plot for (f_M1) - SL')
        plt.axhline(y=eps, xmin=-0.5, xmax=len(keys_SL_M1) - 0.5, color='k', linestyle='dashed', label='$\epsilon_S = %.2f$' % eps, zorder=4)  # Add a horizontal dashed line at y=0.25
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=fontsize, ncol=3)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        # plt.ylim(0, 0.4)  # Set the y-axis limit to show values up to 0.75
        plt.tight_layout()
        plt.savefig(f'snapshots/box_plot/{model}_f_M1_SL.png')
        plt.savefig(f'snapshots/box_plot/{model}_f_M1_SL.pdf')

        # Create the box plot for (f_M2) - SL
        plt.figure()
        box = plt.boxplot([values_M2[keys_M2.index(key)] for key in keys_SL_M2], labels=[math_key[key] for key in keys_SL_M2], whis=whis)
        for i, key in enumerate(keys_SL_M2):
            box_color = 'g' if failed_values_M2[keys_M2.index(key)] else 'r'
            plt.setp(box['boxes'][i], color=box_color)
            plt.setp(box['medians'][i], color=box_color)
        plt.xlabel('Method')
        plt.ylabel('FDR-E')
        # plt.title('Box Plot for (f_M2) - SL')
        plt.axhline(y=eps, xmin=-0.5, xmax=len(keys_SL_M2) - 0.5, color='k', linestyle='dashed', label='$\epsilon_S = %.2f$' % eps, zorder=4)  # Add a horizontal dashed line at y=0.25
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=fontsize, ncol=3)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        # plt.ylim(0, 0.4)  # Set the y-axis limit to show values up to 0.75
        plt.tight_layout()
        plt.savefig(f'snapshots/box_plot/{model}_f_M2_SL.png')
        plt.savefig(f'snapshots/box_plot/{model}_f_M2_SL.pdf')



def quan_plot(output_fdrs, output_effs, faileds, ablations, eps):
    mdls = ['alpaca7B', 'gpt3.5']
    
    categories = {
        'gpt3.5': [
            [r'$\mathtt{SGen}^\mathtt{Semi}_\mathtt{NoMS}$' + f' {i // 1000}k' for i in ablations['gpt3.5'][0]],
            [r'$\mathtt{SGen}^\mathtt{Semi}$' + f' {i // 1000}k' for i in ablations['gpt3.5'][0]]
        ],
        'alpaca7B': [
            [r'$\mathtt{SGen}^\mathtt{Semi}_\mathtt{NoMS}$' + f' {i // 1000}k' for i in ablations['alpaca7B'][0]],
            [r'$\mathtt{SGen}^\mathtt{Semi}$' + f' {i // 1000}k' for i in ablations['alpaca7B'][0]]
        ]
    }

    plt.figure(1)
    plt.clf()
    for mdl in mdls:
        for idx in range(2):
            fdrs = [pickle.load(open(output_fdrs[mdl][i], 'rb')) for i in range(len(output_fdrs[mdl]))]
            effs = [pickle.load(open(output_effs[mdl][i], 'rb')) for i in range(len(output_effs[mdl]))]

            fdr_mean = [np.mean(fdrs[i]['SGen_NoMS-Semi(f_M2)']) for i in range(len(fdrs))] if idx == 0 else [np.mean(fdrs[i]['SGen-Semi']) for i in range(len(fdrs))]
            eff_mean = [np.mean(effs[i]['SGen_NoMS-Semi(f_M2)']) for i in range(len(effs))] if idx == 0 else [np.mean(effs[i]['SGen-Semi']) for i in range(len(effs))]
            fdr_std = [np.std(fdrs[i]['SGen_NoMS-Semi(f_M2)']) for i in range(len(fdrs))] if idx == 0 else [np.std(fdrs[i]['SGen-Semi']) for i in range(len(fdrs))]
            eff_std = [np.std(effs[i]['SGen_NoMS-Semi(f_M2)']) for i in range(len(effs))] if idx == 0 else [np.std(effs[i]['SGen-Semi']) for i in range(len(effs))]

            value1 = fdr_mean
            value2 = eff_mean
            name = 'noms' if idx == 0 else 'ms'
            category = categories[mdl][idx]

            os.makedirs(f'snapshots/quan_plot', exist_ok=True)
            with PdfPages(f'snapshots/quan_plot/{mdl}_{name}_{mdl}' + '.pdf') as pdf:
                fontsize = 20

                x = np.arange(len(category))
                width = 0.4 

                fig, ax1 = plt.subplots(figsize=(10, 6))

                # fdr-axis
                bars1 = ax1.bar(x - width / 2, value1, width, label='FDR-E', color='#77dd77', zorder=3, capsize=5)
                ax1.errorbar(x - width / 2, value1, yerr=fdr_std, fmt='none', ecolor='black', elinewidth=1, capsize=5, capthick=1, zorder=4)
                ax1.set_xlabel('Methods', fontsize=fontsize)
                ax1.set_ylabel('FDR-E', fontsize=fontsize)
                ax1.tick_params(axis='y', labelsize=fontsize * 0.75)
                ax1.set_ylim([None, 0.3])

                ax1.set_xticks(x)
                ax1.set_xticklabels(category, fontsize=fontsize * 0.7)

                # eff-axis
                ax2 = ax1.twinx()
                bars2 = ax2.bar(x + width / 2, value2, width, label='efficiency', color='#FF6F61', zorder=3)
                ax2.errorbar(x + width / 2, value2, yerr=eff_std, fmt='none', ecolor='black', elinewidth=1, capsize=5, capthick=1, zorder=4)
                ax2.set_ylabel('efficiency', fontsize=fontsize)
                ax2.tick_params(axis='y', labelsize=fontsize * 0.75)
                ax2.set_ylim([0.58, 0.78]) if mdl == 'gpt3.5' else ax2.set_ylim([0.22, 0.36])

                ax1.hlines(eps, -0.5, len(category) - 0.5, colors='k', linestyles='dashed', label=r'$\epsilon_S = %.2f$' % eps, zorder=4)

                fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=fontsize, ncol=3)

                ax1.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
                ax2.grid(False)

                fig.tight_layout()

                plt.show()

                plt.savefig(f'snapshots/quan_plot/{mdl}_{name}_{mdl}.png', bbox_inches='tight')
                pdf.savefig(bbox_inches='tight')

