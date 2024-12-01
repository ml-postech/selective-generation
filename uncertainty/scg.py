import os, sys
import numpy as np
import pickle
import types
import warnings
from tqdm import tqdm
import math
import copy

import torch as tc

from .base import *
from .util import *

from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer

from models.util import compute_inclusion, compute_EM

import time

from datasets import concatenate_datasets

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt

# Ours
class SCGBaseLearner(SGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, params=params, name_postfix=name_postfix)
        self.entail_model = entail_model
        self.device = self.params.device


    # LearnEntSet
    # using upper bound
    @classmethod
    def learn_entailment_set(cls, z_, eps, delta_e, verbose):

        # binary search
        assert z_[0].shape[0] == z_[1].shape[0]

        probs = z_[0]#[(z_[1] != 0).bool()]
        labels = z_[1]#[(z_[1] != 0).bool()]

        probs, i_sorted = probs.sort(descending=False)
        labels = labels[i_sorted]


        n = probs.shape[0]
        i_min = 0
        i_max = n - 1
        n_iters = int(np.ceil(np.log2(n))) + 1
        for i in range(n_iters):
            i_cur = int(np.ceil((i_min + i_max) / 2))

            k_j = ((labels == 0) * (probs >= probs[i_cur])).sum()
            TE = ((labels == 1) * (probs >= probs[i_cur])).sum()
            U = clopper_pearson_worst(k_j.item(), n, delta_e)

            if U <= eps:
                i_max = i_cur
            else:
                i_min = i_cur

            if verbose:
                print(f'[binary serach for tau_e] '
                    f'tau = {probs[i_cur]:.4e}, '
                    f'1 - prec = {k_j} / {n} = {k_j / n:.4e}, '
                    f'TE = {TE} / {n} = {TE / n:.4e}, '
                    f'U = {U:.4e}, 1-eps = {1-eps}, '
                    f'EntSet(i_min, i_max) = ({i_min}, {i_max})'
                    #f'n_contradict = {temp}')
                )

        tau_e_opt = probs[i_cur]
        U_end = U

        return k_j, tau_e_opt
    
    @classmethod
    def compute_U_SSL(cls, z_e, z_u, eps_e, delta_s, delta_e):
        k_i, tau_e_opt = cls.learn_entailment_set(z_e, eps_e, delta_e/2, False)

        z_e_shape = z_e[0].shape[0]
        z_u_shape = z_u.shape[0] if z_u is not None else 0
        
        l = ((z_e[0] < tau_e_opt) * (z_e[1] == 1)).sum()
        k = (z_u < tau_e_opt).sum() if z_u is not None else tc.tensor([0.], device=l.device)
        L_ = clopper_pearson_worst(l.item(), z_e_shape, delta_e/2, lower_bound=True)
        U = clopper_pearson_worst(k.item(), z_u_shape, delta_s/2)
        U_ssl = eps_e - L_ + U

        return tau_e_opt, k_i, U_ssl


    @classmethod
    def fdr_e_upperbound(cls, z_e, z_u, eps, delta_s, eps_e, delta_e, delta_p, fer, K):
        
        z_e_shape = z_e[0].shape[0]
        z_u_shape = z_u.shape[0] if z_u is not None else 0

        k_sl = (z_e[1] == 0).sum()
        U_sl = clopper_pearson_worst(k_sl.item(), z_e_shape, delta_s/2)
        w_sl = clopper_pearson_worst(z_e_shape, z_e_shape + z_u_shape, delta_p/2)
        tau_e_opt = float('inf') #?
        if fer:
            U_ssl = float('inf')
            eps_max = (z_e[1] == 0).sum() / z_e_shape
            for n_k in range(1, K+1):
                tau_e_i, k_i, U_ssl_i = cls.compute_U_SSL(z_e, z_u, eps_max/n_k, delta_s/K, delta_e/K)
                if U_ssl_i < U_ssl:
                    tau_e_opt = tau_e_i
                    U_ssl = U_ssl_i
                
        else:
            raise NotImplementedError
            l = ((z_e[0] >= tau_e_opt) * (z_e[1] == 0)).sum()
            k = (z_u < tau_e_opt).sum() if z_u is not None else tc.tensor([0.], device=l.device)
            U_ = clopper_pearson_worst(l.item(), z_e_shape, delta_e/2)
            U = clopper_pearson_worst(k.item(), z_u_shape, delta_s/2)
            U_ssl = U_ - eps_e + U

        w_ssl = clopper_pearson_worst(z_u_shape, z_e_shape + z_u_shape, delta_p/2)

        U_j = w_sl*U_sl + w_ssl*U_ssl

        return U_j, tau_e_opt

    def precompute_classification(self, ld, desc, n=None):
        t_start = time.time()        
        output_list = []
        if n:
            total = math.ceil(n/ld.batch_size)
        else:
            total = None
            
        for x, labels in tqdm(ld, total=total, desc=desc):
            x = to_device(x, self.params.device)
            
            with tc.no_grad():
                output = self.entail_model.classify(x, labels)
            output_list.append(to_device(output, 'cpu'))
        print(f'[precompute generation] duration = {time.time() - t_start:.2f} sec.')
  
        return output_list
    
        
    def precompute_entail_scores(self, ld, n=None, cache_fn=None):

        # pre-classification
        if cache_fn and os.path.exists(cache_fn):
            print(f'[pre-classification] loading precomputed classification results from {cache_fn}')
            output_list = pickle.load(open(cache_fn, 'rb'))
        else:
            output_list = self.precompute_classification(ld, desc="Precompute classification results")
            if cache_fn:
                pickle.dump(output_list, open(cache_fn, 'wb'))
                print(f'[pre-classification] saving precomputed classification results to {cache_fn}')
        # post-process
        ds = None
        probs = []
        labels = []
        for output in output_list:
            # un-batching
            probs.append(output.pop('probs'))
            labels.append(output.pop('labels'))
            output = {k: [v_i.cpu() for v_i in v] if tc.is_tensor(v) else v for k, v in output.items()}
            if ds is None:
                ds = output
            else:
                ds = {k: ds[k] + output[k] for k in ds.keys()}
        ds['probs'] = tc.vstack(probs)
        ds['labels'] = tc.hstack(labels)

        # truncate scores
        if n:
            ds = {k: v[:n] for k, v in ds.items()}
        n_read = len(ds[list(ds.keys())[0]])
            
        #print(f'[duration = {time.time() - t_start:.2f} sec.] n = {n_read}')
            
        if n:
            assert n_read == n, f'actual data is less than n ({n_read} != {n})'
        
        return ds
    


    ########
    # Test #
    ########
    def eval_generation(self, kwargs, kwargs_ent, tau_s, feat_idx):

        # if 'logprobs'
        answer = kwargs['answer'] if 'answer' in kwargs else self.G.base_model.tokenizer.batch_decode(kwargs['answer_ids'], skip_special_tokens=True)
        answer_pred = kwargs['answer_pred'] if 'answer_pred' in kwargs else self.G.base_model.tokenizer.batch_decode(kwargs['answer_ids_pred'], skip_special_tokens=True)

        # EM
        em = [compute_exact(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

        # F1
        f1 = [compute_f1(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

        # inclusion
        inc = [compute_inclusion(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

        # selection
        # save scores before selection...
        scores_m1 = kwargs['scores_m1']
        scores_m2 = kwargs['scores_m2']
        probs = kwargs_ent['probs']
        
        
        if feat_idx == 1:
            sel = scores_m2 >= tau_s.to(scores_m1.device)
        elif feat_idx == 2:
            tau_s_i, tau_s_j = tau_s
            sel = (scores_m1 >= tau_s_i.to(scores_m1.device)) * (scores_m2 >= tau_s_j.to(scores_m1.device))
        else:
            sel = scores_m1 >= tau_s.to(scores_m1.device)
        ent_sel = probs >= self.entail_model.logtau_model.to(probs.device)()['logits'].exp()

        # e==0 proportion (label={entail:0 , ...})
        e_0 = sel * (kwargs_ent['labels'] == 0)
        e_0_full = kwargs_ent['labels'] == 0

        ph = scores_m1

        
        yh = [normalize_answer(a) for a in answer_pred]
        y = [normalize_answer(a) for a in answer]

        return {'EM': em, 'F1': f1, 'Inc': inc, 'ph': ph, 'y': y, 'yh': yh, 'sel': sel, 'e_0':e_0, 'e_0_full':e_0_full}


    
    def compute_metrics(self, tau_s, feat_idx=None, ld=None):
        em, f1, inc = [], [], []
        yh_ece, y_ece, ph_ece = [], [], []
        sel, ent_sel = [], []
        e_0, e_0_full = [], []
        # question/answer/answer_pred
        q, a, a_pred = [], [], []

        if self.params.cache_eval_fn:
            cache_fn = os.path.join(
                self.params.cache_root,
                self.params.cache_eval_fn,
            )
            os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        else:
            cache_fn = None

        # pre-generation
        output_list = []
        # using 'logprobs'
        rd = self.entail_model.rd['test']
        if self.mdl.G.base_model == None:
            step = self.params.per_device_eval_batch_size
            for i in range(0, len(rd), step):
                # scores = tc.hstack([tc.tensor(lp).mean().exp() for lp in rd[i:i+step]['logprobs']])
                question = [d for d in rd[i:i+step]['question']]
                decoded_answer = [d for d in rd[i:i+step]['answer']]
                decoded_answer_pred = [d for d in rd[i:i+step]['generated_answer']]

                answer = [normalize_answer(a) for a in decoded_answer]
                answer_pred = [normalize_answer(a) for a in decoded_answer_pred]

                # scores_m2 = tc.hstack([(1 - tc.tensor(ss)[..., 0]).mean() for ss in rd[i:i+step]['samples_scores']])
                scores_m1 = tc.hstack([tc.tensor(lp).mean().exp() for lp in rd[i:i+step]['logprobs']])
                # CAUTION, not ids. 
                output_list.append(
                    {
                        'question':question,
                        'answer':decoded_answer,
                        'answer_pred':decoded_answer_pred,
                        'scores_m1': scores_m1,
                        # 'scores_m2': scores_m2,
                    }
                )
            decoded_answer_pred = [v_i for v in output_list for v_i in v['answer_pred']]
        else:
            if cache_fn and os.path.exists(cache_fn):
                print(f'[pre-generation] loading precomputed generation results from {cache_fn}')
                output_list = pickle.load(open(cache_fn, 'rb'))
            else:
                for x, y in tqdm(ld, desc="Precompute generation results"):
                    x = to_device(x, self.params.device)
                    
                    with tc.no_grad():
                        output = self.mdl.generate(x)
                    output['scores_m1'] = tc.hstack([lp.mean().exp() for lp in output['logprobs_answer_pred']])
                    output_list.append(to_device(output, 'cpu'))
                if cache_fn:
                    pickle.dump(output_list, open(cache_fn, 'wb'))
                    print(f'[pre-generation] saving precomputed generation results to {cache_fn}')

            decoded_answer_pred = self.mdl.G.base_model.tokenizer.batch_decode(
                [v_i for v in output_list for v_i in v['answer_ids_pred']],
                skip_special_tokens=True
            )

        if self.params.cache_ent_eval_fn:
            cache_ent_fn = os.path.join(
                self.params.cache_root,
                self.params.cache_ent_eval_fn,
                # self.params.cache_ent_eval_fn + f'-{self.n_e}',
            )
            os.makedirs(os.path.dirname(cache_ent_fn), exist_ok=True)

        # For SelfCheckGPT score
        if self.params.entail_model in ["deberta-v2-xxlarge-mnli", None]:
            scores_m2 = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd])
        else:
            entail_ld_sam = self.entail_model.init_sample_dataset_nli(
                decoded_answer_pred,
                'test'
            )
            cache_ent_fn_sam = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_ent_fn}-CAL2-SAM',
            ) if self.params.cache_ent_fn else None
            
            ent_scores_dict_sam = self.precompute_entail_scores(entail_ld_sam, cache_fn=cache_ent_fn_sam)
            sample_size = len(rd[0]['samples'])
            ent_probs_sam = ent_scores_dict_sam['probs'].view(-1, sample_size, 3)
            scores_m2 = (1 - ent_probs_sam[..., 0]).mean(dim=1)

        # Split scores_m2 into batches and add to output_list
        batch_size = len(output_list[0]['scores_m1'])
        scores_m2_batches = [scores_m2[i:i + batch_size] for i in range(0, len(scores_m2), batch_size)]
        for output, scores_m2_batch in zip(output_list, scores_m2_batches):
            output['scores_m2'] = to_device(scores_m2_batch, 'cpu')

        entail_ld = self.entail_model.init_dataset_nli(
            decoded_answer_pred,
            'test'
        )
        # pre-classification
        ent_output_list = []
        # we can use 'entailment_scores' in the dataset.
        if self.params.entail_model in ["deberta-v2-xxlarge-mnli", None]:
            step = self.params.per_device_eval_batch_size
            for i in range(0, len(rd), step):
                # scores = tc.hstack([tc.tensor(lp).mean().exp() for lp in rd[i:i+step]['logprobs']])
                prob = [d for d in rd[i:i+step]['entail_scores']]
                label = [d for d in rd[i:i+step]['labels']]
                
                ent_output_list.append(
                    {
                        'probs':tc.tensor(prob),
                        'labels':tc.tensor(label),
                    }
                )
        elif cache_ent_fn and os.path.exists(cache_ent_fn):
            print(f'[pre-generation] loading precomputed generation results from {cache_ent_fn}')
            ent_output_list = pickle.load(open(cache_ent_fn, 'rb'))
        else:
            for x, y in tqdm(entail_ld, desc="Precompute generation results"):
                x = to_device(x, self.params.device)
                
                with tc.no_grad():
                    output = self.entail_model.classify(x, y)
                ent_output_list.append(to_device(output, 'cpu'))
            if cache_ent_fn:
                pickle.dump(ent_output_list, open(cache_ent_fn, 'wb'))
                print(f'[pre-generation] saving precomputed generation results to {cache_ent_fn}')
        

        # compute metrics
        for output, ent_output in zip(output_list, ent_output_list):

            llm_stat_i = self.eval_generation(output, ent_output, tau_s, feat_idx)

            # save
            em.append(tc.tensor(llm_stat_i['EM']))
            f1.append(tc.tensor(llm_stat_i['F1']))
            inc.append(tc.tensor(llm_stat_i['Inc']))
            sel.append(llm_stat_i['sel'])
            e_0.append(llm_stat_i['e_0'])
            e_0_full.append(llm_stat_i['e_0_full'])
            
            y_ece += llm_stat_i['y']
            yh_ece += llm_stat_i['yh']
            ph_ece += llm_stat_i['ph']

            # q/a/a_pred
            # if 'logprobs'
            question = output['question'] if 'question' in output else self.G.base_model.tokenizer.batch_decode(output['question_ids'], skip_special_tokens=True)
            answer = output['answer'] if 'answer' in output else self.G.base_model.tokenizer.batch_decode(output['answer_ids'], skip_special_tokens=True)
            answer_pred = output['answer_pred'] if 'answer_pred' in output else self.G.base_model.tokenizer.batch_decode(output['answer_ids_pred'], skip_special_tokens=True)
            q.extend(question)
            a.extend(answer)
            a_pred.extend(answer_pred)       


        # concat
        em, f1, inc, sel, e_0, e_0_full = tc.cat(em).cpu(), tc.cat(f1).cpu(), tc.cat(inc).cpu(), tc.cat(sel).cpu(), tc.cat(e_0).cpu(), tc.cat(e_0_full).cpu()

        
        # compute ECE
        # rel_diag_fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', 'rel_diag')
        ece = ECE(
            np.array(ph_ece),
            np.array(yh_ece),
            np.array(y_ece),
            is_correct=self.G.is_correct,
            # rel_diag_fn=rel_diag_fn
        )
        
        ece_sel = ECE(
            np.array(ph_ece)[sel],
            np.array(yh_ece)[sel],
            np.array(y_ece)[sel],
            is_correct=self.G.is_correct,
            # rel_diag_fn=os.path.join(self.params.snapshot_root,
            #                          self.params.exp_name,
            #                          'figs',
            #                          'rel_diag_selected')
        )
        ret = {
            'EM': em, 'F1': f1, 'Inc': inc, 'ECE': ece, 'ECE_sel': ece_sel, 'sel': sel, 'ent_sel': ent_sel, 'e_0':e_0, 'e_0_full':e_0_full,
        }
        ret_json = {
            'question': q,
            'answer':a,
            'generated_answer': a_pred,
            'EM_test': em.tolist(),
            'F1_test': f1.tolist(),
            'Inc_test': inc.tolist(),
            'sel_test': sel.tolist(),
            'e_0_test': e_0.tolist(),
            'e_0_full_test': e_0_full.tolist(),
        }
        
        return ret, ret_json

    
    def test(self, tau_s, name, ld=None, feat_idx=None, save=False, verbose=True):
        # compute basic metrics for language generators
        fn = os.path.join(self.params.snapshot_root,
                          self.params.exp_name,
                          f'stats_pred_set_{self.name_postfix}.pk'
                          if self.name_postfix else 'stats_pred_set.pk')
        if os.path.exists(fn) and not self.params.rerun:
            print(f'load precomputed results at {fn}')
            res = pickle.load(open(fn, 'rb'))
        else:
            metrics, ref = self.compute_metrics(tau_s, feat_idx, ld)
            for k in [k for k in metrics.keys()]:
                metrics[k + '_test'] = metrics.pop(k)
            res = metrics
            if save:
                # save json
                fp = os.path.join(self.params.snapshot_root,
                          self.params.exp_name,
                          'quantitative_results.json')                
                jres = [dict(zip(ref.keys(), values)) for values in zip(*ref.values())]

                import json
                with open(fp, 'w') as f:
                    json.dump(jres, f, indent=4)

                pickle.dump(res, open(fn, 'wb'))
                
        if verbose:
            print('==================================================')
            print(
                f'[test: {name}]\n'
                f'#test dataset: {res["sel_test"].shape[0]}\n'
                f'#selection = {res["sel_test"].long().sum()}\n'
                f'efficiency = {res["sel_test"].float().mean()*100:.4f}%\n'
                f'EM (full) = {res["EM_test"].float().mean():.4f}\n'
                f'EM (sel) = {res["EM_test"][res["sel_test"]].float().mean():.4f}\n'
                f'F1 (full) = {res["F1_test"].float().mean():.4f}\n'
                f'F1 (sel) = {res["F1_test"][res["sel_test"]].float().mean():.4f}\n'
                f'Inclusion (full) = {res["Inc_test"].float().mean():.4f}\n'
                f'Inclusion (sel) = {res["Inc_test"][res["sel_test"]].float().mean():.4f}\n'
                f'ECE (full) = {res["ECE_test"]*100:.4f}%\n'
                f'ECE (sel) = {res["ECE_sel_test"]*100:.4f}%\n'
                f'E0 (full) = {(res["e_0_full_test"].long().sum()/res["e_0_full_test"].shape[0]):.4f}\n'
                f'E0 (sel) = {(res["e_0_test"].long().sum()/res["e_0_test"].shape[0]):.4f}'
            )
            print('==================================================')
            print()

        # Save to file
        if self.params.method == 'GreedyGen-SG':
            with open(f'{self.params.output_dir}/results.txt', 'a') as f:
                f.write('==================================================\n')
                f.write(
                    f'[test: {name}]\n'
                    f'#test dataset: {res["sel_test"].shape[0]}\n'
                    f'#selection = {res["sel_test"].long().sum()}\n'
                    f'efficiency = {res["sel_test"].float().mean()*100:.4f}%\n'
                    f'EM (full) = {res["EM_test"].float().mean():.4f}\n'
                    f'EM (sel) = {res["EM_test"][res["sel_test"]].float().mean():.4f}\n'
                    f'F1 (full) = {res["F1_test"].float().mean():.4f}\n'
                    f'F1 (sel) = {res["F1_test"][res["sel_test"]].float().mean():.4f}\n'
                    f'Inclusion (full) = {res["Inc_test"].float().mean():.4f}\n'
                    f'Inclusion (sel) = {res["Inc_test"][res["sel_test"]].float().mean():.4f}\n'
                    f'ECE (full) = {res["ECE_test"]*100:.4f}%\n'
                    f'ECE (sel) = {res["ECE_sel_test"]*100:.4f}%\n'
                    f'E0 (full) = {(res["e_0_full_test"].long().sum()/res["e_0_full_test"].shape[0]):.4f}\n'
                    f'E0 (sel) = {(res["e_0_test"].long().sum()/res["e_0_test"].shape[0]):.4f}'
                )
                f.write('\n==================================================\n')
                f.write('\n')
            
        return res
    
# experiment
class SGLearner(SCGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, entail_model=entail_model, params=params, name_postfix=name_postfix)
        
    
    def train(self, ld1, ld2, ld_test, updated_params=None):
        # init params
        params = copy.deepcopy(self.params)
        if updated_params:
            for a in dir(updated_params):
                if a[:2] != '__' and a[-2:] != '__':
                    setattr(params, a, getattr(updated_params, a))
        n = params.n
        self.n_e = n_e = params.n_e
        eps = params.eps
        delta = params.delta

        eps_e = params.eps_e
        delta_p = params.delta_p
        
        delta_s = (delta-delta_p)/2
        delta_e = (delta-delta_p)/2

        fer = params.fer
        K = params.K

        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'n-{n}-eps-{eps:e}-delta-{delta:e}'
        verbose = params.verbose
        
        print(f"# learn a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        # load a pre-trained model
        # if not self.params.rerun and self._check_model(best=False, is_e=True):
        #     if self.params.load_final:
        #         self._load_model(best=False, is_e=True)
        #     else:
        #         self._load_model(best=True, is_e=True)
        #     return True

        if self.params.cache_cal_fn:
            cache_fn = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_cal_fn}-CAL2',
            )
            cache_fn_e = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_cal_fn}-CAL2_E',
            )
            os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        else:
            cache_fn = None
            cache_fn_e = None

        if os.path.exists(f'{params.output_dir}/results.txt'):
            with open(f'{self.params.output_dir}/results.txt', 'w') as f:
                f.truncate(0)


        rd1 = self.entail_model.rd['val1']
        rd2 = self.entail_model.rd['val2']
        rd1_2 = self.entail_model.rd['val1+2']
        # if 'logprobs'
        if self.mdl.G.base_model == None:

            scores_m1_u = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd1])
            scores_m1_e = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd2])
            scores_m1_ue = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd1_2])

            decoded_answer_u = [d['answer'] for d in rd1]
            decoded_answer_pred_u = [d['generated_answer'] for d in rd1]

            decoded_answer_e = [d['answer'] for d in rd2]
            decoded_answer_pred_e = [d['generated_answer'] for d in rd2]

            decoded_answer_ue = [d['answer'] for d in rd1_2]
            decoded_answer_pred_ue = [d['generated_answer'] for d in rd1_2]
        
        # Could be different from the answers in dataset if not greedy.
        else:
            scores_dict_u = self.precompute_scores(ld1, params.z_u, cache_fn=cache_fn)
            scores_dict_e = self.precompute_scores(ld2, params.z_e, cache_fn=cache_fn_e)

            scores_m1_u = tc.hstack([lp.mean().exp() for lp in scores_dict_u['logprobs_answer_pred']])
            scores_m1_e = tc.hstack([lp.mean().exp() for lp in scores_dict_e['logprobs_answer_pred']])
            scores_m1_ue = tc.cat([scores_m1_u, scores_m1_e], dim=0)

            decoded_answer_u = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_u['answer_ids'], skip_special_tokens=True)
            decoded_answer_pred_u = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_u['answer_ids_pred'], skip_special_tokens=True)
            decoded_answer_e = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_e['answer_ids'], skip_special_tokens=True)
            decoded_answer_pred_e = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_e['answer_ids_pred'], skip_special_tokens=True)

            decoded_answer_ue = decoded_answer_u + decoded_answer_e
            decoded_answer_pred_ue = decoded_answer_pred_u + decoded_answer_pred_e
         
        answer_e = [normalize_answer(a) for a in decoded_answer_e]
        answer_pred_e = [normalize_answer(a) for a in decoded_answer_pred_e]
        # errors_e = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_e, answer_pred_e)], device=scores_e.device)

        answer_u = [normalize_answer(a) for a in decoded_answer_u]
        answer_pred_u = [normalize_answer(a) for a in decoded_answer_pred_u]
        # errors_u = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_u, answer_pred_u)], device=scores_e.device)

        # For SG-EM
        answer_ue = [normalize_answer(a) for a in decoded_answer_ue]
        answer_pred_ue = [normalize_answer(a) for a in decoded_answer_pred_ue]
        errors_ue = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_ue, answer_pred_ue)], device=scores_m1_ue.device)

        # entail modl init & score computation
        if params.entail_model in ["deberta-v2-xxlarge-mnli", None]:
            ent_probs_u = tc.hstack([1 - tc.tensor(lp['entail_scores'][0]) for lp in rd1])
            ent_labels_u = tc.hstack([tc.tensor([-1]) for lp in rd1]) # all labels are -1
            ent_probs_e = tc.hstack([1 - tc.tensor(lp['entail_scores'][0]) for lp in rd2])
            ent_labels_e = tc.hstack([tc.tensor(lp['labels']) for lp in rd2])

            # mean, entail/(entail+contradict)
            scores_m2_u = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd1])
            scores_m2_e = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd2])
            scores_m2_ue = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd1_2])
        
        else:
            entail_ld_u = self.entail_model.init_dataset_nli(
                decoded_answer_pred_u,
                # val1, val1+2 or val2.
                'val1'
            )
            entail_ld_e = self.entail_model.init_dataset_nli(
                decoded_answer_pred_e,
                'val2'
            )
            entail_ld_sam_u = self.entail_model.init_sample_dataset_nli(
                decoded_answer_pred_u,
                'val1'
            )
            entail_ld_sam_e = self.entail_model.init_sample_dataset_nli(
                decoded_answer_pred_e,
                'val2'
            )

            if self.params.cache_ent_fn:
                cache_ent_fn_e = os.path.join(
                    self.params.cache_root,
                    f'{self.params.cache_ent_fn}-CAL2_ZE-{params.z_e}',
                )
                cache_ent_fn_u = os.path.join(
                    self.params.cache_root,
                    f'{self.params.cache_ent_fn}-CAL2-ZU-{params.z_u}',
                )
                cache_ent_fn_sam_e = os.path.join(
                    self.params.cache_root,
                    f'{self.params.cache_ent_fn}-CAL2-SAM-ZE-{params.z_e}',
                )
                cache_ent_fn_sam_u = os.path.join(
                    self.params.cache_root,
                    f'{self.params.cache_ent_fn}-CAL2-SAM-ZU-{params.z_u}',
                )
                os.makedirs(os.path.dirname(cache_ent_fn_u), exist_ok=True)
            else:
                cache_ent_fn_u = None
                cache_ent_fn_e = None
                cache_ent_fn_sam_u = None
                cache_ent_fn_sam_e = None


            ent_scores_dict_e = self.precompute_entail_scores(entail_ld_e, cache_fn=cache_ent_fn_e)
            ent_scores_dict_u = self.precompute_entail_scores(entail_ld_u, cache_fn=cache_ent_fn_u)
            ent_scores_dict_sam_e = self.precompute_entail_scores(entail_ld_sam_e, cache_fn=cache_ent_fn_sam_e)
            ent_scores_dict_sam_u = self.precompute_entail_scores(entail_ld_sam_u, cache_fn=cache_ent_fn_sam_u)
            
            ent_probs_e = 1 - ent_scores_dict_e['probs'][..., 0]
            ent_labels_e = ent_scores_dict_e['labels']

            ent_probs_u = 1 - ent_scores_dict_u['probs'][..., 0]
            ent_labels_u = ent_scores_dict_u['labels'] # all labels are -1

            # mean, entail/(entail+contradict)
            sample_size = len(rd2[0]['samples'])
            ent_probs_sam_e = ent_scores_dict_sam_e['probs'].view(-1, sample_size, 3)
            ent_probs_sam_u = ent_scores_dict_sam_u['probs'].view(-1, sample_size, 3)
            
            scores_m2_u = (1 - ent_probs_sam_u[..., 0]).mean(dim=1)
            scores_m2_e = (1 - ent_probs_sam_e[..., 0]).mean(dim=1)
            scores_m2_ue = tc.cat([scores_m2_u, scores_m2_e], dim=0)


        # For filtering, added this temporary lines. later will be fixed
        ent_probs_ue = tc.cat([ent_probs_u, ent_probs_e], dim=0)
        ent_labels_ue = tc.cat([ent_labels_u, ent_labels_e], dim=0)


        # Z_E,U
        scores_m1_ue, i_sorted = scores_m1_ue.sort(descending=False)
        scores_m2_ue, j_sorted = scores_m2_ue.sort(descending=False)
        errors_m1_ue, errors_m2_ue = errors_ue[i_sorted], errors_ue[j_sorted]
        ent_probs_m1_ue, ent_probs_m2_ue = ent_probs_ue[i_sorted], ent_probs_ue[j_sorted]
        ent_labels_m1_ue, ent_labels_m2_ue = ent_labels_ue[i_sorted], ent_labels_ue[j_sorted]

        # These sorts are for SGen-Sup because it does need sorted score.
        scores_m1_EL, i_sorted = scores_m1_e.sort(descending=False)
        scores_m2_EL, j_sorted = scores_m2_e.sort(descending=False)
        ent_labels_m1_EL, ent_labels_m2_EL = ent_labels_e[i_sorted], ent_labels_e[j_sorted]

        # device
        device = self.device
        scores_m1_UE, scores_m2_UE = scores_m1_ue.to(device), scores_m2_ue.to(device)
        errors_m1_UE, errors_m2_UE = errors_m1_ue.to(device), errors_m2_ue.to(device)
        ent_probs_m1_UE, ent_probs_m2_UE = ent_probs_m1_ue.to(device), ent_probs_m2_ue.to(device)
        ent_labels_m1_UE, ent_labels_m2_UE = ent_labels_m1_ue.to(device), ent_labels_m2_ue.to(device)

        scores_m1_E, scores_m2_E = scores_m1_e.to(device), scores_m2_e.to(device)
        ent_probs_E, ent_labels_E = ent_probs_e.to(device), ent_labels_e.to(device)

        scores_m1_U, scores_m2_U = scores_m1_u.to(device), scores_m2_u.to(device)
        ent_probs_U = ent_probs_u.to(device)

        scores_m1_EL, scores_m2_EL = scores_m1_EL.to(device), scores_m2_EL.to(device)
        ent_labels_m1_EL, ent_labels_m2_EL = ent_labels_m1_EL.to(device), ent_labels_m2_EL.to(device)
                                                                                              

        quali_tau_s = []
        print()
        ############################################
        #            SGen_EM(f_M1)             #
        tau_s_opt, U_min_opt, eff = SG_Baseline.train(
            scores=scores_m1_UE,
            eps=eps,
            delta=delta,
            verbose=params.verbose,
            errors=errors_m1_UE,
            fer=fer,
        )
        if U_min_opt <= eps:
            res = f'[SGen_EM(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_EM(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()

        ## test
        self.test(tau_s_opt, '[SGen_EM(f_M1)]', ld_test)
        quali_tau_s.append(tau_s_opt)
        ############################################
        print()
        ############################################
        #            SGen_EM(f_M2)             #
        tau_s_opt, U_min_opt, eff = SG_Baseline.train(
            scores=scores_m2_UE,
            eps=eps,
            delta=delta,
            verbose=params.verbose,
            errors=errors_m2_UE,
            fer=fer,
        )
        if U_min_opt <= eps:
            res = f'[SGen_EM(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_EM(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()

        ## test
        self.test(tau_s_opt, '[SGen_EM(f_M2)]', ld_test, feat_idx=1)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #            SGen_PL-H-Semi(f_M1)             #
        tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
            scores_UE=scores_m1_UE,
            scores_U=scores_m1_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m1_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta=delta,
            verbose=params.verbose,
            fer=fer,
        )

        if U_min_opt <= eps:
            res = f'[SGen_PL-H-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_PL-H-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_PL-H-Semi(f_M1)]', ld_test)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #            SGen_PL-H-Semi(f_M2)             #
        tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
            scores_UE=scores_m2_UE,
            scores_U=scores_m2_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta=delta,
            verbose=params.verbose,
            fer=fer,
        )

        if U_min_opt <= eps:
            res = f'[SGen_PL-H-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_PL-H-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_PL-H-Semi(f_M2)]', ld_test, feat_idx=1)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #       SGen_PL-H-Semi(f_M1, filtering)       #
        tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
            scores_UE=scores_m1_UE,
            scores_U=scores_m1_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m1_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta=delta,
            verbose=params.verbose,
            filtering=True,
            ent_probs_UE=ent_probs_m1_UE,
            ent_labels_UE=ent_labels_m1_UE,
            fer=fer,
        )

        if U_min_opt <= eps:
            res = f'[SGen_PFL-H-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_PFL-H-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_PFL-H-Semi(f_M1)]', ld_test)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #       SGen_PL-H-Semi(f_M2, filtering)       #
        tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
            scores_UE=scores_m2_UE,
            scores_U=scores_m2_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta=delta,
            verbose=params.verbose,
            filtering=True,
            ent_probs_UE=ent_probs_m2_UE,
            ent_labels_UE=ent_labels_m2_UE,
            fer=fer,
        )

        if U_min_opt <= eps:
            res = f'[SGen_PFL-H-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_PFL-H-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_PFL-H-Semi(f_M2)]', ld_test, feat_idx=1)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #             SGen_NoMS-Semi(f_M1)              #
        tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
            scores_UE = scores_m1_UE,
            scores_U=scores_m1_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m1_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s,
            delta_e=delta_e,
            delta_p=delta_p,
            verbose=params.verbose,
            fer=fer,
            K=K,
        )

        if U_min_opt <= eps:
            res = f'[SGen_NoMS-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_NoMS-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_NoMS-Semi(f_M1)]', ld_test)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #             SGen_NoMS-Semi(f_M2)              #
        tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
            scores_UE = scores_m2_UE,
            scores_U=scores_m2_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s,
            delta_e=delta_e,
            delta_p=delta_p,
            verbose=params.verbose,
            fer=fer,
            K=K,
        )

        if U_min_opt <= eps:
            res = f'[SGen_NoMS-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_NoMS-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_NoMS-Semi(f_M2)]', ld_test, feat_idx=1)
        quali_tau_s.append(tau_s_opt)
        ############################################


        ############################################
        #            SGen-Sup(f_M1)             #
        tau_s_opt, U_min_opt, eff = SG_Baseline.train(
            # ent_labels=ent_labels_E1, # this is implemented in errors.
            scores=scores_m1_EL,
            eps=eps,
            delta=delta,
            verbose=params.verbose,
            errors=(ent_labels_m1_EL==0),
            fer=fer,
        )

        if U_min_opt <= eps:
            res = f'[SGen-Sup(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen-Sup(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen-Sup(f_M1)]', ld_test)
        quali_tau_s.append(tau_s_opt)

        ############################################
        #            SGen-Sup(f_M2)             #
        tau_s_opt, U_min_opt, eff = SG_Baseline.train(
            # ent_labels=ent_labels_E2, # this is implemented in errors.
            scores=scores_m2_EL,
            eps=eps,
            delta=delta,
            verbose=params.verbose,
            errors=(ent_labels_m2_EL==0),
            fer=fer,
        )

        if U_min_opt <= eps:
            res = f'[SGen-Sup(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen-Sup(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen-Sup(f_M2)]', ld_test, feat_idx=1)
        quali_tau_s.append(tau_s_opt)
        ############################################


        ############################################
        #               SGen-Semi               #
        #          with new decomposition          #
        tau_s_opt, feat_idx, _ = SG_MS.train(
            scores_m1_UE=scores_m1_UE,
            scores_m2_UE=scores_m2_UE,
            scores_m1_E=scores_m1_E,
            scores_m2_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            ent_labels_E=ent_labels_E,
            scores_m1_U=scores_m1_U,
            scores_m2_U=scores_m2_U,
            ent_probs_U=ent_probs_U,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s,
            delta_e=delta_e,
            delta_p=delta_p,
            verbose=params.verbose,
            fer=fer,
            K=K,
            params=params,
        )
        print('#'*20)
        self.test(tau_s_opt, '[SGen-Semi]', ld_test, feat_idx)
        quali_tau_s.append(tau_s_opt)
        print('#'*20)
        print()
        ############################################

        # SGen_NoMS-Semi-Sup
        # 
        ############################################
        #           SGen_NoMS-Semi-Sup(f_M1)            #
        tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
            scores_UE = scores_m1_E,
            scores_U=None,
            ent_probs_U=None,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m1_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s,
            delta_e=delta_e,
            delta_p=delta_p,
            verbose=params.verbose,
            fer=fer,
            K=K,
        )

        if U_min_opt <= eps:
            res = f'[SGen_NoMS-Semi-Sup(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_NoMS-Semi-Sup(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_NoMS-Semi-Sup(f_M1)]', ld_test)
        quali_tau_s.append(tau_s_opt)
        ############################################

        ############################################
        #           SGen_NoMS-Semi-Sup(f_M2)            #
        tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
            scores_UE = scores_m2_E,
            scores_U=None,
            ent_probs_U=None,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s,
            delta_e=delta_e,
            delta_p=delta_p,
            verbose=params.verbose,
            fer=fer,
            K=K,
        )

        if U_min_opt <= eps:
            res = f'[SGen_NoMS-Semi-Sup(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}'
        else:
            res = f'[SGen_NoMS-Semi-Sup(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        with open(f'{self.params.output_dir}/results.txt', 'a') as f:
            f.write(res + '\n')
        print(res)
        print()
        ## test
        self.test(tau_s_opt, '[SGen_NoMS-Semi-Sup(f_M2)]', ld_test, feat_idx=1)
        quali_tau_s.append(tau_s_opt)
        ############################################


        # sys.exit()

    def plot(self, ld1, ld2, ld_test, updated_params=None):
        # init params
        params = copy.deepcopy(self.params)
        if updated_params:
            for a in dir(updated_params):
                if a[:2] != '__' and a[-2:] != '__':
                    setattr(params, a, getattr(updated_params, a))
        n = params.n
        self.n_e = n_e = params.n_e
        eps = params.eps
        delta = params.delta

        eps_e = params.eps_e
        delta_p = params.delta_p
        
        delta_s = (delta-delta_p)/2
        delta_e = (delta-delta_p)/2

        fer = params.fer
        K = params.K

        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'n-{n}-eps-{eps:e}-delta-{delta:e}'
        verbose = params.verbose
        
        print(f"# learn a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        # load a pre-trained model
        # if not self.params.rerun and self._check_model(best=False, is_e=True):
        #     if self.params.load_final:
        #         self._load_model(best=False, is_e=True)
        #     else:
        #         self._load_model(best=True, is_e=True)
        #     return True
        
        if self.params.cache_cal_fn:
            cache_fn = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_cal_fn}-CAL2',
            )
            cache_fn_e = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_cal_fn}-CAL2_E',
            )
            os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        else:
            cache_fn = None
            cache_fn_e = None


        # Load the results
        output_fdr = f'snapshots/box_plot/{params.exp_name}_FDR_zu-{params.z_u}_ze-{params.z_e}_epS-{eps}'
        output_failed = f'snapshots/box_plot/{params.exp_name}_n-failed_zu-{params.z_u}_ze-{params.z_e}_epS-{eps}'
        
        if os.path.exists(output_fdr) and os.path.exists(output_failed):
            print(f'Load the results from {output_fdr} and {output_failed}')
            box_plot(output_fdr, output_failed, params.exp_method, params.model, eps)
            return
    

        eff_res = []
        fdr_res = []
        if_failed_res = []

        rd1 = self.entail_model.rd['val1']
        rd2_origin = self.entail_model.rd['val2']
        rd_test_origin = self.entail_model.rd['test']

        for plot_idx in tqdm(range(100)):

            if params.cache_ent_eval_fn:
                cache_ent_fn = os.path.join(
                    params.cache_root,
                    params.cache_ent_eval_fn,
                    # self.params.cache_ent_eval_fn + f'-{self.n_e}',
                )
                if os.path.exists(cache_ent_fn):
                    os.remove(cache_ent_fn)

            if self.mdl.G.base_model == None:
                # For whisker plot
                test_size = len(rd_test_origin)

                rd2_test = concatenate_datasets([rd2_origin, rd_test_origin])
                # rd2_test = rd2_test.shuffle(params.seed)
                rd2_split = rd2_test.train_test_split(test_size=test_size, shuffle=True, seed=params.seed+plot_idx)
                rd2 = rd2_split['train']
                rd_test = rd2_split['test']

                self.entail_model.rd['val1'] = rd1
                self.entail_model.rd['val2'] = rd2
                self.entail_model.rd['test'] = rd_test


                # for new decomposition method
                self.entail_model.rd['val1+2'] = concatenate_datasets([rd1, rd2])
                rd1_2 = self.entail_model.rd['val1+2']

                scores_m1_u = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd1])
                scores_m1_e = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd2])
                scores_m1_ue = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd1_2])

                decoded_answer_u = [d['answer'] for d in rd1]
                decoded_answer_pred_u = [d['generated_answer'] for d in rd1]

                decoded_answer_e = [d['answer'] for d in rd2]
                decoded_answer_pred_e = [d['generated_answer'] for d in rd2]

                decoded_answer_ue = [d['answer'] for d in rd1_2]
                decoded_answer_pred_ue = [d['generated_answer'] for d in rd1_2]

            # Could be different from the answers in dataset if not greedy.
            else:
                
                scores_dict_u = self.precompute_scores(ld1, params.z_u, cache_fn=cache_fn)
                scores_dict_e = self.precompute_scores(ld2, params.z_e, cache_fn=cache_fn_e)

                scores_m1_u = tc.hstack([lp.mean().exp() for lp in scores_dict_u['logprobs_answer_pred']])
                scores_m1_e = tc.hstack([lp.mean().exp() for lp in scores_dict_e['logprobs_answer_pred']])
                scores_m1_ue = tc.cat([scores_m1_u, scores_m1_e], dim=0)

                decoded_answer_u = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_u['answer_ids'], skip_special_tokens=True)
                decoded_answer_pred_u = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_u['answer_ids_pred'], skip_special_tokens=True)
                decoded_answer_e = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_e['answer_ids'], skip_special_tokens=True)
                decoded_answer_pred_e = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_e['answer_ids_pred'], skip_special_tokens=True)

                decoded_answer_ue = decoded_answer_u + decoded_answer_e
                decoded_answer_pred_ue = decoded_answer_pred_u + decoded_answer_pred_e

            answer_e = [normalize_answer(a) for a in decoded_answer_e]
            answer_pred_e = [normalize_answer(a) for a in decoded_answer_pred_e]
            # errors_e = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_e, answer_pred_e)], device=scores_e.device)

            answer_u = [normalize_answer(a) for a in decoded_answer_u]
            answer_pred_u = [normalize_answer(a) for a in decoded_answer_pred_u]
            # errors_u = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_u, answer_pred_u)], device=scores_e.device)

            answer_ue = [normalize_answer(a) for a in decoded_answer_ue]
            answer_pred_ue = [normalize_answer(a) for a in decoded_answer_pred_ue]
            errors_ue = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_ue, answer_pred_ue)], device=scores_m1_ue.device)
            

            # entail modl init & score computation
            if params.entail_model in ["deberta-v2-xxlarge-mnli", None]:
                ent_probs_u = tc.hstack([1 - tc.tensor(lp['entail_scores'][0]) for lp in rd1])
                ent_labels_u = tc.hstack([tc.tensor([-1]) for lp in rd1]) # all labels are -1
                ent_probs_e = tc.hstack([1 - tc.tensor(lp['entail_scores'][0]) for lp in rd2])
                ent_labels_e = tc.hstack([tc.tensor(lp['labels']) for lp in rd2])

                # mean, entail/(entail+contradict)
                scores_m2_u = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd1])
                scores_m2_e = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd2])
                scores_m2_ue = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd1_2])
            
            else:
                entail_ld_u = self.entail_model.init_dataset_nli(
                    decoded_answer_pred_u,
                    # val1, val1+2 or val2.
                    'val1'
                )
                entail_ld_e = self.entail_model.init_dataset_nli(
                    decoded_answer_pred_e,
                    'val2'
                )
                entail_ld_sam_u = self.entail_model.init_sample_dataset_nli(
                    decoded_answer_pred_u,
                    'val1'
                )
                entail_ld_sam_e = self.entail_model.init_sample_dataset_nli(
                    decoded_answer_pred_e,
                    'val2'
                )

                if self.params.cache_ent_fn:
                    cache_ent_fn_e = os.path.join(
                        self.params.cache_root,
                        f'{self.params.cache_ent_fn}-CAL2_ZE-{params.z_e}',
                    )
                    cache_ent_fn_u = os.path.join(
                        self.params.cache_root,
                        f'{self.params.cache_ent_fn}-CAL2-ZU-{params.z_u}',
                    )
                    cache_ent_fn_sam_e = os.path.join(
                        self.params.cache_root,
                        f'{self.params.cache_ent_fn}-CAL2-SAM-ZE-{params.z_e}',
                    )
                    cache_ent_fn_sam_u = os.path.join(
                        self.params.cache_root,
                        f'{self.params.cache_ent_fn}-CAL2-SAM-ZU-{params.z_u}',
                    )
                    os.makedirs(os.path.dirname(cache_ent_fn_u), exist_ok=True)
                else:
                    cache_ent_fn_u = None
                    cache_ent_fn_e = None
                    cache_ent_fn_sam_u = None
                    cache_ent_fn_sam_e = None


                ent_scores_dict_e = self.precompute_entail_scores(entail_ld_e, cache_fn=cache_ent_fn_e)
                ent_scores_dict_u = self.precompute_entail_scores(entail_ld_u, cache_fn=cache_ent_fn_u)
                ent_scores_dict_sam_e = self.precompute_entail_scores(entail_ld_sam_e, cache_fn=cache_ent_fn_sam_e)
                ent_scores_dict_sam_u = self.precompute_entail_scores(entail_ld_sam_u, cache_fn=cache_ent_fn_sam_u)
                
                ent_probs_e = 1 - ent_scores_dict_e['probs'][..., 0]
                ent_labels_e = ent_scores_dict_e['labels']

                ent_probs_u = 1 - ent_scores_dict_u['probs'][..., 0]
                ent_labels_u = ent_scores_dict_u['labels'] # all labels are -1

                # mean, entail/(entail+contradict)
                sample_size = len(rd2[0]['samples'])
                ent_probs_sam_e = ent_scores_dict_sam_e['probs'].view(-1, sample_size, 3)
                ent_probs_sam_u = ent_scores_dict_sam_u['probs'].view(-1, sample_size, 3)
                
                scores_m2_u = (1 - ent_probs_sam_u[..., 0]).mean(dim=1)
                scores_m2_e = (1 - ent_probs_sam_e[..., 0]).mean(dim=1)
                scores_m2_ue = tc.cat([scores_m2_u, scores_m2_e], dim=0)


            # For filtering, added this temporary lines. later will be fixed
            ent_probs_ue = tc.cat([ent_probs_u, ent_probs_e], dim=0)
            ent_labels_ue = tc.cat([ent_labels_u, ent_labels_e], dim=0)


            # Z_E,U
            scores_m1_ue, i_sorted = scores_m1_ue.sort(descending=False)
            scores_m2_ue, j_sorted = scores_m2_ue.sort(descending=False)
            errors_m1_ue, errors_m2_ue = errors_ue[i_sorted], errors_ue[j_sorted]
            ent_probs_m1_ue, ent_probs_m2_ue = ent_probs_ue[i_sorted], ent_probs_ue[j_sorted]
            ent_labels_m1_ue, ent_labels_m2_ue = ent_labels_ue[i_sorted], ent_labels_ue[j_sorted]

            # These sorts are for SGen-Sup because it does need sorted score.
            scores_m1_EL, i_sorted = scores_m1_e.sort(descending=False)
            scores_m2_EL, j_sorted = scores_m2_e.sort(descending=False)
            ent_labels_m1_EL, ent_labels_m2_EL = ent_labels_e[i_sorted], ent_labels_e[j_sorted]

            # device
            device = self.device
            scores_m1_UE, scores_m2_UE = scores_m1_ue.to(device), scores_m2_ue.to(device)
            errors_m1_UE, errors_m2_UE = errors_m1_ue.to(device), errors_m2_ue.to(device)
            ent_probs_m1_UE, ent_probs_m2_UE = ent_probs_m1_ue.to(device), ent_probs_m2_ue.to(device)
            ent_labels_m1_UE, ent_labels_m2_UE = ent_labels_m1_ue.to(device), ent_labels_m2_ue.to(device)

            scores_m1_E, scores_m2_E = scores_m1_e.to(device), scores_m2_e.to(device)
            ent_probs_E, ent_labels_E = ent_probs_e.to(device), ent_labels_e.to(device)

            scores_m1_U, scores_m2_U = scores_m1_u.to(device), scores_m2_u.to(device)
            ent_probs_U = ent_probs_u.to(device)

            scores_m1_EL, scores_m2_EL = scores_m1_EL.to(device), scores_m2_EL.to(device)
            ent_labels_m1_EL, ent_labels_m2_EL = ent_labels_m1_EL.to(device), ent_labels_m2_EL.to(device)
            
            
            effs = {}
            fdrs = {}
            if_failed = {}

            ############################################
            #            SGen_EM(f_M1)             #
            tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                scores=scores_m1_UE,
                eps=eps,
                delta=delta,
                verbose=params.verbose,
                errors=errors_m1_UE,
                fer=fer,
            )
            if U_min_opt <= eps:
                # print(f'[SGen_EM(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_EM(f_M1)'] = True
            else:
                # print(f'[SGen_EM(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_EM(f_M1)'] = False

            ## test
            res = self.test(tau_s_opt, '[SGen_EM(f_M1)]', ld_test, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_EM(f_M1)'] = eff
            fdrs['SGen_EM(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################
            ############################################
            #            SGen_EM(f_M2)             #
            tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                scores=scores_m2_UE,
                eps=eps,
                delta=delta,
                verbose=params.verbose,
                errors=errors_m2_UE,
                fer=fer,
            )
            if U_min_opt <= eps:
                # print(f'[SGen_EM(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_EM(f_M2)'] = True
            else:
                # print(f'[SGen_EM(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_EM(f_M2)'] = False

            ## test
            res = self.test(tau_s_opt, '[SGen_EM(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_EM(f_M2)'] = eff
            fdrs['SGen_EM(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #            SGen_PL-H-Semi(f_M1)             #
            tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                scores_UE=scores_m1_UE,
                scores_U=scores_m1_U,
                ent_probs_U=ent_probs_U,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m1_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta=delta,
                verbose=params.verbose,
                fer=fer,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_PL-H-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_PL-H-Semi(f_M1)'] = True
            else:
                # print(f'[SGen_PL-H-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_PL-H-Semi(f_M1)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_PL-H-Semi(f_M1)]', ld_test, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_PL-H-Semi(f_M1)'] = eff
            fdrs['SGen_PL-H-Semi(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #            SGen_PL-H-Semi(f_M2)             #
            tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                scores_UE=scores_m2_UE,
                scores_U=scores_m2_U,
                ent_probs_U=ent_probs_U,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m2_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta=delta,
                verbose=params.verbose,
                fer=fer,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_PL-H-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_PL-H-Semi(f_M2)'] = True
            else:
                # print(f'[SGen_PL-H-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_PL-H-Semi(f_M2)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_PL-H-Semi(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_PL-H-Semi(f_M2)'] = eff
            fdrs['SGen_PL-H-Semi(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #       SGen_PL-H-Semi(f_M1, filtering)       #
            tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                scores_UE=scores_m1_UE,
                scores_U=scores_m1_U,
                ent_probs_U=ent_probs_U,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m1_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta=delta,
                verbose=params.verbose,
                filtering=True,
                ent_probs_UE=ent_probs_m1_UE,
                ent_labels_UE=ent_labels_m1_UE,
                fer=fer,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_PFL-H-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_PFL-H-Semi(f_M1)'] = True
            else:
                # print(f'[SGen_PFL-H-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_PFL-H-Semi(f_M1)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_PFL-H-Semi(f_M1)]', ld_test, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_PFL-H-Semi(f_M1)'] = eff
            fdrs['SGen_PFL-H-Semi(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #       SGen_PL-H-Semi(f_M2, filtering)       #
            tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                scores_UE=scores_m2_UE,
                scores_U=scores_m2_U,
                ent_probs_U=ent_probs_U,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m2_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta=delta,
                verbose=params.verbose,
                filtering=True,
                ent_probs_UE=ent_probs_m2_UE,
                ent_labels_UE=ent_labels_m2_UE,
                fer=fer,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_PFL-H-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_PFL-H-Semi(f_M2)'] = True
            else:
                # print(f'[SGen_PFL-H-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_PFL-H-Semi(f_M2)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_PFL-H-Semi(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_PFL-H-Semi(f_M2)'] = eff
            fdrs['SGen_PFL-H-Semi(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #             SGen_NoMS-Semi(f_M1)              #
            tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                scores_UE = scores_m1_UE,
                scores_U=scores_m1_U,
                ent_probs_U=ent_probs_U,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m1_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta_s=delta_s,
                delta_e=delta_e,
                delta_p=delta_p,
                verbose=params.verbose,
                fer=fer,
                K=K,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_NoMS-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_NoMS-Semi(f_M1)'] = True
            else:
                # print(f'[SGen_NoMS-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_NoMS-Semi(f_M1)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_NoMS-Semi(f_M1)]', ld_test, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_NoMS-Semi(f_M1)'] = eff
            fdrs['SGen_NoMS-Semi(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #             SGen_NoMS-Semi(f_M2)              #
            tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                scores_UE = scores_m2_UE,
                scores_U=scores_m2_U,
                ent_probs_U=ent_probs_U,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m2_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta_s=delta_s,
                delta_e=delta_e,
                delta_p=delta_p,
                verbose=params.verbose,
                fer=fer,
                K=K,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_NoMS-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_NoMS-Semi(f_M2)'] = True
            else:
                # print(f'[SGen_NoMS-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_NoMS-Semi(f_M2)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_NoMS-Semi(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_NoMS-Semi(f_M2)'] = eff
            fdrs['SGen_NoMS-Semi(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################


            ############################################
            #            SGen-Sup(f_M1)             #
            tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                # ent_labels=ent_labels_E1, # this is implemented in errors.
                scores=scores_m1_EL,
                eps=eps,
                delta=delta,
                verbose=params.verbose,
                errors=(ent_labels_m1_EL==0),
                fer=fer,
            )

            if U_min_opt <= eps:
                # print(f'[SGen-Sup(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen-Sup(f_M1)'] = True
            else:
                # print(f'[SGen-Sup(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen-Sup(f_M1)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen-Sup(f_M1)]', ld_test, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen-Sup(f_M1)'] = eff
            fdrs['SGen-Sup(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################
            #            SGen-Sup(f_M2)             #
            tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                # ent_labels=ent_labels_E2, # this is implemented in errors.
                scores=scores_m2_EL,
                eps=eps,
                delta=delta,
                verbose=params.verbose,
                errors=(ent_labels_m2_EL==0),
                fer=fer,
            )

            if U_min_opt <= eps:
                # print(f'[SGen-Sup(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen-Sup(f_M2)'] = True
            else:
                # print(f'[SGen-Sup(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen-Sup(f_M2)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen-Sup(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen-Sup(f_M2)'] = eff
            fdrs['SGen-Sup(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################


            ############################################
            #               SGen-Semi               #
            #          with new decomposition          #
            tau_s_opt, feat_idx, U_min_opt = SG_MS.train(
                scores_m1_UE=scores_m1_UE,
                scores_m2_UE=scores_m2_UE,
                scores_m1_E=scores_m1_E,
                scores_m2_E=scores_m2_E,
                ent_probs_E=ent_probs_E,
                ent_labels_E=ent_labels_E,
                scores_m1_U=scores_m1_U,
                scores_m2_U=scores_m2_U,
                ent_probs_U=ent_probs_U,
                eps=eps,
                eps_e=eps_e,
                delta_s=delta_s,
                delta_e=delta_e,
                delta_p=delta_p,
                verbose=params.verbose,
                fer=fer,
                K=K,
            )
            if_failed['SGen-Semi'] = True if U_min_opt <= eps else False

            
            res = self.test(tau_s_opt, '[SGen-Semi]', ld_test, feat_idx, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen-Semi'] = eff
            fdrs['SGen-Semi'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            # print('#'*20)
            ############################################

            # SGen_NoMS-Semi-Sup
            # 
            ############################################
            #           SGen_NoMS-Semi-Sup(f_M1)            #
            tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                scores_UE = scores_m1_E,
                scores_U=None,
                ent_probs_U=None,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m1_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta_s=delta_s,
                delta_e=delta_e,
                delta_p=delta_p,
                verbose=params.verbose,
                fer=fer,
                K=K,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_NoMS-Semi-Sup(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_NoMS-Semi-Sup(f_M1)'] = True
            else:
                # print(f'[SGen_NoMS-Semi-Sup(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_NoMS-Semi-Sup(f_M1)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_NoMS-Semi-Sup(f_M1)]', ld_test, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_NoMS-Semi-Sup(f_M1)'] = eff
            fdrs['SGen_NoMS-Semi-Sup(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################

            ############################################
            #           SGen_NoMS-Semi-Sup(f_M2)            #
            tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                scores_UE = scores_m2_E,
                scores_U=None,
                ent_probs_U=None,
                ent_labels_E=ent_labels_E,
                scores_E=scores_m2_E,
                ent_probs_E=ent_probs_E,
                eps=eps,
                eps_e=eps_e,
                delta_s=delta_s,
                delta_e=delta_e,
                delta_p=delta_p,
                verbose=params.verbose,
                fer=fer,
                K=K,
            )

            if U_min_opt <= eps:
                # print(f'[SGen_NoMS-Semi-Sup(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                if_failed['SGen_NoMS-Semi-Sup(f_M2)'] = True
            else:
                # print(f'[SGen_NoMS-Semi-Sup(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                if_failed['SGen_NoMS-Semi-Sup(f_M2)'] = False
            ## test
            res = self.test(tau_s_opt, '[SGen_NoMS-Semi-Sup(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
            eff = res["sel_test"].float().mean()
            effs['SGen_NoMS-Semi-Sup(f_M2)'] = eff
            fdrs['SGen_NoMS-Semi-Sup(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
            ############################################


            fdr_res.append(fdrs)
            eff_res.append(effs)
            if_failed_res.append(if_failed)

        methods = fdr_res[0].keys()
        result = {}
        falied_result = {}
        for method in methods:
            result[method] = [i[method].item() for i in fdr_res]
            falied_result[method] = [i[method] for i in if_failed_res]

            
        # Ensure the directories exist
        os.makedirs(f'snapshots/box_plot', exist_ok=True)

        # Save the results
        pickle.dump(result, open(output_fdr, 'wb'))
        pickle.dump(falied_result, open(output_failed, 'wb'))

        box_plot(output_fdr, output_failed, params.exp_method, params.model, eps)

    def quan_plot(self, ld1, ld2, ld_test, updated_params=None):
        # init params
        params = copy.deepcopy(self.params)
        if updated_params:
            for a in dir(updated_params):
                if a[:2] != '__' and a[-2:] != '__':
                    setattr(params, a, getattr(updated_params, a))
        n = params.n
        self.n_e = n_e = params.n_e
        eps = params.eps
        delta = params.delta

        eps_e = params.eps_e
        delta_p = params.delta_p
        
        delta_s = (delta-delta_p)/2
        delta_e = (delta-delta_p)/2

        fer = params.fer
        K = params.K

        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'n-{n}-eps-{eps:e}-delta-{delta:e}'
        verbose = params.verbose
        
        print(f"# learn a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        # load a pre-trained model
        # if not self.params.rerun and self._check_model(best=False, is_e=True):
        #     if self.params.load_final:
        #         self._load_model(best=False, is_e=True)
        #     else:
        #         self._load_model(best=True, is_e=True)
        #     return True

        if self.params.cache_cal_fn:
            cache_fn = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_cal_fn}-CAL2',
            )
            cache_fn_e = os.path.join(
                self.params.cache_root,
                f'{self.params.cache_cal_fn}-CAL2_E',
            )
            os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        else:
            cache_fn = None
            cache_fn_e = None
        

        
        # shuffled only once for each zu_size
        rd1_origin = self.entail_model.rd['val1']
        rd2_origin = self.entail_model.rd['val2']
        rd_test_origin = self.entail_model.rd['test']

        # TODO params
        if params.model == 'gpt3.5':
            ablations = [1000, 3000, 5000, 10000]
        elif params.model == 'alpaca7B':
            ablations = [10000, 15000, 20000, 25000]
            
        # Check the results (temp, not modularized)
        models = {
            'gpt3.5': [[1000, 3000, 5000, 10000], 2757],
            'alpaca7B': [[10000, 15000, 20000, 25000], 4424]
        }
        if_cached_all = True
        output_fdrs = {'gpt3.5': [], 'alpaca7B': []}
        output_effs = {'gpt3.5': [], 'alpaca7B': []}
        output_faileds = {'gpt3.5': [], 'alpaca7B': []}
        for model, abs in models.items():
            abus, abe = abs[0], abs[1]
            for abu in abus:
                output_fdr = f'snapshots/quan_plot/{params.tag}-nli-nq_{model}-{model}-GreedyGen-SGQuanPlot-EXP-SSL_FDR_zu-{abu}_ze-{abe}_epS-{eps}'
                output_eff = f'snapshots/quan_plot/{params.tag}-nli-nq_{model}-{model}-GreedyGen-SGQuanPlot-EXP-SSL_EFF_zu-{abu}_ze-{abe}_epS-{eps}'
                output_failed = f'snapshots/quan_plot/{params.tag}-nli-nq_{model}-{model}-GreedyGen-SGQuanPlot-EXP-SSL_n-failed_zu-{abu}_ze-{abe}_epS-{eps}'
                if os.path.exists(output_fdr) and os.path.exists(output_failed) and os.path.exists(output_eff):
                    output_fdrs[model].append(output_fdr)
                    output_effs[model].append(output_eff)
                    output_faileds[model].append(output_failed)
                    continue
                if_cached_all = False
        if if_cached_all:
            print(f'Load the results from {output_fdr} and {output_failed}')
            quan_plot(output_fdrs, output_effs, output_faileds, models, eps)
            return
        
        for ablation in ablations:
            rd1 = rd1_origin.select(range(ablation))
            eff_res = []
            fdr_res = []
            if_failed_res = []

            for plot_idx in tqdm(range(10)):

                if params.cache_ent_eval_fn:
                    cache_ent_fn = os.path.join(
                        params.cache_root,
                        params.cache_ent_eval_fn,
                        # self.params.cache_ent_eval_fn + f'-{self.n_e}',
                    )
                    if os.path.exists(cache_ent_fn):
                        os.remove(cache_ent_fn)

                if self.mdl.G.base_model == None:
                    # For whisker plot
                    test_size = len(rd_test_origin)

                    rd2_test = concatenate_datasets([rd2_origin, rd_test_origin])
                    rd2_split = rd2_test.train_test_split(test_size=test_size, shuffle=True, seed=params.seed+plot_idx)
                    rd2 = rd2_split['train']
                    rd_test = rd2_split['test']

                    self.entail_model.rd['val1'] = rd1
                    self.entail_model.rd['val2'] = rd2
                    self.entail_model.rd['test'] = rd_test

                    # for new decomposition method
                    self.entail_model.rd['val1+2'] = concatenate_datasets([rd1, rd2])
                    rd1_2 = self.entail_model.rd['val1+2']

                    scores_m1_u = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd1])
                    scores_m1_e = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd2])
                    scores_m1_ue = tc.hstack([tc.tensor(lp['logprobs']).mean().exp() for lp in rd1_2])

                    decoded_answer_u = [d['answer'] for d in rd1]
                    decoded_answer_pred_u = [d['generated_answer'] for d in rd1]

                    decoded_answer_e = [d['answer'] for d in rd2]
                    decoded_answer_pred_e = [d['generated_answer'] for d in rd2]

                    decoded_answer_ue = [d['answer'] for d in rd1_2]
                    decoded_answer_pred_ue = [d['generated_answer'] for d in rd1_2]

                # Could be different from the answers in dataset if not greedy.
                else:

                    scores_dict_u = self.precompute_scores(ld1, params.z_u, cache_fn=cache_fn)
                    scores_dict_e = self.precompute_scores(ld2, params.z_e, cache_fn=cache_fn_e)

                    scores_m1_u = tc.hstack([lp.mean().exp() for lp in scores_dict_u['logprobs_answer_pred']])
                    scores_m1_e = tc.hstack([lp.mean().exp() for lp in scores_dict_e['logprobs_answer_pred']])
                    scores_m1_ue = tc.cat([scores_m1_u, scores_m1_e], dim=0)

                    decoded_answer_u = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_u['answer_ids'], skip_special_tokens=True)
                    decoded_answer_pred_u = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_u['answer_ids_pred'], skip_special_tokens=True)
                    decoded_answer_e = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_e['answer_ids'], skip_special_tokens=True)
                    decoded_answer_pred_e = self.mdl.G.base_model.tokenizer.batch_decode(scores_dict_e['answer_ids_pred'], skip_special_tokens=True)

                    decoded_answer_ue = decoded_answer_u + decoded_answer_e
                    decoded_answer_pred_ue = decoded_answer_pred_u + decoded_answer_pred_e            

                answer_e = [normalize_answer(a) for a in decoded_answer_e]
                answer_pred_e = [normalize_answer(a) for a in decoded_answer_pred_e]
                # errors_e = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_e, answer_pred_e)], device=scores_e.device)

                answer_u = [normalize_answer(a) for a in decoded_answer_u]
                answer_pred_u = [normalize_answer(a) for a in decoded_answer_pred_u]
                # errors_u = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_u, answer_pred_u)], device=scores_e.device)

                # For SG-EM
                answer_ue = [normalize_answer(a) for a in decoded_answer_ue]
                answer_pred_ue = [normalize_answer(a) for a in decoded_answer_pred_ue]
                errors_ue = tc.tensor([not compute_EM(a_gold, a_pred) for a_gold, a_pred in zip(answer_ue, answer_pred_ue)], device=scores_m1_ue.device)

                # entail modl init & score computation
                if params.entail_model in ["deberta-v2-xxlarge-mnli", None]:
                    ent_probs_u = tc.hstack([1 - tc.tensor(lp['entail_scores'][0]) for lp in rd1])
                    ent_labels_u = tc.hstack([tc.tensor([-1]) for lp in rd1]) # all labels are -1
                    ent_probs_e = tc.hstack([1 - tc.tensor(lp['entail_scores'][0]) for lp in rd2])
                    ent_labels_e = tc.hstack([tc.tensor(lp['labels']) for lp in rd2])

                    # mean, entail/(entail+contradict)
                    scores_m2_u = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd1])
                    scores_m2_e = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd2])
                    scores_m2_ue = tc.hstack([(1 - tc.tensor(ss['samples_scores'])[..., 0]).mean() for ss in rd1_2])
                
                else:
                    entail_ld_u = self.entail_model.init_dataset_nli(
                        decoded_answer_pred_u,
                        # val1, val1+2 or val2.
                        'val1'
                    )
                    entail_ld_e = self.entail_model.init_dataset_nli(
                        decoded_answer_pred_e,
                        'val2'
                    )
                    entail_ld_sam_u = self.entail_model.init_sample_dataset_nli(
                        decoded_answer_pred_u,
                        'val1'
                    )
                    entail_ld_sam_e = self.entail_model.init_sample_dataset_nli(
                        decoded_answer_pred_e,
                        'val2'
                    )

                    if self.params.cache_ent_fn:
                        cache_ent_fn_e = os.path.join(
                            self.params.cache_root,
                            f'{self.params.cache_ent_fn}-CAL2_ZE-{params.z_e}',
                        )
                        cache_ent_fn_u = os.path.join(
                            self.params.cache_root,
                            f'{self.params.cache_ent_fn}-CAL2-ZU-{params.z_u}',
                        )
                        cache_ent_fn_sam_e = os.path.join(
                            self.params.cache_root,
                            f'{self.params.cache_ent_fn}-CAL2-SAM-ZE-{params.z_e}',
                        )
                        cache_ent_fn_sam_u = os.path.join(
                            self.params.cache_root,
                            f'{self.params.cache_ent_fn}-CAL2-SAM-ZU-{params.z_u}',
                        )
                        os.makedirs(os.path.dirname(cache_ent_fn_u), exist_ok=True)
                    else:
                        cache_ent_fn_u = None
                        cache_ent_fn_e = None
                        cache_ent_fn_sam_u = None
                        cache_ent_fn_sam_e = None


                    ent_scores_dict_e = self.precompute_entail_scores(entail_ld_e, cache_fn=cache_ent_fn_e)
                    ent_scores_dict_u = self.precompute_entail_scores(entail_ld_u, cache_fn=cache_ent_fn_u)
                    ent_scores_dict_sam_e = self.precompute_entail_scores(entail_ld_sam_e, cache_fn=cache_ent_fn_sam_e)
                    ent_scores_dict_sam_u = self.precompute_entail_scores(entail_ld_sam_u, cache_fn=cache_ent_fn_sam_u)
                    
                    ent_probs_e = 1 - ent_scores_dict_e['probs'][..., 0]
                    ent_labels_e = ent_scores_dict_e['labels']

                    ent_probs_u = 1 - ent_scores_dict_u['probs'][..., 0]
                    ent_labels_u = ent_scores_dict_u['labels'] # all labels are -1

                    # mean, entail/(entail+contradict)
                    sample_size = len(rd2[0]['samples'])
                    ent_probs_sam_e = ent_scores_dict_sam_e['probs'].view(-1, sample_size, 3)
                    ent_probs_sam_u = ent_scores_dict_sam_u['probs'].view(-1, sample_size, 3)
                    
                    scores_m2_u = (1 - ent_probs_sam_u[..., 0]).mean(dim=1)
                    scores_m2_e = (1 - ent_probs_sam_e[..., 0]).mean(dim=1)
                    scores_m2_ue = tc.cat([scores_m2_u, scores_m2_e], dim=0)


                # For filtering, added this temporary lines. later will be fixed
                ent_probs_ue = tc.cat([ent_probs_u, ent_probs_e], dim=0)
                ent_labels_ue = tc.cat([ent_labels_u, ent_labels_e], dim=0)


                # Z_E,U
                scores_m1_ue, i_sorted = scores_m1_ue.sort(descending=False)
                scores_m2_ue, j_sorted = scores_m2_ue.sort(descending=False)
                errors_m1_ue, errors_m2_ue = errors_ue[i_sorted], errors_ue[j_sorted]
                ent_probs_m1_ue, ent_probs_m2_ue = ent_probs_ue[i_sorted], ent_probs_ue[j_sorted]
                ent_labels_m1_ue, ent_labels_m2_ue = ent_labels_ue[i_sorted], ent_labels_ue[j_sorted]

                # These sorts are for SGen-Sup because it does need sorted score.
                scores_m1_EL, i_sorted = scores_m1_e.sort(descending=False)
                scores_m2_EL, j_sorted = scores_m2_e.sort(descending=False)
                ent_labels_m1_EL, ent_labels_m2_EL = ent_labels_e[i_sorted], ent_labels_e[j_sorted]

                # device
                device = self.device
                scores_m1_UE, scores_m2_UE = scores_m1_ue.to(device), scores_m2_ue.to(device)
                errors_m1_UE, errors_m2_UE = errors_m1_ue.to(device), errors_m2_ue.to(device)
                ent_probs_m1_UE, ent_probs_m2_UE = ent_probs_m1_ue.to(device), ent_probs_m2_ue.to(device)
                ent_labels_m1_UE, ent_labels_m2_UE = ent_labels_m1_ue.to(device), ent_labels_m2_ue.to(device)

                scores_m1_E, scores_m2_E = scores_m1_e.to(device), scores_m2_e.to(device)
                ent_probs_E, ent_labels_E = ent_probs_e.to(device), ent_labels_e.to(device)

                scores_m1_U, scores_m2_U = scores_m1_u.to(device), scores_m2_u.to(device)
                ent_probs_U = ent_probs_u.to(device)

                scores_m1_EL, scores_m2_EL = scores_m1_EL.to(device), scores_m2_EL.to(device)
                ent_labels_m1_EL, ent_labels_m2_EL = ent_labels_m1_EL.to(device), ent_labels_m2_EL.to(device)
                
                
                effs = {}
                fdrs = {}
                if_failed = {}

                ############################################
                #            SGen_EM(f_M1)             #
                tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                    scores=scores_m1_UE,
                    eps=eps,
                    delta=delta,
                    verbose=params.verbose,
                    errors=errors_m1_UE,
                    fer=fer,
                )
                if U_min_opt <= eps:
                    # print(f'[SGen_EM(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_EM(f_M1)'] = True
                else:
                    # print(f'[SGen_EM(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_EM(f_M1)'] = False

                ## test
                res = self.test(tau_s_opt, '[SGen_EM(f_M1)]', ld_test, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_EM(f_M1)'] = eff
                fdrs['SGen_EM(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################
                ############################################
                #            SGen_EM(f_M2)             #
                tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                    scores=scores_m2_UE,
                    eps=eps,
                    delta=delta,
                    verbose=params.verbose,
                    errors=errors_m2_UE,
                    fer=fer,
                )
                if U_min_opt <= eps:
                    # print(f'[SGen_EM(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_EM(f_M2)'] = True
                else:
                    # print(f'[SGen_EM(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_EM(f_M2)'] = False

                ## test
                res = self.test(tau_s_opt, '[SGen_EM(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_EM(f_M2)'] = eff
                fdrs['SGen_EM(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #            SGen_PL-H-Semi(f_M1)             #
                tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                    scores_UE=scores_m1_UE,
                    scores_U=scores_m1_U,
                    ent_probs_U=ent_probs_U,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m1_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta=delta,
                    verbose=params.verbose,
                    fer=fer,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_PL-H-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_PL-H-Semi(f_M1)'] = True
                else:
                    # print(f'[SGen_PL-H-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_PL-H-Semi(f_M1)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_PL-H-Semi(f_M1)]', ld_test, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_PL-H-Semi(f_M1)'] = eff
                fdrs['SGen_PL-H-Semi(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #            SGen_PL-H-Semi(f_M2)             #
                tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                    scores_UE=scores_m2_UE,
                    scores_U=scores_m2_U,
                    ent_probs_U=ent_probs_U,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m2_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta=delta,
                    verbose=params.verbose,
                    fer=fer,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_PL-H-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_PL-H-Semi(f_M2)'] = True
                else:
                    # print(f'[SGen_PL-H-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_PL-H-Semi(f_M2)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_PL-H-Semi(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_PL-H-Semi(f_M2)'] = eff
                fdrs['SGen_PL-H-Semi(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #       SGen_PL-H-Semi(f_M1, filtering)       #
                tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                    scores_UE=scores_m1_UE,
                    scores_U=scores_m1_U,
                    ent_probs_U=ent_probs_U,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m1_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta=delta,
                    verbose=params.verbose,
                    filtering=True,
                    ent_probs_UE=ent_probs_m1_UE,
                    ent_labels_UE=ent_labels_m1_UE,
                    fer=fer,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_PFL-H-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_PFL-H-Semi(f_M1)'] = True
                else:
                    # print(f'[SGen_PFL-H-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_PFL-H-Semi(f_M1)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_PFL-H-Semi(f_M1)]', ld_test, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_PFL-H-Semi(f_M1)'] = eff
                fdrs['SGen_PFL-H-Semi(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #       SGen_PL-H-Semi(f_M2, filtering)       #
                tau_s_opt, U_min_opt, eff = SG_Heuristic.train(
                    scores_UE=scores_m2_UE,
                    scores_U=scores_m2_U,
                    ent_probs_U=ent_probs_U,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m2_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta=delta,
                    verbose=params.verbose,
                    filtering=True,
                    ent_probs_UE=ent_probs_m2_UE,
                    ent_labels_UE=ent_labels_m2_UE,
                    fer=fer,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_PFL-H-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_PFL-H-Semi(f_M2)'] = True
                else:
                    # print(f'[SGen_PFL-H-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_PFL-H-Semi(f_M2)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_PFL-H-Semi(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_PFL-H-Semi(f_M2)'] = eff
                fdrs['SGen_PFL-H-Semi(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #             SGen_NoMS-Semi(f_M1)              #
                tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                    scores_UE = scores_m1_UE,
                    scores_U=scores_m1_U,
                    ent_probs_U=ent_probs_U,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m1_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta_s=delta_s,
                    delta_e=delta_e,
                    delta_p=delta_p,
                    verbose=params.verbose,
                    fer=fer,
                    K=K,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_NoMS-Semi(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_NoMS-Semi(f_M1)'] = True
                else:
                    # print(f'[SGen_NoMS-Semi(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_NoMS-Semi(f_M1)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_NoMS-Semi(f_M1)]', ld_test, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_NoMS-Semi(f_M1)'] = eff
                fdrs['SGen_NoMS-Semi(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #             SGen_NoMS-Semi(f_M2)              #
                tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                    scores_UE = scores_m2_UE,
                    scores_U=scores_m2_U,
                    ent_probs_U=ent_probs_U,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m2_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta_s=delta_s,
                    delta_e=delta_e,
                    delta_p=delta_p,
                    verbose=params.verbose,
                    fer=fer,
                    K=K,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_NoMS-Semi(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_NoMS-Semi(f_M2)'] = True
                else:
                    # print(f'[SGen_NoMS-Semi(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_NoMS-Semi(f_M2)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_NoMS-Semi(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_NoMS-Semi(f_M2)'] = eff
                fdrs['SGen_NoMS-Semi(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################


                ############################################
                #            SGen-Sup(f_M1)             #
                tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                    # ent_labels=ent_labels_E1, # this is implemented in errors.
                    scores=scores_m1_EL,
                    eps=eps,
                    delta=delta,
                    verbose=params.verbose,
                    errors=(ent_labels_m1_EL==0),
                    fer=fer,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen-Sup(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen-Sup(f_M1)'] = True
                else:
                    # print(f'[SGen-Sup(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen-Sup(f_M1)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen-Sup(f_M1)]', ld_test, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen-Sup(f_M1)'] = eff
                fdrs['SGen-Sup(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################
                #            SGen-Sup(f_M2)             #
                tau_s_opt, U_min_opt, eff = SG_Baseline.train(
                    # ent_labels=ent_labels_E2, # this is implemented in errors.
                    scores=scores_m2_EL,
                    eps=eps,
                    delta=delta,
                    verbose=params.verbose,
                    errors=(ent_labels_m2_EL==0),
                    fer=fer,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen-Sup(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen-Sup(f_M2)'] = True
                else:
                    # print(f'[SGen-Sup(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen-Sup(f_M2)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen-Sup(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen-Sup(f_M2)'] = eff
                fdrs['SGen-Sup(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################


                ############################################
                #               SGen-Semi               #
                #          with new decomposition          #
                tau_s_opt, feat_idx, U_min_opt = SG_MS.train(
                    scores_m1_UE=scores_m1_UE,
                    scores_m2_UE=scores_m2_UE,
                    scores_m1_E=scores_m1_E,
                    scores_m2_E=scores_m2_E,
                    ent_probs_E=ent_probs_E,
                    ent_labels_E=ent_labels_E,
                    scores_m1_U=scores_m1_U,
                    scores_m2_U=scores_m2_U,
                    ent_probs_U=ent_probs_U,
                    eps=eps,
                    eps_e=eps_e,
                    delta_s=delta_s,
                    delta_e=delta_e,
                    delta_p=delta_p,
                    verbose=params.verbose,
                    fer=fer,
                    K=K,
                )
                if_failed['SGen-Semi'] = True if U_min_opt <= eps else False

                
                res = self.test(tau_s_opt, '[SGen-Semi]', ld_test, feat_idx, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen-Semi'] = eff
                fdrs['SGen-Semi'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                # print('#'*20)
                ############################################

                # SGen_NoMS-Semi-Sup
                # 
                ############################################
                #           SGen_NoMS-Semi-Sup(f_M1)            #
                tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                    scores_UE = scores_m1_E,
                    scores_U=None,
                    ent_probs_U=None,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m1_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta_s=delta_s,
                    delta_e=delta_e,
                    delta_p=delta_p,
                    verbose=params.verbose,
                    fer=fer,
                    K=K,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_NoMS-Semi-Sup(f_M1) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_NoMS-Semi-Sup(f_M1)'] = True
                else:
                    # print(f'[SGen_NoMS-Semi-Sup(f_M1) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_NoMS-Semi-Sup(f_M1)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_NoMS-Semi-Sup(f_M1)]', ld_test, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_NoMS-Semi-Sup(f_M1)'] = eff
                fdrs['SGen_NoMS-Semi-Sup(f_M1)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################

                ############################################
                #           SGen_NoMS-Semi-Sup(f_M2)            #
                tau_s_opt, U_j, U_min_opt, eff = SG_Semi.train(
                    scores_UE = scores_m2_E,
                    scores_U=None,
                    ent_probs_U=None,
                    ent_labels_E=ent_labels_E,
                    scores_E=scores_m2_E,
                    ent_probs_E=ent_probs_E,
                    eps=eps,
                    eps_e=eps_e,
                    delta_s=delta_s,
                    delta_e=delta_e,
                    delta_p=delta_p,
                    verbose=params.verbose,
                    fer=fer,
                    K=K,
                )

                if U_min_opt <= eps:
                    # print(f'[SGen_NoMS-Semi-Sup(f_M2) success] U_min (={U_min_opt:.4e}) <= eps (={eps}), efficiency = {eff}, tau = {tau_s_opt}') # U_min?
                    if_failed['SGen_NoMS-Semi-Sup(f_M2)'] = True
                else:
                    # print(f'[SGen_NoMS-Semi-Sup(f_M2) fail] U_min (={U_min_opt:.4e}) > eps (={eps}), tau = {tau_s_opt}')
                    if_failed['SGen_NoMS-Semi-Sup(f_M2)'] = False
                ## test
                res = self.test(tau_s_opt, '[SGen_NoMS-Semi-Sup(f_M2)]', ld_test, feat_idx=1, verbose=params.verbose)
                eff = res["sel_test"].float().mean()
                effs['SGen_NoMS-Semi-Sup(f_M2)'] = eff
                fdrs['SGen_NoMS-Semi-Sup(f_M2)'] = (res["e_0_test"].long().sum()/res["e_0_test"].shape[0])
                ############################################


                fdr_res.append(fdrs)
                eff_res.append(effs)
                if_failed_res.append(if_failed)

            methods = fdr_res[0].keys()
            result = {}
            eff_result = {}
            falied_result = {}
            for method in methods:
                eff_result[method] = [i[method].item() for i in eff_res]
                result[method] = [i[method].item() for i in fdr_res]
                falied_result[method] = [i[method] for i in if_failed_res]

            
            # Ensure the directories exist
            os.makedirs(f'snapshots/quan_plot', exist_ok=True)

            output_fdr = f'snapshots/quan_plot/{params.exp_name}_FDR_zu-{ablation}_ze-{params.z_e}_epS-{eps}'
            output_eff = f'snapshots/quan_plot/{params.exp_name}_EFF_zu-{ablation}_ze-{params.z_e}_epS-{eps}'
            output_failed = f'snapshots/quan_plot/{params.exp_name}_n-failed_zu-{ablation}_ze-{params.z_e}_epS-{eps}'

            # Save the results
            pickle.dump(result, open(output_fdr, 'wb'))
            pickle.dump(eff_result, open(output_eff, 'wb'))
            pickle.dump(falied_result, open(output_failed, 'wb'))

        print(f'The results are saved in {output_fdr}, {output_eff}, {output_failed}. You need to learn all models manually. Check /uncertainty/util.py')

# SG
class SG_MS(SCGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, entail_model=entail_model, params=params, name_postfix=name_postfix)
        
    @classmethod
    def train(
        cls,
        scores_m1_UE,
        scores_m2_UE,
        scores_m1_E,
        scores_m2_E,
        ent_probs_E,
        ent_labels_E,
        scores_m1_U,
        scores_m2_U,
        ent_probs_U,
        eps,
        eps_e,
        delta_s,
        delta_e,
        delta_p,
        verbose,
        fer,
        K,
        params=None,
    ):

        U_opt = 0
        temp = []
        tau_s_opt, U_j_opt1, U_min_opt1, eff_opt1 = SG_Semi.train(
            scores_UE=scores_m1_UE,
            scores_U=scores_m1_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m1_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s/3,
            delta_e=delta_e/3,
            delta_p=delta_p/3,
            verbose=verbose,
            fer=fer,
            K=K,
        )
        temp.append(tau_s_opt)

        if U_j_opt1 <= eps and U_j_opt1 > U_opt:
            U_opt = U_j_opt1
            feat_idx = 0

        if U_min_opt1 <= eps:
            res = f'[SGen-Semi-1 success] U_j (={U_j_opt1:.4e}), U_min (={U_min_opt1:.4e}) <= eps (={eps}), efficiency = {eff_opt1}, tau = {tau_s_opt}'
        else:
            res = f'[SGen-Semi-1 fail] U_j (={U_j_opt1:.4e}), U_min (={U_min_opt1:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        if params is not None:
            with open(f'{params.output_dir}/results.txt', 'a') as f:
                f.write(res + '\n')
            print(res)

        tau_s_opt, U_j_opt2, U_min_opt2, eff_opt2 = SG_Semi.train(
            scores_UE=scores_m2_UE,
            scores_U=scores_m2_U,
            ent_probs_U=ent_probs_U,
            ent_labels_E=ent_labels_E,
            scores_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s/3,
            delta_e=delta_e/3,
            delta_p=delta_p/3,
            verbose=verbose,
            fer=fer,
            K=K,
        )
        temp.append(tau_s_opt)

        if U_j_opt2 <= eps and U_j_opt2 > U_opt:
            U_opt = U_j_opt2
            feat_idx = 1

        if U_min_opt2 <= eps:
            res = f'[SGen-Semi-2 success] U_j (={U_j_opt2:.4e}), U_min (={U_min_opt2:.4e}) <= eps (={eps}), efficiency = {eff_opt2}, tau = {tau_s_opt}'
        else:
            res = f'[SGen-Semi-2 fail] U_j (={U_j_opt2:.4e}), U_min (={U_min_opt2:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        if params is not None:
            with open(f'{params.output_dir}/results.txt', 'a') as f:
                f.write(res + '\n')
            print(res)

        tau_s_i_opt, tau_s_j_opt, U_j_opt3, U_min_opt3, eff_opt3 = SG_Semi2.train(
            scores_m1_UE=scores_m1_UE,
            scores_m2_UE=scores_m2_UE,
            scores_m1_E=scores_m1_E,
            scores_m2_E=scores_m2_E,
            ent_probs_E=ent_probs_E,
            ent_labels_E=ent_labels_E,
            scores_m1_U=scores_m1_U,
            scores_m2_U=scores_m2_U,
            ent_probs_U=ent_probs_U,
            eps=eps,
            eps_e=eps_e,
            delta_s=delta_s/3,
            delta_e=delta_e/3,
            delta_p=delta_p/3,
            verbose=verbose,
            fer=fer,
            K=K,
        )
        temp.append((tau_s_i_opt, tau_s_j_opt))

        if U_j_opt3 <= eps and U_j_opt3 > U_opt:
            U_opt = U_j_opt3
            feat_idx = 2

        if U_min_opt3 <= eps:
            res = f'[SGen-Semi-3 success] U_j (={U_j_opt3:.4e}), U_min (={U_min_opt3:.4e}) <= eps (={eps}), efficiency = {eff_opt3}, tau = {(tau_s_i_opt, tau_s_j_opt)}'
        else:
            res = f'[SGen-Semi-3 fail] U_j (={U_j_opt3:.4e}), U_min (={U_min_opt3:.4e}) > eps (={eps}), tau = {tau_s_opt}'
        if params is not None:
            with open(f'{params.output_dir}/results.txt', 'a') as f:
                f.write(res + '\n')
            print(res)


        
        if min(U_min_opt1, U_min_opt2, U_min_opt3) > eps:
            feat_idx = [U_j_opt1, U_j_opt2, U_j_opt3].index(min([U_j_opt1, U_j_opt2, U_j_opt3]))
            res = f'[SGen-Semi fail] U_min (={min(U_min_opt1, U_min_opt2, U_min_opt3):.4e}) <= eps (={eps}), tau = {tau_s_opt}'
        if params is not None:
            with open(f'{params.output_dir}/results.txt', 'a') as f:
                f.write(res + '\n')
            print(res)
        tau_s_opt = temp[feat_idx]

        ## test
        # feat_idx = [U_min_opt1, U_min_opt2, U_min_opt3].index(min([U_min_opt1, U_min_opt2, U_min_opt3]))
        # tau_s_opt = temp[feat_idx]

        return tau_s_opt, feat_idx, min([U_min_opt1, U_min_opt2, U_min_opt3])


# Ours(revised) using over-approximating
# SG_Semi
class SG_Semi(SCGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, entail_model=entail_model, params=params, name_postfix=name_postfix)
    
    @classmethod
    def train(
        cls,
        scores_UE,
        scores_U,
        ent_probs_U,
        ent_labels_E,
        scores_E,
        ent_probs_E,
        eps,
        eps_e,
        delta_s,
        delta_e,
        delta_p,
        verbose,
        fer,
        K,
    ):
        
        n = scores_UE.shape[0]
        # binary search
        i_min = 0
        i_max = n-1
        n_iters = int(np.ceil(np.log2(n))) + 1
        U_min = float('inf')
        U_min_cur = -1
        for i in range(n_iters):
            i_cur = int(np.ceil((i_min + i_max) / 2))
            tau_s_opt = scores_UE[i_cur]
            
            # Z'
            prob_ = ent_probs_E[(scores_E >= tau_s_opt).bool()]
            e_ = ent_labels_E[(scores_E >= tau_s_opt).bool()]
            z_ = (prob_, e_)
            if prob_.shape[0] == 0:
                i_min = i_cur
                continue

            z_u = ent_probs_U[(scores_U >= tau_s_opt).bool()] if scores_U is not None else None
            U_j, tau_e = cls.fdr_e_upperbound(z_, z_u, eps, delta_s/n_iters, eps_e, delta_e/n_iters, delta_p/n_iters, fer=fer, K=K)

            # print(U_j, tau_s_opt.item())
            if (U_j <= U_min):
                U_min_cur = i_cur
                U_min = U_j
            # U_min = min(U_min, U_j)

            if U_j <= eps:
                i_max = i_cur
            else:
                i_min = i_cur

            if verbose:
                # print(f'[success] tau_e = {tau_e_opt}') if tau_e_opt != float('inf') and tau_e_opt != 0 else print(f'[fail] tau_e = {tau_e_opt}')
                print(
                    f'[binary serach for tau_s] '
                    f'tau = {scores_UE[i_cur]:.4e}, '
                    f'U = {U_j:.4e}, '
                    f'U_j = {U_j:.4e}, eps = {eps}, eps_e = {eps_e}, '
                    f'SelGen(i_min, i_max) = ({i_min}, {i_max})')
                print()

        tau_s_opt = scores_UE[i_cur]
        
        if U_min <= eps:
            # print(f'[success] U (={U_j:.4e}) <= eps (={eps}), tau = {tau_s_opt}')
            pass
        else:
            # print(f'[fail] U (={U_min:.4e}) > eps (={eps}), tau = {tau_s_opt}')
            tau_s_opt = scores_UE[U_min_cur]
        # print(tau_e)
        if tau_e >= 1:
            print(f'[entailment set fail] tau_e = {tau_e}')

        return tau_s_opt, U_j, U_min, ((scores_UE>=tau_s_opt).sum()/n)


# Ours(revised) using multi-tau
# SG_Semi2
class SG_Semi2(SCGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, entail_model=entail_model, params=params, name_postfix=name_postfix)
        
    @classmethod
    def train(
        cls,
        scores_m1_UE,
        scores_m2_UE,
        scores_m1_E,
        scores_m2_E,
        ent_probs_E,
        ent_labels_E,
        scores_m1_U,
        scores_m2_U,
        ent_probs_U,
        eps,
        eps_e,
        delta_s,
        delta_e,
        delta_p,
        verbose,
        fer,
        K,
    ):

        n = scores_m1_UE.shape[0]

        # binary search
        i_min = 0
        i_max = n-1
        n_iters = int(np.ceil(np.log2(n))) + 1
        U_min = float('inf')
        U_min_i_cur = -1
        U_min_j_cur = -1

        for i in range(n_iters):
            i_cur = int(np.ceil((i_min + i_max) / 2))
            tau_s_i = scores_m1_UE[i_cur]
            j_min = 0
            j_max = n-1
            n_iters = int(np.ceil(np.log2(n))) + 1
            U_min_i = float('inf')
            for j in range(n_iters):
                j_cur = int(np.ceil((j_min + j_max) / 2))
                tau_s_j = scores_m2_UE[j_cur]

                prob_ = ent_probs_E[((scores_m1_E >= tau_s_i) * (scores_m2_E >= tau_s_j)).bool()]
                e_ = ent_labels_E[((scores_m1_E >= tau_s_i) * (scores_m2_E >= tau_s_j)).bool()]
                if prob_.shape[0] == 0:
                    j_min = j_cur
                    break
                z_ = (prob_, e_)

                z_u = ent_probs_U[((scores_m1_U >= tau_s_i) * (scores_m2_U >= tau_s_j)).bool()]

                U_ij, tau_e = cls.fdr_e_upperbound(z_, z_u, eps, delta_s/(n_iters**2), eps_e, delta_e/(n_iters**2), delta_p/(n_iters**2), fer=fer, K=K)

                U_min_i = min(U_min_i, U_ij)
                if (U_ij <= U_min):
                    U_min_i_cur = i_cur
                    U_min_j_cur = j_cur
                    # U_min_i = U_ij

                if U_ij <= eps:
                    j_max = j_cur
                else:
                    j_min = j_cur

            if (U_min_i <= U_min and U_min > eps):
                U_min_i_cur = i_cur
                U_min_j_cur = j_cur
                U_min = U_min_i
            
            if U_min_i <= eps:
                i_max = i_cur
            else:
                i_min = i_cur

            if verbose:
                print(
                    f'[binary serach for tau_i] '
                    f'tau_i = {scores_m1_UE[i_cur]:.4e}, '
                    f'tau_j = {scores_m2_UE[j_cur]:.4e}, '
                    # f'1 - prec = {k_j} / {n_j} = {k_j / n_j:.4e}, '
                    f'U = {U_min:.4e}, '
                    # f'l / |Z_E| = {l} / {n_l} = {l / n_l:.4e}, '
                    # f'U_ = {U_:.4e}, '
                    # f'U_j = {U_j:.4e}, eps = {eps}, eps_e = {eps_e}, '
                    f'SelGen(i_min, i_max) = ({i_min}, {i_max})')
                print()

        tau_s_i_opt = scores_m1_UE[i_cur]
        tau_s_j_opt = scores_m2_UE[j_cur]


        if U_min <= eps:
            # print(f'[success] U (={U_ij:.4e}) <= eps (={eps}), tau_i = {tau_s_i_opt}, tau_j = {tau_s_j_opt}')
            pass
        else:
            # print(f'[fail] U (={U_min:.4e}) > eps (={eps}), tau = {tau_s_i_opt}, tau_j = {tau_s_j_opt}')
            tau_s_i_opt = scores_m1_UE[U_min_i_cur]
            tau_s_j_opt = scores_m2_UE[U_min_j_cur]
            
        if tau_e >= 1:
            print(f'[entailment set fail] tau_e = {tau_e}')

        return tau_s_i_opt, tau_s_j_opt, U_ij, U_min, (((scores_m1_UE >= tau_s_i) * (scores_m2_UE >= tau_s_j)).sum() / n)
    

# heuristic tau_E
# SGen_PL-H-Semi
class SG_Heuristic(SCGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, entail_model=entail_model, params=params, name_postfix=name_postfix)
    
    @classmethod
    def train(
        cls,
        scores_UE,
        scores_U,
        ent_probs_U,
        ent_labels_E,
        scores_E,
        ent_probs_E,
        eps,
        eps_e,
        delta,
        verbose,
        fer,
        filtering=False,
        ent_probs_UE=None,
        ent_labels_UE=None,
    ):
        
        tau_pl = 0.9 # temp. Adjust this value if you want to see other results of SG-PFL.
        if filtering:
            filter_idx_u = ((ent_probs_U >= tau_pl) | (1 - ent_probs_U >= tau_pl)).bool()
            scores_U = scores_U[filter_idx_u]
            ent_probs_U = ent_probs_U[filter_idx_u]

            filter_idx_ue = ((ent_labels_UE == -1) * ((ent_probs_UE >= tau_pl) | (1 - ent_probs_UE >= tau_pl))).bool()
            scores_UE = scores_UE[filter_idx_ue]
            ent_probs_UE = ent_probs_UE[filter_idx_ue]

            # print('filterd:', filter_idx_u.sum())
            assert filter_idx_u.sum() == filter_idx_ue.sum()

        n = scores_UE.shape[0]
        # binary search
        i_min = 0
        i_max = n-1
        n_iters = int(np.ceil(np.log2(n))) + 1
        U_min = float('inf')
        U_min_cur = -1


        for i in range(n_iters):
            i_cur = int(np.ceil((i_min + i_max) / 2))
            tau_s_opt = scores_UE[i_cur] # not neccessary
            n_j = (scores_UE >= tau_s_opt).sum()
            k_j = ((ent_labels_E == 0) * (scores_E >= tau_s_opt)).sum() + ((ent_probs_U <= tau_pl) * (scores_U >= tau_s_opt)).sum() # the second term is for pseudo-labeling
            
            U_j = clopper_pearson_worst(k_j.item(), n_j.item(), delta/n_iters)
            if U_j <= eps:
                i_max = i_cur
            else:
                i_min = i_cur

            if (U_j <= U_min):
                U_min_cur = i_cur
                U_min = U_j

            if verbose:
                # print(f'[success] tau_e = {tau_e_opt}') if tau_e_opt != float('inf') and tau_e_opt != 0 else print(f'[fail] tau_e = {tau_e_opt}')
                print(
                    f'[binary serach for tau_s] '
                    f'tau = {scores_UE[i_cur]:.4e}, '
                    f'U = {U_j:.4e}, '
                    f'U_j = {U_j:.4e}, eps = {eps}, eps_e = {eps_e}, '
                    f'SelGen(i_min, i_max) = ({i_min}, {i_max})')
                print()

        tau_s_opt = scores_UE[i_cur]

        if U_min <= eps:
            # print(f'[success] U (={U_j:.4e}) <= eps (={eps}), tau = {tau_s_opt}')
            pass
        else:
            # print(f'[fail] U (={U_min:.4e}) > eps (={eps}), tau = {tau_s_opt}')
            tau_s_opt = scores_UE[U_min_cur]

        # print(tau_s_opt, U_j, (n_j.item()/n))
        # print(k_j/n_j)
        return tau_s_opt, U_min, (n_j.item()/n)
    
    
# Baselines
class SG_Baseline(SCGBaseLearner):
    
    def __init__(self, model, entail_model, params=None, name_postfix=None):
        super().__init__(model=model, entail_model=entail_model, params=params, name_postfix=name_postfix)
        
    @classmethod
    def train(
        cls,
        scores,
        eps, # eps
        delta, #delta+delta_e
        verbose,
        errors, # error function
        fer,
    ):
        n = scores.shape[0]
        # binary search
        i_min = 0
        i_max = n-1
        n_iters = int(np.ceil(np.log2(n))) + 1
        TER_max = 0
        U_min = float('inf')
        U_min_cur = -1

        for i in range(n_iters):
            i_cur = int(np.ceil((i_min + i_max) / 2))
            tau_s_opt = scores[i_cur] # not neccessary?
            n_j = (scores >= tau_s_opt).sum()
            k_j = (errors * (scores >= tau_s_opt)).sum()
            U = clopper_pearson_worst(k_j.item(), n_j.item(), delta/n_iters)
            if U <= eps:
                i_max = i_cur
            else:
                i_min = i_cur

            if (U <= U_min):
                U_min_cur = i_cur
                U_min = U

            if verbose:
                print(f'[binary serach for tau] '
                      f'tau = {scores[i_cur]:.4e}, '
                      f'1 - prec = {k_j} / {n_j} = {k_j / n_j:.4e}, '
                      f'U = {U:.4e}, eps = {eps}, '
                      f'(i_min, i_max) = ({i_min}, {i_max})')


        tau_s_opt = scores[i_cur]

        if U_min > eps:
            tau_s_opt = scores[U_min_cur]

        return tau_s_opt, U_min, (n_j.item()/n)

