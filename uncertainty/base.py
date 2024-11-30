import os, sys
from learning import *
import numpy as np
import pickle
import types
import itertools
import scipy
from abc import abstractmethod
from tqdm import tqdm
import time
import pickle
import glob
import math

import torch as tc
from .util import *

from uncertainty import ECE
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer

from models.util import compute_inclusion


class LGBaseLearner(BaseLearner):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model, params, name_postfix)

        self.G = self.mdl


    def precompute_generation(self, ld, desc, n=None):
        t_start = time.time()        
        output_list = []
        if n:
            total = math.ceil(n/ld.batch_size)
        else:
            total = None
            
        for x, y in tqdm(ld, total=total, desc=desc):
            x = to_device(x, self.params.device)
            
            with tc.no_grad():
                output = self.mdl.generate(x)
                    
            output_list.append(to_device(output, 'cpu'))
        print(f'[precompute generation] duration = {time.time() - t_start:.2f} sec.')
  
        return output_list
    
        
    #def precompute_scores(self, ld, n=None, cache=False, cache_fn=None): # remove cache_fn
    def precompute_scores(self, ld, n=None, cache_fn=None):

        # pre-generation
        if cache_fn and os.path.exists(cache_fn):
            print(f'[pre-generation] loading precomputed generation results from {cache_fn}')
            output_list = pickle.load(open(cache_fn, 'rb'))
        else:
            output_list = self.precompute_generation(ld, desc="Precompute generation results")
            if cache_fn:
                pickle.dump(output_list, open(cache_fn, 'wb'))
                print(f'[pre-generation] saving precomputed generation results to {cache_fn}')
    

        # post-process
        ds = None
        for output in output_list:
            # un-batching
            output = {k: [v_i.cpu() for v_i in v] if tc.is_tensor(v) else v for k, v in output.items()}
            if ds is None:
                ds = output
            else:
                ds = {k: ds[k] + output[k] for k in ds.keys()}
            
        # import json
        # if len(ds['logprobs_answer_pred']) == 83490:
        #     data_json = json.load(open('/home/saemin21/sg-llm/data/nli/nq_alpaca7B/nq_Z_u.json', 'r'))
        #     for data, log in zip(data_json, ds['logprobs_answer_pred']):
        #         data['logprobs'] = log.tolist()
        #     json.dump(data_json, open('/home/saemin21/sg-llm/data/nli/nq_alpaca7B/nq_Z_u_log.json', 'w'), indent=4)
        # # elif len(ds['logprobs_answer_pred']) == 4607:
        # #     data_json = json.load(open('/home/saemin21/sg-llm/data/nli/nq_alpaca7B/nq_val.json', 'r'))

        # #     for data, log in zip(data_json, ds['logprobs_answer_pred']):
        # #         data['logprobs'] = log.tolist()
        # #     json.dump(data_json, open('/home/saemin21/sg-llm/data/nli/nq_alpaca7B/nq_val_log.json', 'w'), indent=4)
        # else:
        #     raise ValueError
        # sys.exit()

        # #t_start = time.time()        
        # ds = None
        # for x, y in tqdm(ld, total=math.ceil(n/ld.batch_size), desc="Precompute generation results"):
        #     x = to_device(x, self.params.device)
        #     #y = to_device(y, self.params.device)

        #     with tc.no_grad():
        #         output = self.G.generate(x)
        #     # un-batching
        #     output = {k: [v_i.cpu() for v_i in v] if tc.is_tensor(v) else v for k, v in output.items()}
        #     # # cpu
        #     # output = {k: [v_i.cpu() for v_i in v] for k, v in output.items()}
        #     if ds is None:
        #         ds = output
        #     else:
        #         ds = {k: ds[k] + output[k] for k in ds.keys()}

                
        #     if len(ds[list(ds.keys())[0]]) >= n:
        #         break

        # truncate scores
        if n:
            ds = {k: v[:n] for k, v in ds.items()}
        n_read = len(ds[list(ds.keys())[0]])
            
        #print(f'[duration = {time.time() - t_start:.2f} sec.] n = {n_read}')
            
        if n:
            assert n_read == n, f'actual data is less than n ({n_read} != {n})'
        
        return ds

    
    def eval_generation(self, kwargs):

        answer = self.G.base_model.tokenizer.batch_decode(kwargs['answer_ids'], skip_special_tokens=True)
        answer_pred = self.G.base_model.tokenizer.batch_decode(kwargs['answer_ids_pred'], skip_special_tokens=True)

        # EM
        em = [compute_exact(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

        # F1
        f1 = [compute_f1(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

        # inclusion
        inc = [compute_inclusion(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

        # selection
        ##TODO: save scores before selection...
        scores = kwargs['scores_cal']  if 'scores_cal' in kwargs else kwargs['scores']
        sel = scores >= self.mdl.logtau_model.to(scores.device)()['logits'].exp()        
        # if 'selection' in kwargs:
        #     sel = kwargs['selection']
        # else:
        #     sel = tc.tensor([True] * len(answer))

        # for ECE
        ph = scores
        # if 'cal_probs' in kwargs:
        #     ph = [p.item() for p in kwargs['cal_probs']]
        # else:
        #     ph = [lp.sum().exp().item() for lp in kwargs['logprobs_answer_pred']]

        
        yh = [normalize_answer(a) for a in answer_pred]
        y = [normalize_answer(a) for a in answer]

        return {'EM': em, 'F1': f1, 'Inc': inc, 'ph': ph, 'y': y, 'yh': yh, 'sel': sel}


    
    def compute_metrics(self, ld):
        em, f1, inc = [], [], []
        yh_ece, y_ece, ph_ece = [], [], []
        sel = []
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
        if cache_fn and os.path.exists(cache_fn):
            print(f'[pre-generation] loading precomputed generation results from {cache_fn}')
            output_list = pickle.load(open(cache_fn, 'rb'))
        else:
            for x, y in tqdm(ld, desc="Precompute generation results"):
                x = to_device(x, self.params.device)
                
                with tc.no_grad():
                    output = self.mdl.generate(x)
                output_list.append(to_device(output, 'cpu'))
            if cache_fn:
                pickle.dump(output_list, open(cache_fn, 'wb'))
                print(f'[pre-generation] saving precomputed generation results to {cache_fn}')
        

        # compute metrics
        for output in tqdm(output_list, desc="compute metrics"):

            llm_stat_i = self.eval_generation(output)

            # save
            em.append(tc.tensor(llm_stat_i['EM']))
            f1.append(tc.tensor(llm_stat_i['F1']))
            inc.append(tc.tensor(llm_stat_i['Inc']))
            sel.append(llm_stat_i['sel'])
            
            y_ece += llm_stat_i['y']
            yh_ece += llm_stat_i['yh']
            ph_ece += llm_stat_i['ph']

            # q/a/a_pred
            q.extend(self.G.base_model.tokenizer.batch_decode(output['question_ids'], skip_special_tokens=True))
            a.extend(self.G.base_model.tokenizer.batch_decode(output['answer_ids'], skip_special_tokens=True))
            a_pred.extend(self.G.base_model.tokenizer.batch_decode(output['answer_ids_pred'], skip_special_tokens=True))        

        # # compute metrics
        # for x, y in tqdm(ld, desc="compute metrics"):
        #     x = to_device(x, self.params.device)

        #     with tc.no_grad():
        #         output = self.mdl.generate(x)

        #     llm_stat_i = self.eval_generation(output)

        #     # save
        #     em.append(tc.tensor(llm_stat_i['EM']))
        #     f1.append(tc.tensor(llm_stat_i['F1']))
        #     inc.append(tc.tensor(llm_stat_i['Inc']))
        #     sel.append(llm_stat_i['sel'])
            
        #     y_ece += llm_stat_i['y']
        #     yh_ece += llm_stat_i['yh']
        #     ph_ece += llm_stat_i['ph']
                        
        #     # #DBG
        #     # rel_diag_fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', 'rel_diag')
        #     # print('[DBG] ECE =', ECE(np.array(ph_ece), np.array(yh_ece), np.array(y_ece), is_correct=self.G.is_correct, rel_diag_fn=rel_diag_fn))
        #     # print(f'inclusion = {tc.cat(inc).float().mean()}')

        # concat
        em, f1, inc, sel = tc.cat(em).cpu(), tc.cat(f1).cpu(), tc.cat(inc).cpu(), tc.cat(sel).cpu()
        
        # compute ECE
        rel_diag_fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', 'rel_diag')
        ece = ECE(
            np.array(ph_ece),
            np.array(yh_ece),
            np.array(y_ece),
            is_correct=self.G.is_correct,
            rel_diag_fn=rel_diag_fn
        )
        
        ece_sel = ECE(
            np.array(ph_ece)[sel],
            np.array(yh_ece)[sel],
            np.array(y_ece)[sel],
            is_correct=self.G.is_correct,
            rel_diag_fn=os.path.join(self.params.snapshot_root,
                                     self.params.exp_name,
                                     'figs',
                                     'rel_diag_selected')
        )

        ret = {
            'EM': em, 'F1': f1, 'Inc': inc, 'ECE': ece, 'ECE_sel': ece_sel, 'sel': sel
        }
        ret_json = {
            'question': q,
            'answer':a,
            'generated_answer': a_pred,
            'EM_test': em.tolist(),
            'F1_test': f1.tolist(),
            'Inc_test': inc.tolist(),
            'sel_test': sel.tolist()
        }
        
        return ret, ret_json

    
    def test(self, ld, ld_name, verbose=False, save=True):

        # compute basic metrics for language generators
        fn = os.path.join(self.params.snapshot_root,
                          self.params.exp_name,
                          f'stats_pred_set_{self.name_postfix}.pk'
                          if self.name_postfix else 'stats_pred_set.pk')
        if os.path.exists(fn) and not self.params.rerun:
            print(f'load precomputed results at {fn}')
            res = pickle.load(open(fn, 'rb'))
        else:
            metrics, ref = self.compute_metrics(ld)
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
                f'[test: {ld_name}]\n'
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
                f'ECE (sel) = {res["ECE_sel_test"]*100:.4f}%'
            )
            print('==================================================')
            print()
            
        return res


class SGBaseLearner(LGBaseLearner):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model, params, name_postfix)

        self.G = self.mdl.G
        

        
# ##TODO: legacy
# class PSBaseLearner(BaseLearner):
#     def __init__(self, model, params=None, name_postfix=None):
#         super().__init__(model, params, name_postfix)

        
#     def train(self, ld):
#         # do nothing
#         return True

    
#     def compute_metrics(self, ld):
#         size, size_ideal, error = [], [], []
#         for x, y in ld:
#             # size_i = loss_set_size(x, self.mdl, reduction='none', device=self.params.device)['loss']
#             # error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
            
#             output = loss_set_error_size(x, y, self.mdl, reduction='none', device=self.params.device)
#             error_i = output['loss_error']
#             size_i = output['loss_size']
#             size_ideal_i = output['loss_size_ideal']
#             error.append(error_i)
#             size.append(size_i)
#             size_ideal.append(size_ideal_i)
#             raise NotImplementedError #TODO: return logprob
#         size, size_ideal, error = tc.cat(size).cpu(), tc.cat(size_ideal).cpu(), tc.cat(error).cpu()

#         return size, size_ideal, error
        

#     def test(self, ld, ld_name, verbose=False, save=True):

#         ## compute set size and error
#         fn = os.path.join(self.params.snapshot_root, self.params.exp_name, f'stats_pred_set_{self.name_postfix}.pk' if self.name_postfix else 'stats_pred_set.pk')
#         if os.path.exists(fn) and not self.params.rerun:
#         #if False:
#             print(f'load precomputed results at {fn}')
#             res = pickle.load(open(fn, 'rb'))
#         else:
#             metrics = self.compute_metrics(ld)
#             for k in [k for k in metrics.keys()]:
#                 metrics[k + '_test'] = metrics.pop(k)
            
#             res = {
#                 **metrics,
#                 'n': self.mdl.n if hasattr(self.mdl, 'n') else None,
#                 'eps': self.mdl.eps if hasattr(self.mdl, 'eps') else None,
#                 'delta': self.mdl.delta if hasattr(self.mdl, 'delta') else None
#             }
#             if save:
#                 pickle.dump(res, open(fn, 'wb'))

#         if verbose:
#             size = res['size_test']
#             size_ideal = res['size_ideal_test']
#             error = res['error_test']

#             mn = size.min()
#             Q1 = size.kthvalue(int(round(size.size(0)*0.25)))[0]
#             Q2 = size.median()
#             Q3 = size.kthvalue(int(round(size.size(0)*0.75)))[0]
#             mx = size.max()
#             av = size.mean()

#             mn_id = size_ideal.min()
#             Q1_id = size_ideal.kthvalue(int(round(size_ideal.size(0)*0.25)))[0]
#             Q2_id = size_ideal.median()
#             Q3_id = size_ideal.kthvalue(int(round(size_ideal.size(0)*0.75)))[0]
#             mx_id = size_ideal.max()
#             av_id = size_ideal.mean()

#             print(
#                 f'[test: {ld_name}, '
#                 f'n = {self.mdl.n if hasattr(self.mdl, "n") else "None"}, '
#                 f'eps = {self.mdl.eps if hasattr(self.mdl, "eps") else "None"}, '
#                 f'delta = {self.mdl.delta if hasattr(self.mdl, "delta") else "None"}] '
#                 f'error = {error.mean():.4f}, \n'
#                 f'(size) min = {mn}, 1st-Q = {Q1}, median = {Q2}, 3rd-Q = {Q3}, max = {mx}, mean = {av:.2f}, \n'
#                 f'(size_ideal) min = {mn_id}, 1st-Q = {Q1_id}, median = {Q2_id}, 3rd-Q = {Q3_id}, max = {mx_id}, mean = {av_id:.2f}'                
#             )

#             ## plot results

            
#         return res


    


# ##TODO: use class LGBaseLearner(BaseLearner):    
# class PSLLMBaseLearner(PSBaseLearner):
    
#     def __init__(self, model, params=None, name_postfix=None):
#         super().__init__(model, params, name_postfix)

        
#     def precompute_scores(self, ld, n=None, cache_fn=None):

#         assert(cache_fn is None)

#         self.mdl = self.mdl.to(self.params.device)

#         t_start = time.time()        
#         ds = None
#         for x, y in tqdm(ld, desc="Precompute log-probabilities"):
#             x = to_device(x, self.params.device)
#             #y = to_device(y, self.params.device)

#             with tc.no_grad():
#                 output = self.mdl.generate(x)
#             # un-batching
#             output = {k: [v_i for v_i in v] if tc.is_tensor(v) else v for k, v in output.items()}
#             # cpu
#             output = {k: [v_i.cpu() for v_i in v] for k, v in output.items()}
#             if ds is None:
#                 ds = output
#             else:
#                 ds = {k: ds[k] + output[k] for k in ds.keys()}

                
#             if len(ds[list(ds.keys())[0]]) >= n:
#                 break

#         # truncate scores
#         ds = {k: v[:n] for k, v in ds.items()}
#         n_read = len(ds[list(ds.keys())[0]])
            
#         print(f'[compute scores, duration = {time.time() - t_start:.2f} sec.] n = {n_read}')
            
#         if n:
#             assert n_read == n, f'actual data is less than n ({n_read} != {n})'
        
#         return ds

    
#     def eval_generation(self, kwargs):

#         answer = self.mdl.base_model.tokenizer.batch_decode(kwargs['answer_ids'], skip_special_tokens=True)
#         answer_pred = self.mdl.base_model.tokenizer.batch_decode(kwargs['answer_ids_pred'], skip_special_tokens=True)

#         # EM
#         em = [compute_exact(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

#         # F1
#         f1 = [compute_f1(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

#         # inclusion
#         inc = [compute_inclusion(a_gold, a_pred) for a_gold, a_pred in zip(answer, answer_pred)]

#         # for ECE
#         if 'cal_probs' in kwargs:
#             ph = [p.item() for p in kwargs['cal_probs']]
#         else:
#             ph = [lp.sum().exp().item() for lp in kwargs['logprobs_answer_pred']]

        
#         yh = [normalize_answer(a) for a in answer_pred]
#         y = [normalize_answer(a) for a in answer]

#         return {'EM': em, 'F1': f1, 'Inc': inc, 'ph': ph, 'y': y, 'yh': yh}
        
    
#     def compute_metrics(self, ld):
#         size, size_ideal, error, logprob, em, f1, inc = [], [], [], [], [], [], []
#         yh_ece, y_ece, ph_ece = [], [], []

#         # compute metrics
#         n_computed = 0
#         for x, y in tqdm(ld, desc="compute metrics"):
#             x = to_device(x, self.params.device)

#             with tc.no_grad():
#                 output = self.mdl.generate(x)

#             #TODO: enable this
#             # ps_stat_i = self.mdl.error_size(output={
#             #     'hidden_states': output['hidden_states'],
#             #     'logprobs': output['logprobs'],
#             #     'logprobs_y': output['logprobs_answer']
#             # })
#             llm_stat_i = self.eval_generation(output)

#             # save
#             #TODO: enable this
#             # error.append(ps_stat_i['error'].cpu().float())
#             # size.append(ps_stat_i['size'].cpu())
#             # size_ideal.append(ps_stat_i['size_ideal'].cpu())
#             logprob.append(output['logprobs'].cpu().numpy())
#             em.append(tc.tensor(llm_stat_i['EM']))
#             f1.append(tc.tensor(llm_stat_i['F1']))
#             inc.append(tc.tensor(llm_stat_i['Inc']))

#             # # drop an empty string in evaluation
#             # y_i, yh_i, ph_i = [], [], []
#             # for y_ii, yh_ii, ph_ii in zip(llm_stat_i['y'], llm_stat_i['yh'], llm_stat_i['ph']):
#             #     if yh_ii != '':
#             #         y_i.append(y_ii)
#             #         yh_i.append(yh_ii)
#             #         ph_i.append(ph_ii)
                
#             # y_ece += y_i
#             # yh_ece += yh_i
#             # ph_ece += ph_i
            
#             y_ece += llm_stat_i['y']
#             yh_ece += llm_stat_i['yh']
#             ph_ece += llm_stat_i['ph']
            
            
#             # #DBG
#             # if any(np.array(llm_stat_i['ph']) > 0.9):
#             #     ph_np = np.array(llm_stat_i['ph'])
#             #     yh_np = np.array(llm_stat_i['yh'])
#             #     y_np = np.array(llm_stat_i['y'])
                
#             #     i_dbg = ph_np > 0.9
                
#             #     print('ph =', ph_np[i_dbg])
#             #     print('yh =', yh_np[i_dbg])
#             #     print('y =', y_np[i_dbg])

#             #DBG
#             rel_diag_fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', 'rel_diag')
#             print('[DBG] ECE =', ECE(np.array(ph_ece), np.array(yh_ece), np.array(y_ece), is_correct=self.mdl.is_correct, rel_diag_fn=rel_diag_fn))
#             print(f'inclusion = {tc.cat(inc).float().mean()}')

#         # size, size_ideal, error = tc.cat(size), tc.cat(size_ideal), tc.cat(error)
#         em, f1, inc = tc.cat(em), tc.cat(f1), tc.cat(inc)

#         # compute ECE
#         rel_diag_fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'figs', 'rel_diag')
#         ece = ECE(np.array(ph_ece), np.array(yh_ece), np.array(y_ece), is_correct=self.mdl.is_correct, rel_diag_fn=rel_diag_fn)

#         ret = {
#             #'size': size, 'size_ideal': size_ideal, 'error': error, #'logprob': logprob,
#             'EM': em, 'F1': f1, 'Inc': inc, 'ECE': ece
#         }
        
#         print(ret)

#         return ret
    

    
#     def test(self, ld, ld_name, verbose=False, save=True):
#         res = super().test(ld=ld, ld_name=ld_name, verbose=verbose, save=save)
#         print(f'EM = {res["EM_test"].float().mean():.4f}, F1 = {res["F1_test"].float().mean():.4f}, ECE = {res["ECE_test"]*100:.4f}%')
        
#         return res
        
