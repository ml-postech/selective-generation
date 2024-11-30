import os, sys
import time
import numpy as np
import copy

import torch as tc
from torch import nn, optim

class BaseLearner:
    
    def __init__(self, mdl, params=None, name_postfix=None):
        self.params = params
        self.mdl = mdl
        self.name_postfix = name_postfix
        self.loss_fn_train = None
        self.loss_fn_val = None
        self.loss_fn_test = None
        if params:
            self.mdl_fn_best = os.path.join(params.snapshot_root, params.exp_name, 'model_params%s_best') 
            self.mdl_fn_final = os.path.join(params.snapshot_root, params.exp_name, 'model_params%s_final')
            self.mdl_fn_chkp = os.path.join(params.snapshot_root, params.exp_name, 'model_params%s_chkp')


    def _load_model(self, best=True, tag='', is_e=False):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
            e_model_fn = self.mdl_fn_best%('_'+self.name_postfix+'_deberta-v3'if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
            e_model_fn = self.mdl_fn_best%('_'+self.name_postfix+'_deberta-v3'if self.name_postfix else '')
        model_fn = model_fn + tag
        
        print(f'[the {"best" if best else "final" } model is loaded] {model_fn}')
        self.mdl.load_state_dict(tc.load(model_fn, map_location='cpu'))
        try: self.entail_model.load_state_dict(tc.load(e_model_fn, map_location='cpu'))
        except: return model_fn

        
    def _save_model(self, best=True, tag='', is_e=False):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
            # MJ: temp
            e_model_fn = self.mdl_fn_best%('_'+self.name_postfix+'_deberta-v3'if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
            # MJ: temp
            e_model_fn = self.mdl_fn_final%('_'+self.name_postfix+'_deberta-v3' if self.name_postfix else '')
        model_fn = model_fn + tag

        os.makedirs(os.path.dirname(model_fn), exist_ok=True)
        tc.save(self.mdl.state_dict(), model_fn)
        try: tc.save(self.entail_model.state_dict(), e_model_fn)
        except: return model_fn

    
    def _check_model(self, best=True, tag='', is_e=False):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
            e_model_fn = self.mdl_fn_best%('_'+self.name_postfix+'_deberta-v3'if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
            e_model_fn = self.mdl_fn_best%('_'+self.name_postfix+'_deberta-v3'if self.name_postfix else '')
        model_fn = model_fn + tag

        # MJ: temp
        try:
            print(f'[there is the {"best" if best else "final" } entail model.]')
            return os.path.exists(model_fn) and os.path.exists(e_model_fn)
        except: return os.path.exists(model_fn)
        
    
    def _save_chkp(self):
        model_fn = self.mdl_fn_chkp%('_'+self.name_postfix if self.name_postfix else '')
        chkp = {
            'epoch': self.i_epoch,
            'mdl_state': self.mdl.state_dict(),
            'opt_state': self.opt.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'error_val_best': self.error_val_best,
        }
        tc.save(chkp, model_fn)
        return model_fn


    def _load_chkp(self, chkp_fn):
        return tc.load(chkp_fn, map_location=tc.device('cpu'))

    
    def train(self, ld_tr, ld_val=None, ld_test=None):

        ## load a model
        if not self.params.rerun and not self.params.resume and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return
        
        self._train_begin(ld_tr, ld_val, ld_test)
        for i_epoch in range(self.epoch_init, self.params.n_epochs+1):
            self.i_epoch = i_epoch
            self._train_epoch_begin(i_epoch)
            self._train_epoch(i_epoch, ld_tr)
            self._train_epoch_end(i_epoch, ld_val, ld_test)
        self._train_end(ld_val, ld_test)
        

    def validate(self, ld):
        return self.test(ld, mdl=self.mdl, loss_fn=self.loss_fn_val)

    
    def test(self, ld, model=None, loss_fn=None):
        model = model if model else self.mdl
        loss_fn = loss_fn if loss_fn else self.loss_fn_test
        loss_vec = []
        with tc.no_grad():
            for x, y in ld:
                loss_dict = loss_fn(x, y, model, reduction='none', device=self.params.device)
                loss_vec.append(loss_dict['loss'])
        loss_vec = tc.cat(loss_vec)
        loss = loss_vec.mean()
        return loss,
            

    def _train_begin(self, ld_tr, ld_val, ld_test):
        self.time_train_begin = time.time()
        
        ## init an optimizer
        if self.params.optimizer == "Adam":
            self.opt = optim.Adam(self.mdl.parameters(), lr=self.params.lr)
        elif self.params.optimizer == "AMSGrad":
            self.opt = optim.Adam(self.mdl.parameters(), lr=self.params.lr, amsgrad=True)
        elif self.params.optimizer == "SGD":
            self.opt = optim.SGD(self.mdl.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        else:
            raise NotImplementedError
        
        ## init a lr scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt, self.params.lr_decay_epoch, self.params.lr_decay_rate)    

        ## resume training
        if self.params.resume:
            chkp = self._load_chkp(self.params.resume)
            self.epoch_init = chkp['epoch'] + 1
            self.opt.load_state_dict(chkp['opt_state'])
            self.scheduler.load_state_dict(chkp['sch_state'])
            self.mdl.load_state_dict(chkp['mdl_state'])
            self.error_val_best = chkp['error_val_best']
            print(f'## resume training from {self.params.resume}: epoch={self.epoch_init} ')
        else:
            ## init the epoch_init
            self.epoch_init = 1
        
            ## measure the initial model validation loss
            if ld_val:
                self.error_val_best, *_ = self.validate(ld_val)
            else:
                self.error_val_best = np.inf

            self._save_model(best=True)
        
    
    def _train_end(self, ld_val, ld_test):

        ## save the final model
        fn = self._save_model(best=False)
        print('## save the final model to %s'%(fn))
        
        ## load the model
        if not self.params.load_final:
            fn = self._load_model(best=True)
            print("## load the best model from %s"%(fn))

        ## print training time
        if hasattr(self, 'time_train_begin'):
            print("## training time: %f sec."%(time.time() - self.time_train_begin))
        
    
    def _train_epoch_begin(self, i_epoch):
        self.time_epoch_begin = time.time()
        

    def _train_epoch_batch_begin(self, i_epoch):
        pass

    
    def _train_epoch_batch_end(self, i_epoch):
        pass

    
    def _train_epoch(self, i_epoch, ld_tr):
        for x, y in ld_tr:
            self._train_epoch_batch_begin(i_epoch)
            self.opt.zero_grad()            
            self.loss_dict = self.loss_fn_train(x, y, lambda x: self.mdl(x, training=True), reduction='mean', device=self.params.device)
            self.loss_dict['loss'].backward()
            self.opt.step()
            self._train_epoch_batch_end(i_epoch)
        self.scheduler.step()


    def _train_epoch_end(self, i_epoch, ld_val, ld_test):

        msg = ''
        
        ## print loss
        for k, v in self.loss_dict.items():
            msg += '%s = %.4f, '%(k, v)

        ## test error
        if ld_test:
            error_te, *_ = self.test(ld_test)
            msg += 'error_test = %.4f, '%(error_te)
            
        ## validate the current model and save if it is the best so far
        if ld_val and (i_epoch % self.params.val_period==0):
            error_val, *_ = self.validate(ld_val)
            msg += 'error_val = %.4f (error_val_best = %.4f)'%(error_val, self.error_val_best)
            if self.error_val_best >= error_val:
                msg += ', saved'
                self._save_model(best=True)
                self.error_val_best = error_val
        elif ld_val is None:
            self._save_model(best=False)
            msg += 'saved'

        ## print the current status
        msg = '[%d/%d epoch, lr=%.2e, %.2f sec.] '%(
            i_epoch, self.params.n_epochs, 
            self.opt.param_groups[0]['lr'],
            time.time()-self.time_epoch_begin,
        ) + msg
        
        print(msg)

        ## save checkpoint
        self._save_chkp()
                
    
