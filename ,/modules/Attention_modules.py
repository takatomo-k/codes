import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

class Energy(nn.Module):
    def __init__(self, method, hidden_size):
        super(Energy, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method in {'general','monotonic'}:
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, enc_out):
        #import pdb; pdb.set_trace()
        if self.method == 'dot':
            energy = hidden.bmm(enc_out.transpose(1,2))

        elif self.method in {'general','monotonic'}:
            energy = self.attn(enc_out)
            energy = hidden.bmm(energy.transpose(1,2))
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, enc_out), 1))
            energy = self.other.bmm(energy.transpose(1,2))

        return energy

class BaseAttention(nn.Module) :
    NEG_NUM = -10000.0
    def __init__(self) :
        super(BaseAttention, self).__init__()
        self.state = None

    def apply_mask(self, score, mask, mask_value=NEG_NUM) :
        # TODO inplace masked_fill_
        return score.masked_fill(mask == 0, mask_value)

    def forward_single(self, input) :
        """
        key : batch x src_len x dim_k
        query : batch x dim_q
        """
        raise NotImplementedError()

    def forward_multiple(self, input) :
        """
        key : batch x src_len x dim_k
        query : batch x tgt_len x dim_q
        """
        raise NotImplementedError()

    def calc_expected_context(self, p_ctx, ctx) :
        """
        p_ctx = (batch x srcL)
        ctx = (batch x srcL x dim)
        """
        p_ctx_3d = p_ctx.unsqueeze(1) # (batch x 1 x enc_len)
        expected_ctx = torch.bmm(p_ctx_3d, ctx).squeeze(1) # (batch x dim)
        return expected_ctx

    def reset(self) :
        self.state = None
        pass

class MLPAttention(BaseAttention) :
    def __init__(self, memory_size, hidden_size, att_hid_size=256, act_fn=F.tanh) :
        super(MLPAttention, self).__init__()
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.att_hid_size = att_hid_size
        self.act_fn = act_fn
        self.lin_in2proj = nn.Linear(memory_size + hidden_size, att_hid_size)
        self.lin_proj2score = nn.Linear(att_hid_size, 1)
        self.out_features = self.memory_size
        pass

    def forward(self,  memory,hidden,mask=None) :
        batch, enc_len, enc_dim = memory.size()

        combined_input = torch.cat([memory, hidden.unsqueeze(1).expand(batch, enc_len, self.hidden_size)], 2) # batch x enc_len x (enc_dim + dec_dim) #
        combined_input_2d = combined_input.view(batch * enc_len, -1)
        score_memory = self.lin_proj2score(self.act_fn(self.lin_in2proj(combined_input_2d)))
        score_memory = score_memory.view(batch, enc_len) # batch x enc_len #
        if mask is not None :
            score_memory = self.apply_mask(score_memory, mask)
        p_memory = F.softmax(score_memory, dim=-1)
        expected_memory = self.calc_expected_context(p_memory, memory)
        return expected_memory, p_memory
        pass

    pass

class LocalGMMScorerAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, n_mixtures=1, monotonic=True,
            att_hid_size=256, act_fn=F.tanh,
            alpha_fn=torch.exp, beta_fn=torch.exp, kappa_fn=torch.exp,
            bprop_prev=False, alpha_bias=0, beta_bias=-1, kappa_bias=-1,
            normalize_alpha=False,
            scorer_cfg={'type':'mlp'},
            normalize_scorer=False,
            prune_range=[-3, 3],
            normalize_post=True, # if True, sum(posterior) == 1 #
            beta_val=None, # if provided, beta_val will be fixed instead of input dependent #
            # if prune = 3 = (2*std_dev) then beta_val = 0.222 = (1.0 / (2*(1.5**2))) formula :  beta_val = 1/(2*sqrt(std_dev^2)) #
            kappa_val=None, # 'auto' mode : determined by model, N (int) : increment position every decode step #
            ignore_likelihood=False,
            ) :
        """
        GMM attention (Alex Graves - synthesis network)
        alpha : mixture weight
        beta : inverse width of window
        kappa : centre of window
        monotonic : option for strictly moving from left to right
        stat_bias : initialize alpha, beta, kappa bias
        Args :
            att_hid_size : int (use None to avoid projection layer)

        """
        super(LocalGMMScorerAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.n_mixtures = n_mixtures
        self.monotonic = monotonic

        self.att_hid_size = att_hid_size
        self.act_fn = act_fn

        self.alpha_fn = alpha_fn if not isinstance(alpha_fn, (str)) else generator_act_fn(alpha_fn)
        self.beta_fn = beta_fn if not isinstance(beta_fn, (str)) else generator_act_fn(beta_fn)
        self.kappa_fn = kappa_fn if not isinstance(kappa_fn, (str)) else generator_act_fn(kappa_fn)

        self.bprop_prev = bprop_prev
        self.alpha_bias = alpha_bias
        self.beta_bias = beta_bias
        self.kappa_bias = kappa_bias
        self.normalize_alpha = normalize_alpha


        self.scorer_cfg = scorer_cfg
        self.normalize_scorer = normalize_scorer
        if isinstance(prune_range, int) :
            self.prune_range = [-prune_range, prune_range]
        else :
            self.prune_range = prune_range
        assert self.prune_range is None or len(self.prune_range) == 2
        if self.prune_range is not None :
            assert self.prune_range[0] < 0 and self.prune_range[1] > 0

        self.normalize_post = normalize_post
        self.beta_val = beta_val
        if beta_val is not None :
            assert beta_val > 0
        self.ignore_likelihood = ignore_likelihood
        self.kappa_val = kappa_val

        if self.att_hid_size is not None :
            self.lin_query2proj = nn.Linear(query_size, self.att_hid_size)
            self.lin_proj2stat = nn.Linear(self.att_hid_size, self.n_mixtures*3)
        else :
            self.lin_query2proj = lambda x : x
            self.act_fn = lambda x : x
            self.lin_proj2stat = nn.Linear(query_size, self.n_mixtures*3)
        init.constant_(self.lin_proj2stat.bias[0:self.n_mixtures], alpha_bias)
        init.constant_(self.lin_proj2stat.bias[self.n_mixtures:2*self.n_mixtures], beta_bias)
        init.constant_(self.lin_proj2stat.bias[2*self.n_mixtures:3*self.n_mixtures], kappa_bias)

        # scorer #
        if scorer_cfg['type'] == 'mlp' :
            self.scorer = MLPAttention(ctx_size, query_size, att_hid_size, act_fn)
            self.scorer_module = nn.ModuleList(
                    [nn.Linear(ctx_size+query_size, att_hid_size),
                    nn.Linear(att_hid_size, 1)])
            def _fn_scorer(_ctx, _query) :
                batch, enc_len, enc_dim = _ctx.size()
                combined_input = torch.cat([_ctx, _query.unsqueeze(1).expand(batch, enc_len, query_size)], 2)
                combined_input_2d = combined_input.view(batch * enc_len, -1)
                score_ctx = self.scorer_module[1](self.act_fn(self.scorer_module[0](combined_input_2d)))
                score_ctx = score_ctx.view(batch, enc_len)
                return score_ctx
            self.scorer_fn = _fn_scorer

        else :
            raise NotImplementedError()

        self.out_features = self.ctx_size
        pass

    def _init_stat(self, batch) :
        _zero_stat =torch.zeros((batch, self.n_mixtures),requires_grad=False).cuda()
        #_zero_stat = torchauto(self).FloatTensor(batch, self.n_mixtures).zero_()
        return {"ALPHA":_zero_stat.clone(),
                "BETA":_zero_stat.clone(),
                "KAPPA":_zero_stat.clone()}

    def forward(self, query,ctx) :

        #ctx = input['ctx'] # batch x enc_len x enc_dim #
        #query = input['query'] # batch x dec_dim #
        mask = None#input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()

        if self.state is None :
            self.state = {'stat':self._init_stat(batch)}
        stat_prev = self.state['stat'] # dict of 3 batch x n_mix #

        if not self.bprop_prev :
            for stat_name in ["ALPHA", "BETA", "KAPPA"] :
                stat_prev[stat_name] = stat_prev[stat_name].detach()
        _alpha_prev, _beta_prev, _kappa_prev = stat_prev["ALPHA"], stat_prev["BETA"], stat_prev["KAPPA"] # batch x n_mix #

        _alpha_beta_kappa = self.lin_proj2stat(self.act_fn(self.lin_query2proj(query)))
        _alpha, _beta, _kappa = _alpha_beta_kappa.split(self.n_mixtures, dim=1) # batch x n_mix #
        _alpha = self.alpha_fn(_alpha)

        if self.beta_val is None :
            _beta = self.beta_fn(_beta)
        else : # replace with fixed beta value #
            import pdb; pdb.set_trace()
            _beta = torchauto(self).FloatTensor([self.beta_val]).expand_as(_beta)

        if self.kappa_val is None :
            _kappa = self.kappa_fn(_kappa)
        else : # replace with fixed kappa value #
            import pdb; pdb.set_trace()

            _kappa = torchauto(self).FloatTensor([self.kappa_val]).expand_as(_kappa)

        if self.normalize_alpha :
            _alpha = _alpha / _alpha.sum(1).expand_as(_alpha)

        if self.monotonic :
            _kappa = _kappa + _kappa_prev
        #import pdb; pdb.set_trace()
        #pos_range = tensorauto(self, torch.arange(0, enc_len)). # enc_len #
        #        unsqueeze(0).expand(batch, enc_len). # batch x enc_len #
        #        unsqueeze(1).expand(batch, self.n_mixtures, enc_len)) # batch x n_mix x enc_len #
        pos_range=torch.arange(0, enc_len,requires_grad=False).unsqueeze(0).expand(batch, enc_len).unsqueeze(1).expand(batch, self.n_mixtures, enc_len).cuda()
        alpha = _alpha.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        beta = _beta.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        kappa = _kappa.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        # calculate prior prob #
        prior_score = alpha * torch.exp(-beta * (pos_range - kappa)**2)

        # generate pruning mask based on current kappa #
        if self.prune_range is not None :
            #import pdb; pdb.set_trace()

            #prune_mask = torchauto(self).FloatTensor(batch, self.n_mixtures, enc_len).zero_()
            prune_mask=torch.zeros((batch, self.n_mixtures, enc_len),requires_grad=False).cuda()
            for ii in range(batch) :
                for jj in range(self.n_mixtures) :
                    _center = _kappa.data[ii, jj]
                    _left = round(_center.item()) + self.prune_range[0]
                    _right = round(_center.item()) + self.prune_range[1] + 1
                    if _right <= 0 or _left >= enc_len :
                        continue
                    else :
                        _left = max(0, _left)
                        _right = min(enc_len, _right)
                        ##import pdb; pdb.set_trace()
                        prune_mask[ii, jj, _left:_right] = 1
            prune_mask = prune_mask.expand(batch, self.n_mixtures, enc_len)
            if mask is not None :
                prune_mask = prune_mask * mask.data.unsqueeze(1).expand(batch, self.n_mixtures, enc_len).float()
            prune_mask = prune_mask
        else :
            prune_mask = None
        # import ipdb; ipdb.set_trace()

        if prune_mask is not None :
            # mask prior attention with pruning mask #
            prior_score = prior_score * prune_mask

        prior_score = torch.sum(prior_score, dim=1).squeeze(1)

        if mask is not None :
            prior_score = self.apply_mask(prior_score, mask, 0)


        # calculate posterior prob #
        if self.ignore_likelihood :
            # use prior as posterior #
            posterior_score = prior_score
        else :
            # generate scorer #
            # calculate likelihood prob #
            # TODO : normalize or not ? give prune_mask or mask ? #
            assert self.n_mixtures == 1
            likelihood_score = self.scorer_fn(ctx, query)
            if prune_mask is not None :
                likelihood_score = self.apply_mask(likelihood_score, prune_mask[:, 0, :])
            if self.normalize_scorer :
                likelihood_score = F.softmax(likelihood_score, dim=-1)
            else :
                likelihood_score = torch.exp(likelihood_score)
            # combine likelihood x prior score #
            posterior_score = likelihood_score * prior_score

        if self.prune_range is not None :
            posterior_score = self.apply_mask(posterior_score, prune_mask[:, 0, :], 0)

        # normalize posterior #
        if self.normalize_post :
            EPS = 1e-7
            posterior_score = posterior_score / (posterior_score.sum(1, keepdim=True) + EPS)

        # TODO normalize or not ? - original no normalize #
        p_ctx = posterior_score
        expected_ctx = self.calc_expected_context(p_ctx, ctx)

        # save state
        self.state = {'stat':{"ALPHA":_alpha, "BETA":_beta, "KAPPA":_kappa}}
        return expected_ctx, p_ctx

    def __call__(self, *input, **kwargs) :
        result = super(LocalGMMScorerAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return p_ctx,expected_ctx
        #return  {
        #            "p_ctx":p_ctx,
        #            "expected_ctx":expected_ctx
        #        }

    pass

class LocationAttention(nn.Module):
    """
    Calculates context vector based on previous decoder hidden state (query vector),
    encoder output features, and convolutional features extracted from previous attention weights.
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf
    """


    def __init__(self, encoded_dim, query_dim, attention_dim, num_location_features=32):
        super(LocationAttention, self).__init__()
        self.f = nn.Conv1d(in_channels=1, out_channels=num_location_features,
                           kernel_size=31, padding=15, bias=False)
        self.U = nn.Linear(num_location_features, attention_dim)
        self.W = nn.Linear(query_dim, attention_dim)
        self.V = nn.Linear(encoded_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def score(self, query_vector, encoder_out, mask):
        encoder_out = self.V(encoder_out)  # (seq, batch, atten_dim) # project to attn dim
        query_vector = self.W(query_vector)  # (seq, batch, atten_dim)
        attention_energies = encoder_out + query_vector
        location_features = self.f(mask.permute(1, 0, 2))  # (batch, 1, seq1_len)
        attention_energies += self.U(location_features.permute(2, 0, 1))  # (seq, batch, numfeats)
        return self.w(self.tanh(attention_energies))

    def forward(self, query_vector, encoder_out, mask):
        energies = self.score(query_vector, encoder_out, mask)
        mask = F.softmax(energies, dim=0)
        context = encoder_out.permute(1, 2, 0) @ mask.permute(1, 0, 2)  # (batch, seq1, seq2)
        context = context.permute(2, 0, 1)  # (seq2, batch, encoder_dim)
        mask = mask.permute(2, 1, 0)  # (seq2, batch, seq1)
        return context, mask


class Attention(nn.Module):
    def __init__(self,method,hidden_size):
        """
        [Monotonic Attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """
        super().__init__()
        self.energy = Energy(method,hidden_size)
        self.method=method
        self.softmax=nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

    def reset(self):
        self.previous_alpha=None

    def forward(self,dec_h,enc_out):
        if self.method in {'dot','general','concat'}:
            energy=self.energy(dec_h,enc_out)
            return self.softmax(energy)
        elif self.method in {'monotonic'}:
            #import pdb; pdb.set_trace()
            return self.soft(dec_h,enc_out).unsqueeze(1)


    def gaussian_noise(self, *size):
        """Additive gaussian nosie to encourage discreteness"""
        return torch.cuda.FloatTensor(*size).normal_()

    def safe_cumprod(self, x):
        """Numerically stable cumulative product by cumulative sum in log-space"""
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1))

    def exclusive_cumprod(self, x):
        """Exclusive cumulative product [a, b, c] => [1, a, a * b]
        * TensorFlow: https://www.tensorflow.org/api_docs/python/tf/cumprod
        * PyTorch: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614
        """
        batch_size, sequence_length = x.size()
        one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def soft(self,decoder_h,enc_out):
        """
        Soft monotonic attention (Train)
        Args:
            enc_out [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        #import pdb; pdb.set_trace()
        batch_size, sequence_length, enc_dim = enc_out.size()

        monotonic_energy = self.energy( decoder_h,enc_out).squeeze(1)
        #import pdb; pdb.set_trace()
        p_select = self.sigmoid(monotonic_energy + self.gaussian_noise(monotonic_energy.size()))
        cumprod_1_minus_p = self.safe_cumprod(1 - p_select)

        #if self.previous_alpha is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
        alpha=p_select * cumprod_1_minus_p
        #else:
        #import pdb; pdb.set_trace()
            #alpha = p_select * cumprod_1_minus_p * torch.cumsum(self.previous_alpha / cumprod_1_minus_p, dim=1)
        self.previous_alpha=alpha
        #import pdb; pdb.set_trace()
        return self.previous_alpha


    def hard(self, enc_out, decoder_h, previous_attention=None):
        """
        Hard monotonic attention (Test)
        Args:
            enc_out [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = enc_out.size()

        if previous_attention is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            attention = torch.zeros(batch_size, sequence_length)
            attention[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available:
                attention = attention.cuda()
        else:
            # TODO: Linear Time Decoding
            # It's not clear if authors' TF implementation decodes in linear time.
            # https://github.com/craffel/mad/blob/master/example_decoder.py#L235
            # They calculate energies for whole encoder outputs
            # instead of scanning from previous attended encoder output.
            monotonic_energy = self.monotonic_energy(enc_out, decoder_h)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            above_threshold = (monotonic_energy > 0).float()

            p_select = above_threshold * torch.cumsum(previous_attention, dim=1)
            attention = p_select * self.exclusive_cumprod(1 - p_select)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            attended = attention.sum(dim=1)
            for batch_i in range(batch_size):
                if not attended[batch_i]:
                    attention[batch_i, -1] = 1

            # Ex)
            # p_select                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_select                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_select) = [1, 1, 1, 1, 0, 0, 0, 0]
            # attention: product of above     = [0, 0, 0, 1, 0, 0, 0, 0]
        return attention
