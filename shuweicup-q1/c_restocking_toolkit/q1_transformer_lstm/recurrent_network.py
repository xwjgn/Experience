"""Anatomical LSTM streams, wrist-query attention, and a shared probabilistic motion decoder."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODELS = ("M1","M2","M3","M2_uniform")
KNOTS = np.array([1,4,9,16,23,30])
SHORT = np.arange(17,-1,-1)
LONG = np.arange(76,-1,-4)
BATCH = 128


def deterministic():
    torch.set_num_threads(3)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)


class Histories:
    def __init__(self,data,device):
        self.device = device
        self.d = {k:torch.tensor(np.asarray(data[k]),device=device) for k in
                  ("nodes","global_x","node_mask","qinfo","basis","time","cad","start")}
        self.short = torch.tensor(SHORT,device=device)
        self.long = torch.tensor(LONG,device=device)

    def indices(self,ids,offsets):
        source = ids[:,None]-offsets
        n = (source>=self.d["start"][ids,None]).sum(1)
        j = torch.arange(len(offsets),device=self.device)[None]
        idx = source.gather(1,(len(offsets)-n[:,None]+j).clamp(max=len(offsets)-1))
        return idx,j<n[:,None],n-1

    def get(self,rows):
        ids = torch.as_tensor(rows,device=self.device,dtype=torch.long)
        d = self.d
        si,mask,last = self.indices(ids,self.short)
        li,lmask,_ = self.indices(ids,self.long)
        age = torch.log1p((d["time"][ids,None]-d["time"][si]).clamp_min(0))[...,None]
        lag = torch.log1p((d["time"][ids,None]-d["time"][li]).clamp_min(0))[...,None]
        return (d["nodes"][si],d["node_mask"][si],d["global_x"][si],d["qinfo"][si],age,mask,last,
                d["global_x"][li],d["qinfo"][li],lag,lmask,d["basis"][ids],d["cad"][ids])


def last_token(sequence,index):
    return sequence[torch.arange(len(sequence),device=sequence.device),index]


class HandLSTM(nn.Module):
    def __init__(self,dims,kind,flag_rate):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(kind)
        node,common,directions = dims
        self.kind,self.direction_count = kind,directions
        self.flag_offset = float(np.log(max(flag_rate,1.e-4)/max(1-flag_rate,1.e-4)))
        self.global_geom = nn.Sequential(nn.Linear(common+1,32),nn.LayerNorm(32),nn.SiLU())
        if kind=="M1":
            self.flat = nn.Sequential(nn.Linear(6*node+32,44),nn.LayerNorm(44),nn.SiLU())
            self.geometry_lstm = nn.LSTM(44,48,batch_first=True)
            geom_dim = 48
        else:
            self.joint_type = nn.Parameter(torch.randn(6,5)*.03)
            self.joint_input = nn.Sequential(nn.Linear(node+6,24),nn.LayerNorm(24),nn.SiLU())
            self.joint_lstm = nn.LSTM(24,24,batch_first=True)
            if kind!="M2_uniform":
                self.query = nn.Linear(24,24)
                self.key = nn.Linear(24,24)
            self.value = nn.Linear(24,24)
            geom_dim = 80
        self.flag_head = nn.Sequential(nn.Linear(geom_dim,28),nn.SiLU(),nn.Linear(28,1))
        self.temporal = nn.LSTM(geom_dim+8,56,batch_first=True)
        if kind=="M3":
            self.long_proj = nn.Sequential(nn.Linear(common+9,48),nn.LayerNorm(48),nn.SiLU())
            self.long_conv = nn.Conv1d(48,48,3,dilation=2)
            self.long_attention = nn.MultiheadAttention(48,4,.045,batch_first=True)
            self.long_to_short = nn.Linear(48,56)
            self.mix = nn.Linear(104,56)
        self.context = nn.Sequential(nn.Linear(56*2+32+8,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.07),nn.Linear(80,64),nn.SiLU())
        self.stage_head = nn.Linear(64,3)
        self.shared_rates = nn.Linear(64,len(KNOTS)*(directions+3))
        self.mode_rates = nn.Linear(64,3*len(KNOTS)*(directions+3))
        self.width_head = nn.Linear(64,3*len(KNOTS))
        for m in (self.shared_rates,self.mode_rates):
            nn.init.normal_(m.weight,std=.013)
            nn.init.zeros_(m.bias)
        interp = np.stack([np.interp(np.arange(1,31),KNOTS,v) for v in np.eye(len(KNOTS))],axis=1)
        self.register_buffer("interp",torch.tensor(interp,dtype=torch.float32))

    def encode_quality(self,nodes,nmask,global_x,age):
        b,t,n,_ = nodes.shape
        g = self.global_geom(torch.cat((global_x,age),-1))
        if self.kind=="M1":
            geom,_ = self.geometry_lstm(self.flat(torch.cat(((nodes*nmask[...,None]).flatten(2),g),-1)))
        else:
            ids = self.joint_type[None,None].expand(b,t,-1,-1)
            emb = self.joint_input(torch.cat((nodes,ids,age[:,:,None].expand(-1,-1,n,-1)),-1))*nmask[...,None]
            history,_ = self.joint_lstm(emb.transpose(1,2).reshape(b*n,t,24))
            history = history.reshape(b,n,t,24).transpose(1,2)
            wrist = history[:,:,0]
            if self.kind=="M2_uniform":
                att = nmask[...,None].expand(-1,-1,-1,3).to(nodes.dtype)
            else:
                q = self.query(wrist).view(b,t,1,3,8)
                k = self.key(history).view(b,t,n,3,8)
                score = (q*k).sum(-1)/np.sqrt(8.)
                att = score.masked_fill(~nmask[...,None],-1e4).softmax(2)*nmask[...,None]
            att = att/att.sum(2,keepdim=True).clamp_min(1.e-6)
            ctx = (att[...,None]*self.value(history).view(b,t,n,3,8)).sum(2).reshape(b,t,24)
            geom = torch.cat((wrist,ctx,g),-1)
        return geom,self.flag_head(geom).squeeze(-1),g

    def forward(self,nodes,nmask,global_x,qinfo,age,valid,last,lx,lq,lage,lvalid,directions,cad):
        geom,flag_logits,g = self.encode_quality(nodes,nmask,global_x,age)
        if self.kind!="M1":
            risk = (flag_logits.detach()+self.flag_offset).sigmoid()
            weight = (.30+.70*qinfo[:,:,3])*(1-.55*risk)
            geom = geom*weight[:,:,None]
        seq,_ = self.temporal(torch.cat((geom,qinfo),-1))
        latest = last_token(seq,last)
        if self.kind=="M3":
            long = self.long_proj(torch.cat((lx,lq,lage),-1))*lvalid[...,None]
            effect = self.long_conv(F.pad(long.transpose(1,2),(4,0))).transpose(1,2)
            long = (long+F.silu(effect))*lvalid[...,None]
            query = last_token(long,lvalid.sum(1)-1)[:,None]
            long = self.long_attention(query,long,long,key_padding_mask=~lvalid,need_weights=False)[0][:,0]
            gate = self.mix(torch.cat((latest,long),-1)).sigmoid()
            latest = latest+gate*self.long_to_short(long)
        pooled = (seq*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
        hidden = self.context(torch.cat((latest,pooled,last_token(g,last),last_token(qinfo,last)),-1))
        logits = self.stage_head(hidden)
        b = len(cad)
        dim = self.direction_count+3
        coef = self.shared_rates(hidden).view(b,1,-1,dim)+.55*self.mode_rates(hidden).view(b,3,-1,dim)
        rate = torch.einsum("bkfd,bda->bkfa",coef[...,:-3],directions)+coef[...,-3:]
        step_rate = torch.einsum("hf,bkfa->bkha",self.interp,rate)
        mu = cad[:,None,None,None]*torch.cumsum(step_rate,dim=2)
        age_h = cad[:,None]*torch.arange(1,31,device=cad.device)
        logscale = torch.einsum("hf,bkf->bkh",self.interp,self.width_head(hidden).reshape(b,3,-1))
        sd = .0015+(.005+.21*age_h[:,None])*F.softplus(logscale)
        return logits,mu,sd,last_token(flag_logits,last)


def objective(out,y,stages,cad,quality_target):
    n = len(y)
    logits,mu,sd,qlogit = out
    s,mu,sd = logits[:n],mu[:n],sd[:n]
    p = s.softmax(-1)
    pred = (p[:,:,None,None]*mu).sum(1)
    age = cad[:n,None]*torch.arange(1,31,device=y.device)
    unit = .020+.16*age
    dist = (y-pred).norm(dim=-1)/unit
    k = torch.arange(n,device=y.device)
    oracle = (y-mu[k,stages]).norm(dim=-1)/unit
    point = (dist.mean(1)+.22*dist[:,-1]+.16*oracle.mean(1))*(1+.30*(stages==1))
    delta2 = (y[:,None]-mu).square().sum(-1)
    logpdf = -1.5*np.log(2*np.pi)-3*sd.log()-.5*delta2/sd.square()
    marginal = -torch.logsumexp(F.log_softmax(s,1)[:,:,None]+logpdf,dim=1).mean()
    # Quality examples are sampled separately so losses include occluded hand-frames.
    qloss = F.binary_cross_entropy_with_logits(qlogit[n:],quality_target)
    return point.mean()+.095*marginal+.20*F.cross_entropy(s,stages)+.23*qloss
