"""Quality-aware short temporal convolutions and a long patch Transformer for wrist trajectories."""
from functools import lru_cache
import hashlib
import importlib.util
import zipfile
import zipimport

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paths_q1 import DATA,SNAP
from wrist_observations import LENGTH

VARIANTS = ("quality_stage","short_only","no_quality","single_gaussian")
RHO = .85


@lru_cache(maxsize=1)
def patch_layers():
    archive = next(SNAP.glob("03-LSTTN-68185aa8.zip"))
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith("/models/transformer.py"))
        if hashlib.sha256(z.read(name)).hexdigest()!="52746e2018fe126e6ea8e70956fda1699919b8f4d19593b316a24b022a9e54f9":
            raise ValueError("LSTTN patch source does not match the recorded author snapshot")
    loader = zipimport.zipimporter(str(archive)+"/"+name.rsplit("/",1)[0])
    module = importlib.util.module_from_spec(loader.find_spec("transformer"))
    loader.exec_module(module)
    return module.InputEmbedding,module.LearnableTemporalPositionalEncoding


def deterministic():
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)


class SequenceInputs:
    def __init__(self, data, device):
        self.device = device
        self.values = {k:torch.tensor(np.asarray(data[k]),device=device) for k in
                       ("x","nodes","node_ok","quality","time","vec","cad","start")}
        self.offsets = torch.arange(LENGTH-1,-1,-1,device=device)

    @classmethod
    def recorded(cls,device):
        return cls({k:np.load(DATA/(k+".npy"),mmap_mode="r") for k in
                   ("x","nodes","node_ok","quality","time","vec","cad","start")},device)

    def batch(self, rows):
        ids = torch.as_tensor(rows,device=self.device,dtype=torch.long)
        d = self.values
        idx = ids[:,None]-self.offsets
        mask = idx>=d["start"][ids,None]
        idx = torch.maximum(idx,d["start"][ids,None])
        age = d["time"][ids,None]-d["time"][idx]
        return d["x"][idx],d["nodes"][idx],d["node_ok"][idx],d["quality"][idx],mask,age,d["vec"][ids],d["cad"][ids]


def elapsed(cad):
    age = cad[:,None]*torch.arange(1,31,device=cad.device)
    basis = torch.stack((age.clip(0,.10),(age-.10).clip(0,.20),(age-.30).clip(0,.30),(age-.60).clip(min=0)),-1)
    return age,basis


class WristLSTTN(nn.Module):
    def __init__(self, xdim, vectors, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.long = variant!="short_only"
        self.quality_aware = variant!="no_quality"
        self.components = 1 if variant=="single_gaussian" else 3
        self.nv = vectors
        width = 48
        self.joints = nn.Linear(6,12)
        self.joint_ids = nn.Parameter(torch.randn(6,12)*.025)
        self.spatial = nn.Linear(12,1)
        if self.quality_aware:
            self.reliability = nn.Sequential(nn.Linear(8,12),nn.SiLU(),nn.Linear(12,1))
        dim = xdim+12+1+(8 if self.quality_aware else 0)
        self.tokens = nn.Sequential(nn.Linear(dim,width),nn.LayerNorm(width),nn.SiLU())
        self.short_blocks = nn.ModuleList(nn.Conv1d(width,2*width,3,dilation=d) for d in (1,3))
        if self.long:
            embedding,position = patch_layers()
            self.patch = embedding(6,width,width)
            self.position = position(width,.045,max_len=LENGTH//6)
            self.clock = nn.Linear(1,width)
            self.attention = nn.TransformerEncoderLayer(width,4,96,.055,activation="gelu",batch_first=True)
            self.trend = nn.Conv1d(width,width,3,dilation=2)
            self.merge = nn.Linear(2*width,width)
        self.decode = nn.Sequential(nn.Linear(2*width,88),nn.LayerNorm(88),nn.SiLU(),nn.Dropout(.075),
                                    nn.Linear(88,72),nn.SiLU())
        self.stage = nn.Linear(72,3)
        self.common = nn.Linear(72,4*(vectors+3))
        self.offsets = nn.Linear(72, self.components*4*(vectors+3)) if self.components>1 else None
        self.spread = nn.Linear(72,self.components*4)
        for layer in (self.common,self.offsets):
            if layer is not None:
                nn.init.normal_(layer.weight,std=.012)
                nn.init.zeros_(layer.bias)

    def forward(self,x,nodes,node_ok,quality,valid,age,vectors,cad):
        jp = torch.tanh(self.joints(nodes)+self.joint_ids)
        att = self.spatial(jp).squeeze(-1).masked_fill(~node_ok,-1e4)
        w = torch.softmax(att,-1)*node_ok
        spatial = (jp*w[...,None]).sum(-2)/w.sum(-1,keepdim=True).clamp_min(1e-5)
        geometry = torch.cat((x,spatial),-1)
        if self.quality_aware:
            gate = (.35+.65*torch.sigmoid(self.reliability(quality)))*(.22+.78*quality[:,:,:1])*quality[:,:,3:4]
            inputs = [geometry*gate,quality]
        else:
            inputs = [geometry]
        tokens = self.tokens(torch.cat(inputs+[torch.log1p(age)[...,None]],-1))*valid[...,None]
        short = tokens[:,-15:].transpose(1,2)
        for conv in self.short_blocks:
            delta = F.glu(conv(F.pad(short,(2*conv.dilation[0],0))),dim=1)
            short = (short+delta)*valid[:,-15:,None].transpose(1,2)
        local = short[:,:,-1]
        if self.long:
            patches = self.patch(tokens.transpose(1,2)[:,None])[:,0].transpose(1,2)
            present = valid.reshape(len(x),-1,6).any(-1)
            patch_age = age.reshape(len(x),-1,6).mean(-1)
            patches = self.position(patches+self.clock(torch.log1p(patch_age)[...,None]),None)
            states = self.attention(patches,src_key_padding_mask=~present)*present[...,None]
            trend = F.gelu(self.trend(F.pad(states.transpose(1,2),(4,0)))).transpose(1,2)
            long = states[:,-1]+((states+trend)*present[...,None]).sum(1)/present.sum(1,keepdim=True)
            weight = torch.sigmoid(self.merge(torch.cat((long,local),-1)))
            fused = weight*long+(1-weight)*local
        else:
            fused = local
        h = self.decode(torch.cat((fused,tokens[:,-1]),-1))
        logits = self.stage(h)
        coef = self.common(h).view(len(x),1,4,self.nv+3)
        if self.offsets is not None:
            coef = coef+.65*self.offsets(h).view(len(x),self.components,4,self.nv+3)
        rate = torch.einsum("bkcv,bvd->bkcd",coef[:,:,:,:self.nv],vectors)+coef[:,:,:,self.nv:]
        _,basis = elapsed(cad)
        means = torch.einsum("bhc,bkcd->bkhd",basis,rate)
        rates = .004+.23*F.softplus(self.spread(h).view(len(x),self.components,4))
        sigma = .0018+torch.einsum("bhc,bkc->bkh",basis,rates)
        return logits,means,sigma


def joint_log_density(y,means,scale):
    normalized = (y[:,None]-means)/scale[:,:,:,None]
    innov = (normalized[:,:,1:]-RHO*normalized[:,:,:-1])/np.sqrt(1-RHO**2)
    square = normalized[:,:,0].square().sum(-1)+innov.square().sum((-1,-2))
    h = y.shape[1]
    return -.5*square-3*scale.log().sum(-1)-1.5*h*np.log(2*np.pi)-1.5*(h-1)*np.log(1-RHO**2)


def loss_parts(out,y,target,cad):
    logits,means,sigma = out
    k = means.shape[1]
    weights = logits.softmax(1) if k>1 else torch.ones_like(logits[:,:1])
    pred = (weights[:,:,None,None]*means).sum(1)
    age,_ = elapsed(cad)
    units = .018+.16*age
    geom = (pred-y).norm(dim=-1)/units
    true_mode = means[torch.arange(len(y),device=y.device),target if k>1 else torch.zeros_like(target)]
    typed = (true_mode-y).norm(dim=-1)/units
    nll = -torch.logsumexp(weights.clamp_min(1e-8).log()+joint_log_density(y,means,sigma),dim=1)/30
    point = geom.mean(1)+.22*geom[:,-1]+.20*typed.mean(1)
    return (point*(1+.28*(target==1))).mean()+.055*nll.mean()+.22*F.cross_entropy(logits,target)
