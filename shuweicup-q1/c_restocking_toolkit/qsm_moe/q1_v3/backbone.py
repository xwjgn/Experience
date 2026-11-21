"""Materialize freshly issued points and scales without reading future observations."""
import numpy as np

from config import BASE, WORK, read_json
from motion import Inputs
from distribution import gate_features, predict_block


def materialize():
    d=Inputs()
    lock=read_json(BASE/"frozen_before_test.json")
    motion=np.load(WORK/"motion_center.npy",mmap_mode="r")
    residual=np.load(WORK/"residual_knots.npy",mmap_mode="r")
    params=np.load(WORK/"width_parameters.npy")
    n=len(motion)
    center=np.empty_like(motion)
    widths=np.empty((n,30),np.float32)
    mix=np.empty(n,np.float32)
    for start in range(0,n,512):
        take=slice(start,start+512)
        x=gate_features(d.phase[take],d.q[take],d.state[take],d.phys[take])
        center[take],_,mix[take],widths[take]=predict_block(motion[take],residual[take],x,d.phase[take],
                              d.state[take],d.q[take],d.cad[take],lock,params)
    for name,value in (("reference_center",center),("reference_scale",widths),("diffusion_weight",mix)):
        np.save(WORK/(name+".npy"),value)
    print("Fresh reference forecasts issued",n,flush=True)


if __name__=="__main__":
    materialize()
