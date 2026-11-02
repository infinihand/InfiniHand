"""DROID backend only, initialized by predicted c2w poses; no SLAM frontend."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .sparse_ba import SparseBinaryBA, apply_pose_corrections, validate_poses


class DroidSparseOptimizer:
    def __init__(self, frames, intrinsics, boxes, weights, max_keyframes,
                 image_size=(384, 536), iterations=(7, 12)):
        # Lazy imports keep CPU pool/unit tests independent of CUDA extensions.
        import torch
        import cv2
        from scipy.spatial.transform import Rotation
        droid = Path(__file__).resolve().parents[1] / 'third_party/droid/droid_ba'
        sys.path.insert(0, str(droid))
        from backend_runtime import DroidBackendRuntime
        import lietorch
        from .droid_corr_compat import install_corr_compat
        self.correlation_abi = install_corr_compat()
        self.torch, self.cv2, self.Rotation, self.lietorch = torch, cv2, Rotation, lietorch
        self.frames, self.calib, self.boxes = Path(frames), intrinsics, boxes
        self.h, self.w = image_size
        if self.h % 8 or self.w % 8:
            raise ValueError('DROID image dimensions must be multiples of 8')
        if not iterations or any(i < 1 for i in iterations):
            raise ValueError('Backend iterations must be positive')
        self.iterations = iterations
        args = SimpleNamespace(weights=str(weights), buffer=max_keyframes+1,
            image_size=[self.h,self.w], stereo=False, beta=.3,
            backend_thresh=22.,backend_radius=2,backend_nms=3,upsample=False)
        self.system = DroidBackendRuntime(args)
        self.args = vars(args)

    def __call__(self, ids, seeds):
        torch, cv2 = self.torch, self.cv2
        v, f = self.system.video, self.system.encoder
        if len(ids) >= len(v.poses):
            raise ValueError('Keyframe selection exceeds backend buffer')
        with torch.inference_mode():
            start = time.perf_counter()
            v.counter.value = 0
            # No stale depth, recurrent state, sensor priors, or masks across pools.
            v.disps.fill_(1);v.disps_sens.zero_();v.disps_up.zero_();v.dirty.zero_()
            self.system.backend.errors.clear()
            w2c = np.linalg.inv(seeds)
            seed = torch.tensor(np.c_[w2c[:,:3,3], self.Rotation.from_matrix(w2c[:,:3,:3]).as_quat()],device='cuda',dtype=torch.float32)
            for j, stamp in enumerate(ids):
                im = cv2.imread(str(self.frames/f'{stamp:06d}.png'))
                if im is None:
                    raise FileNotFoundError(self.frames/f'{stamp:06d}.png')
                H,W = im.shape[:2]
                if (W,H) != (self.calib['image_width'],self.calib['image_height']):
                    raise ValueError('Frame size differs from original-pixel intrinsics')
                small = cv2.resize(im,(self.w,self.h))
                valid = (cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)>3).astype(np.uint8)*255
                for box in self.boxes[stamp]:
                    if box[2]<=box[0] or box[3]<=box[1]:
                        continue
                    x0,y0,x1,y1=(box*np.array([self.w,self.h,self.w,self.h])).astype(int)
                    cv2.rectangle(valid,(max(0,x0-20),max(0,y0-20)),
                                  (min(self.w-1,x1+20),min(self.h-1,y1+20)),0,-1)
                frame=torch.from_numpy(small).permute(2,0,1)[None]
                mask=torch.from_numpy((valid[3::8,3::8]==0).astype(np.float32)).cuda()
                intr=torch.tensor([self.calib['fx']*self.w/W,self.calib['fy']*self.h/H,
                                   (self.calib['cx']+.5)*self.w/W-.5,(self.calib['cy']+.5)*self.h/H-.5],device='cuda')
                inputs=frame[None,:,[2,1,0]].cuda()/255.
                inputs=inputs.sub_(f.MEAN).div_(f.STDV)
                gmap=f.feature_encoder(inputs)
                net,inp=f.context_encoder(inputs[:,[0]])
                v.append(stamp,frame[0],seed[j],1.,None,intr/8.,gmap,net[0],inp[0],mask)
            torch.cuda.synchronize()
            feature_seconds=time.perf_counter()-start
            # DROID normalization changes monocular gauge scale. Undo that exact
            # numerical normalization, without claiming metric-scale recovery.
            factors=[];normalizer=v.normalize
            def normalize():
                factor=float(v.disps[:v.counter.value].mean())
                if not np.isfinite(factor) or factor<=0:
                    raise RuntimeError('Invalid DROID disparity normalization')
                normalizer();factors.append(factor)
            v.normalize=normalize
            start=time.perf_counter()
            try:
                for iters in self.iterations:
                    self.system.backend(iters)
                torch.cuda.synchronize()
                keyposes=self.lietorch.SE3(v.poses[:len(ids)]).inv().matrix().cpu().numpy().astype(np.float64)
                keyposes[:,:3,3]/=np.prod(factors)
            finally:
                v.normalize=normalizer
            result=validate_poses(keyposes)
            errors=list(self.system.backend.errors)
            if not np.isfinite(errors).all():
                raise RuntimeError('Nonfinite DROID reprojection error')
            return result,dict(correlation_abi=self.correlation_abi,feature_seconds=feature_seconds,backend_seconds=time.perf_counter()-start,
                normalization_factors=factors,backend_errors=errors,keyframes=len(ids),iterations=list(self.iterations))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--intrinsics',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--mode',choices=['binary','fixed'],default='binary')
    p.add_argument('--pool-capacity',type=int,default=160)
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--no-flush',action='store_true')
    p.add_argument('--height',type=int,default=384);p.add_argument('--width',type=int,default=536)
    a=p.parse_args()
    if a.stride<1 or a.pool_capacity<a.stride:
        p.error('Require pool-capacity >= stride >= 1')
    if (a.output/'BA_SUCCESS.json').exists():
        raise FileExistsError('Refusing to overwrite a completed BA result')
    with np.load(a.output/'ba_inputs.npz') as x:
        poses=validate_poses(x['poses']);boxes=x['boxes'].copy()
    if len(poses)<2 or boxes.shape!=(len(poses),2,4) or not np.isfinite(boxes).all():
        raise ValueError('Need >=2 frames and finite boxes [T,2,4]')
    calib=json.loads(a.intrinsics.read_text())
    ids=np.unique(np.r_[np.arange(0,len(poses),a.stride),len(poses)-1]).astype(int)
    maxkeys=len(ids) if a.mode=='fixed' else len(range(0,a.pool_capacity+2,a.stride))
    backend=DroidSparseOptimizer(a.output/'frames',calib,boxes,a.weights,maxkeys,(a.height,a.width))
    if a.mode=='binary':
        controller=SparseBinaryBA(backend,a.pool_capacity,a.stride)
        for pose in poses:
            controller.append(pose)
        refined=controller.finish(flush=not a.no_flush)
        events=controller.events
    else:
        keyposes,diagnostics=backend(ids.tolist(),poses[ids])
        keyposes=poses[ids[0]]@np.linalg.inv(keyposes[0])@keyposes
        refined=apply_pose_corrections(poses,ids,keyposes)
        events=[dict(reason='fixed',selected=ids.tolist(),backend=diagnostics)]
    np.savez_compressed(a.output/'trajectory.npz',poses=refined,initial_poses=poses,
                        corrections=refined@np.linalg.inv(poses))
    report=dict(mode=a.mode,pool_capacity=a.pool_capacity,stride=a.stride,frames=len(poses),
        paper='https://arxiv.org/abs/2609.35743',selection='chronological pool indices starting at oldest' if a.mode=='binary' else 'original frame index stride plus final frame',
        flush=not a.no_flush,events=events,ba_calls=len(events),
        world_gauge='preserve first selected input c2w at each BA call',metric_scale_recovered=False,
        interpolation='SLERP rotation and linear translation of left corrections; hold endpoint correction outside keyframe interval',
        scheduling='sequential backend replay after feed-forward inference, not asynchronous',
        weights=str(a.weights),weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        frontend=False,motion_filter=False,gt_used=False)
    (a.output/'BA_REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (a.output/'BA_SUCCESS.json').write_text(json.dumps(dict(status='complete',ba_calls=len(events),frames=len(poses)),indent=2))
    print(json.dumps(dict(ba_calls=len(events),frames=len(poses),mode=a.mode)))

if __name__=='__main__':
    main()
