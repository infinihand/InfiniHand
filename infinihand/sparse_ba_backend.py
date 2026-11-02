"""DROID backend only, initialized by predicted c2w poses; no SLAM frontend."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .sparse_ba import apply_pose_corrections, validate_poses, reject_bad_updates


def triangulate_depth(frames, boxes, ids, seeds, calib, height=384, width=536):
 """Sparse RGB/LK triangulation in the predicted camera scale; no GT inputs."""
 import cv2
 H,W=height,width;K=np.array([[calib['fx']*W/calib['image_width'],0,(calib['cx']+.5)*W/calib['image_width']-.5],[0,calib['fy']*H/calib['image_height'],(calib['cy']+.5)*H/calib['image_height']-.5],[0,0,1]],float)
 images=[cv2.cvtColor(cv2.resize(cv2.imread(str(Path(frames)/f'{i:06d}.png')),(W,H)),cv2.COLOR_BGR2GRAY) for i in ids]
 masks=[]
 for idx,im in zip(ids,images):
  mask=(im>3).astype(np.uint8)*255
  for box in boxes[idx]:
   if box[2]>box[0] and box[3]>box[1]:
    x0,y0,x1,y1=(box*np.array([W,H,W,H])).astype(int);cv2.rectangle(mask,(max(0,x0-20),max(0,y0-20)),(min(W-1,x1+20),min(H-1,y1+20)),0,-1)
  masks.append(mask)
 depths=np.zeros((len(ids),H//8,W//8),np.float32);counts=[]
 for i in range(len(ids)):
  pts=cv2.goodFeaturesToTrack(images[i],1500,.01,5,mask=masks[i]);acc={}
  if pts is not None:
   for j in [i-1,i+1]:
    if not 0<=j<len(ids):continue
    q,ok,_=cv2.calcOpticalFlowPyrLK(images[i],images[j],pts,None,winSize=(31,31),maxLevel=4)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(images[j],images[i],q,None,winSize=(31,31),maxLevel=4)
    p=pts[:,0];q=q[:,0];keep=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(back[:,0]-p,axis=-1)<.5)
    qi=np.round(q).astype(int);inside=(qi[:,0]>=0)&(qi[:,0]<W)&(qi[:,1]>=0)&(qi[:,1]<H);keep &= inside
    keep[inside] &= masks[j][qi[inside,1],qi[inside,0]]>0
    p,q=p[keep],q[keep]
    if len(p)<8:continue
    T=np.linalg.inv(seeds[j])@seeds[i];X=cv2.triangulatePoints(K@np.eye(3,4),K@T[:3],p.T,q.T);X=(X[:3]/X[3]).T;Y=X@T[:3,:3].T+T[:3,3]
    pp=X@K.T;qq=Y@K.T;err=np.linalg.norm(pp[:,:2]/pp[:,2:]-p,axis=-1)+np.linalg.norm(qq[:,:2]/qq[:,2:]-q,axis=-1)
    rays1=np.c_[p,np.ones(len(p))]@np.linalg.inv(K).T;rays2=np.c_[q,np.ones(len(q))]@np.linalg.inv(K).T@T[:3,:3]
    cos=(rays1*rays2).sum(-1)/(np.linalg.norm(rays1,axis=-1)*np.linalg.norm(rays2,axis=-1));ang=np.degrees(np.arccos(np.clip(cos,-1,1)))
    good=(X[:,2]>.1)&(X[:,2]<5)&(Y[:,2]>.1)&(err<1)&(ang>.5)
    for xy,z in zip(p[good],X[good,2]):
     cell=(min(H//8-1,max(0,int((xy[1]-3)/8+.5))),min(W//8-1,max(0,int((xy[0]-3)/8+.5))))
     acc.setdefault(cell,[]).append(float(z))
  for (y,x),zs in acc.items():depths[i,y,x]=np.median(zs)
  counts.append(int((depths[i]>0).sum()))
 return depths,counts


class DroidSparseOptimizer:
    def __init__(self, frames, intrinsics, boxes, weights, max_keyframes,
                 image_size=(384, 536), iterations=(7, 12), depth_init='none', reject_outliers=True):
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
        if depth_init not in {'none', 'sparse'}:
            raise ValueError('depth_init must be none or sparse')
        self.depth_init, self.reject_outliers = depth_init, reject_outliers
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
        depth, depth_counts = None, []
        if self.depth_init == 'sparse':
            depth, depth_counts = triangulate_depth(self.frames, self.boxes, ids, seeds, self.calib, self.h, self.w)
        with torch.inference_mode():
            start = time.perf_counter()
            v.counter.value = 0
            # Reset depth, sensor priors, and masks before optimization.
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
                if depth is not None:
                    dp = torch.from_numpy(depth[j]).to(device='cuda')
                    valid_depth = dp > 0
                    v.disps[j][valid_depth] = 1 / dp[valid_depth]
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
            weight_history = []
            original_ba = v.ba
            def record_ba(*args, **kwargs):
                original_ba(*args, **kwargs)
                if len(weight_history) < 3 and self.reject_outliers:
                    weights, sources = args[1], args[3]
                    weight_history.append([float(weights[sources == j].mean())
                        if (sources == j).any() else 0. for j in range(len(ids))])
            v.ba = record_ba
            start=time.perf_counter()
            try:
                for iters in self.iterations:
                    self.system.backend(iters)
                torch.cuda.synchronize()
                keyposes=self.lietorch.SE3(v.poses[:len(ids)]).inv().matrix().cpu().numpy().astype(np.float64)
                keyposes[:,:3,3]/=np.prod(factors)
            finally:
                v.normalize=normalizer
                v.ba=original_ba
            result=validate_poses(keyposes)
            rejection = dict(enabled=False, rejected_frame_ids=[])
            if self.reject_outliers:
                result, rejection = reject_bad_updates(seeds, result, weight_history)
                rejection.update(enabled=True, rejected_frame_ids=[int(ids[j]) for j in rejection['rejected_indices']])
            errors=list(self.system.backend.errors)
            if not np.isfinite(errors).all():
                raise RuntimeError('Nonfinite DROID reprojection error')
            return result,dict(correlation_abi=self.correlation_abi,feature_seconds=feature_seconds,backend_seconds=time.perf_counter()-start,
                depth_init=self.depth_init,sparse_depth_cells=depth_counts,rejection=rejection,
                normalization_factors=factors,backend_errors=errors,keyframes=len(ids),iterations=list(self.iterations))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--intrinsics',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--height',type=int,default=384);p.add_argument('--width',type=int,default=536)
    p.add_argument('--depth-init', choices=['none', 'sparse'], default='none')
    p.add_argument('--no-reject-outliers', action='store_true', help='Disable BA update rejection for ablations')
    a=p.parse_args()
    if a.stride<1:
        p.error('Require stride >= 1')
    if (a.output/'BA_SUCCESS.json').exists():
        raise FileExistsError('Refusing to overwrite a completed BA result')
    with np.load(a.output/'ba_inputs.npz') as x:
        poses=validate_poses(x['poses']);boxes=x['boxes'].copy()
    if len(poses)<2 or boxes.shape!=(len(poses),2,4) or not np.isfinite(boxes).all():
        raise ValueError('Need >=2 frames and finite boxes [T,2,4]')
    calib=json.loads(a.intrinsics.read_text())
    ids=np.unique(np.r_[np.arange(0,len(poses),a.stride),len(poses)-1]).astype(int)
    backend=DroidSparseOptimizer(a.output/'frames',calib,boxes,a.weights,len(ids),(a.height,a.width),depth_init=a.depth_init,reject_outliers=not a.no_reject_outliers)
    keyposes,diagnostics=backend(ids.tolist(),poses[ids])
    keyposes=poses[ids[0]]@np.linalg.inv(keyposes[0])@keyposes
    refined=apply_pose_corrections(poses,ids,keyposes)
    events=[dict(reason='fixed',selected=ids.tolist(),backend=diagnostics)]
    np.savez_compressed(a.output/'trajectory.npz',poses=refined,initial_poses=poses,
                        corrections=refined@np.linalg.inv(poses))
    report=dict(depth_init=a.depth_init,reject_outliers=not a.no_reject_outliers,mode='fixed',stride=a.stride,frames=len(poses),
        selection='original frame index stride plus final frame',
        events=events,ba_calls=len(events),
        world_gauge='preserve first selected input c2w at each BA call',metric_scale_recovered=False,
        interpolation='SLERP rotation and linear translation of left corrections; hold endpoint correction outside keyframe interval',
        scheduling='sequential backend replay after feed-forward inference, not asynchronous',
        weights=str(a.weights),weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        frontend=False,motion_filter=False,gt_used=False)
    (a.output/'BA_REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (a.output/'BA_SUCCESS.json').write_text(json.dumps(dict(status='complete',ba_calls=len(events),frames=len(poses)),indent=2))
    print(json.dumps(dict(ba_calls=len(events),frames=len(poses),mode='fixed')))

if __name__=='__main__':
    main()
