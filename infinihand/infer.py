"""Inference with an optional Camera checkpoint and Appendix-C binary-pool sparse BA."""
import argparse,json,os,sys,subprocess
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--checkpoint',type=Path,required=True,help='Complete weights exported from the current training runs');p.add_argument('--config',type=Path,default=ROOT/'configs/infinihand_v1.json')
 p.add_argument('--intrinsics',type=Path,required=True,help='JSON: fx,fy,cx,cy,image_width,image_height, in original video pixels')
 p.add_argument('--mask-source',choices=['sam3','head'],default='head');p.add_argument('--mask-threshold',type=float,default=.9)
 p.add_argument('--sam3-python',default=sys.executable);p.add_argument('--sam3-source',type=Path,default=Path('/path/to/infinihand/third_party/sam3'));p.add_argument('--sam3-checkpoint',type=Path,default=Path('/path/to/weights/sam3.pt'));p.add_argument('--detector',type=Path,default=Path('/path/to/weights/detector.pt'))
 p.add_argument('--ba-python',default=sys.executable);p.add_argument('--droid-weights',type=Path,default=Path('/path/to/weights/droid.pth'));p.add_argument('--no-ba',action='store_true');p.add_argument('--ba-stride',type=int,default=20)
 p.add_argument('--camera-checkpoint',type=Path,help='Trusted camera-head-only training checkpoint; leaves all hand/mask weights untouched')
 p.add_argument('--ba-mode',choices=['binary','fixed'],default='binary');p.add_argument('--ba-pool-capacity',type=int,default=160);p.add_argument('--ba-no-flush',action='store_true')
 p.add_argument('--max-frames',type=int,default=8192);return p

def main():
 a=parser().parse_args()
 if not 0 < a.mask_threshold < 1:raise ValueError('mask threshold must be in (0,1)')
 if not 4 <= a.max_frames <= 8192:raise ValueError('max-frames must be between 4 and 8192')
 required=[a.video,a.intrinsics,a.checkpoint,a.config]
 if a.camera_checkpoint:required.append(a.camera_checkpoint)
 if a.ba_stride<1 or a.ba_pool_capacity<a.ba_stride:raise ValueError('Require ba-pool-capacity >= ba-stride >= 1')
 if a.mask_source=='sam3':required += [a.sam3_checkpoint,a.detector,a.sam3_source]
 if not a.no_ba:required += [a.droid_weights]
 for path in required:
  if not path.exists():raise FileNotFoundError(path)
 a.output.mkdir(parents=True,exist_ok=True);D=a.output/'frames';D.mkdir(exist_ok=True)
 if list(D.glob('*.png')):raise ValueError('Use a fresh output directory to avoid stale frame/mask mixing')
 Kd=json.loads(a.intrinsics.read_text());cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS)
 if not cap.isOpened() or fps<=0:raise ValueError('Cannot open video or invalid FPS')
 n=0
 while n<a.max_frames:
  ok,im=cap.read()
  if not ok:break
  h,w=im.shape[:2]
  if (w,h)!=(Kd['image_width'],Kd['image_height']):raise ValueError('Video dimensions differ from intrinsics metadata; explicitly calibrate/rescale before inference')
  cv2.imwrite(str(D/f'{n:06d}.png'),im);n+=1
 cap.release()
 if n<4:raise ValueError('At least four frames required')
 if n>8192:raise ValueError('Maximum supported temporal index is 8192')
 boxes=np.zeros((n,2,4),np.float32);valid=np.zeros((n,2),bool)
 if a.mask_source=='sam3':
  import yaml
  cfg=yaml.safe_load((ROOT/'third_party/ego_step2/configs/default.yaml').read_text());cfg.update(output_fps=fps,write_visualization=False,wandb_mode="disabled")
  cf=a.output/'sam3_config.yaml';cf.write_text(yaml.safe_dump(cfg));env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'third_party/ego_step2/src')
  subprocess.run([a.sam3_python,'-m','hand_detection_sam3_mask','frames','--frames-dir',str(D.resolve()),'--output-dir',str((a.output/'sam3').resolve()),'--config',str(cf.resolve()),'--detector-weight',str(a.detector.resolve()),'--sam3-checkpoint',str(a.sam3_checkpoint.resolve()),'--sam3-src',str(a.sam3_source.resolve())],env=env,check=True)
  records=[json.loads(line) for line in (a.output/'sam3/masks.jsonl').read_text().splitlines()];assert len(records)==n*2
  seen=set()
  for row in records:
   i=int(row['frame_idx']);s={'right':0,'left':1}[row['side_name']];assert 0<=i<n and (i,s) not in seen;seen.add((i,s))
   if row['mask_box_xyxy'] is not None and row['mask_area']>0:boxes[i,s]=np.array(row['mask_box_xyxy'])/[w,h,w,h];valid[i,s]=True
 import torch
 from .models.stage2_streaming_inference_v6 import build_model,run_dual_stream,validate_contract,camera_to_predicted_global
 model,load_seconds=build_model(a.checkpoint,torch.device('cuda:0'),a.config)
 camera_receipt=None
 if a.camera_checkpoint:
  from .inference_checkpoint import load_camera_checkpoint
  camera_receipt=load_camera_checkpoint(model,a.camera_checkpoint)
 images=np.stack([cv2.resize(cv2.imread(str(D/f'{i:06d}.png')),(518,378))[:,:,::-1] for i in range(n)])
 images=torch.from_numpy(np.ascontiguousarray(images)).permute(0,3,1,2).float().div_(255).unsqueeze(0).cuda()
 K=np.array([[Kd['fx'],0,Kd['cx']],[0,Kd['fy'],Kd['cy']],[0,0,1]],np.float32);K[0]*=518/w;K[1]*=378/h;intr=torch.from_numpy(K).view(1,1,3,3).expand(1,n,3,3).cuda()
 opts=dict(bbox_source='gt' if a.mask_source=='sam3' else 'mask',mask_threshold=a.mask_threshold)
 if a.mask_source=='sam3':opts.update(gt_hand_boxes=torch.from_numpy(boxes)[None].cuda(),gt_hand_valid=torch.from_numpy(valid)[None].cuda())
 with torch.inference_mode():out,stats,timing=run_dual_stream(model,images,intr,**opts);world,_=camera_to_predicted_global(out)
 validate_contract(stats,n);arr={k:v.float().cpu().numpy() for k,v in out.items()};poses=world['pred_camera_to_anchor'].float().cpu().numpy()
 if a.mask_source=='head':boxes=arr['mask_tight_boxes'];valid=arr['presence_mask']>.5
 for key,value in arr.items():
  if not np.isfinite(value).all():raise ValueError('Non-finite model output: '+key)
 np.savez_compressed(a.output/'camera_space.npz',**arr,valid=valid)
 np.savez_compressed(a.output/'ba_inputs.npz',poses=poses,boxes=boxes)
 del out,world,model,images,intr;torch.cuda.empty_cache()
 if not a.no_ba:
  subprocess.run([a.ba_python,'-m','infinihand.sparse_ba_backend','--output',str(a.output.resolve()),'--intrinsics',str(a.intrinsics.resolve()),'--weights',str(a.droid_weights.resolve()),'--stride',str(a.ba_stride),'--mode',a.ba_mode,'--pool-capacity',str(a.ba_pool_capacity)]+(['--no-flush'] if a.ba_no_flush else []),cwd=ROOT,check=True)
  poses=np.load(a.output/'trajectory.npz')['poses']
 if not np.isfinite(poses).all():raise ValueError('Non-finite trajectory')
 rot=poses[:,:3,:3];trans=poses[:,:3,3]
 np.savez_compressed(a.output/'world_space.npz',camera_to_world=poses,world_joints=np.einsum('tij,thnj->thni',rot,arr['camera_joints'])+trans[:,None,None,:],world_vertices=np.einsum('tij,thnj->thni',rot,arr['camera_vertices'])+trans[:,None,None,:],world_global_orient=rot[:,None]@arr['camera_global_orient'],world_translation=np.einsum('tij,thj->thi',rot,arr['camera_translation'])+trans[:,None,:],hand_pose=arr['hand_pose'],betas=arr['betas'],valid=valid)
 receipt=dict(camera_checkpoint=camera_receipt,ba_mode=None if a.no_ba else a.ba_mode,ba_pool_capacity=a.ba_pool_capacity,status='complete',frames=n,fps=fps,hand_slots=['right','left'],mask_source=a.mask_source,mask_threshold=a.mask_threshold,ba_stride=None if a.no_ba else a.ba_stride,coordinate_frame='first predicted camera',checkpoint=str(a.checkpoint),model_load_seconds=load_seconds,timing=timing,cache=stats)
 (a.output/'SUCCESS.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
