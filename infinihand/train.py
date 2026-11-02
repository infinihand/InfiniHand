"""Release training CLI (single GPU or torchrun)."""
import argparse,os,sys,tempfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
PATHS={'metadata_cache_dir','output_dir','ego3r_training_root','wilor_root','dataset_json','preload_manifest_root','finetune_checkpoint','resume_checkpoint','lingbotmap_root','lingbotmap_ckpt','wilor_cfg','wilor_ckpt'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE');p.add_argument('--check-config',action='store_true');a=p.parse_args()
 payload=yaml.safe_load(Path(a.config).read_text())
 for item in a.set:
  key,value=item.split('=',1);payload[key]=yaml.safe_load(value)
 for k in PATHS:
  v=payload.get(k)
  if isinstance(v,str) and v and ':' not in v:payload[k]=str((ROOT/Path(os.path.expandvars(v))).resolve())
 if payload.get('dataset_jsons'):payload['dataset_jsons']=[str((ROOT/Path(os.path.expandvars(v))).resolve()) for v in payload['dataset_jsons']]
 from .training import engine
 with tempfile.TemporaryDirectory(prefix='infinihand-config-') as d:
  config=Path(d)/'train.yaml';config.write_text(yaml.safe_dump(payload));cfg=engine.load_config(str(config))
  if a.check_config:
   from .training.data.split_policy import validate_training_paths
   validate_training_paths(cfg.dataset_jsons or [cfg.dataset_json], cfg.sampling_split_weights)
   from .training.initialization import validate_initialization
   validate_initialization(cfg,finetune_checkpoint=os.environ.get('FINETUNE_CHECKPOINT',cfg.finetune_checkpoint),resume_checkpoint=os.environ.get('RESUME_CHECKPOINT',cfg.resume_checkpoint))
   print('Configuration and initialization paths valid:',Path(a.config).name);return
  # Single-process training uses the same distributed setup as torchrun.
  for k,v in {'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'1','MASTER_ADDR':'127.0.0.1','MASTER_PORT':'29531'}.items():os.environ.__setitem__(k, os.environ.get(k) or v)
  sys.argv=[sys.argv[0],'--config',str(config)];engine.main()
if __name__=='__main__':main()
