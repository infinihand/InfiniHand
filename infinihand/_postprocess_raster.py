"""Triangle silhouette coverage for the mask-guided depth search."""
import numpy as np,torch,triton,triton.language as tl
@triton.jit
def raster(U,F,M,NF:tl.constexpr,NV:tl.constexpr,RES:tl.constexpr,B:tl.constexpr):
 face=tl.program_id(0);batch=tl.program_id(1)
 a=tl.load(F+face*3);b=tl.load(F+face*3+1);c=tl.load(F+face*3+2)
 off=batch*NV*2
 ax=tl.load(U+off+a*2);ay=tl.load(U+off+a*2+1);bx=tl.load(U+off+b*2);by=tl.load(U+off+b*2+1);cx=tl.load(U+off+c*2);cy=tl.load(U+off+c*2+1)
 xmin=tl.maximum(0,tl.floor(tl.minimum(ax,tl.minimum(bx,cx))).to(tl.int32));xmax=tl.minimum(RES-1,tl.ceil(tl.maximum(ax,tl.maximum(bx,cx))).to(tl.int32));ymin=tl.maximum(0,tl.floor(tl.minimum(ay,tl.minimum(by,cy))).to(tl.int32));ymax=tl.minimum(RES-1,tl.ceil(tl.maximum(ay,tl.maximum(by,cy))).to(tl.int32))
 width=tl.maximum(xmax-xmin+1,1);count=tl.maximum(xmax-xmin+1,0)*tl.maximum(ymax-ymin+1,0);area=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
 for start in range(0,tl.cdiv(count,B)):
  t=start*B+tl.arange(0,B);x=xmin+t%width;y=ymin+t//width
  e0=(bx-ax)*(y-ay)-(by-ay)*(x-ax);e1=(cx-bx)*(y-by)-(cy-by)*(x-bx);e2=(ax-cx)*(y-cy)-(ay-cy)*(x-cx)
  inside=((e0>=0)&(e1>=0)&(e2>=0))|((e0<=0)&(e1<=0)&(e2<=0));valid=(t<count)&inside&(tl.abs(area)>1e-7)
  tl.atomic_or(M+batch*RES*RES+y*RES+x,1,mask=valid,sem='relaxed')
def gpu_mask(uv,faces,res):
 buf=torch.zeros((uv.shape[0],res,res),device=uv.device,dtype=torch.int32)
 raster[(faces.shape[0],uv.shape[0])](uv.contiguous(),faces,buf,faces.shape[0],uv.shape[1],res,128)
 return buf.bool()
def cpu_mask(uv,faces,res):
 out=np.zeros((res,res),bool)
 for f in faces:
  a,b,c=uv[f];lo=np.maximum(np.floor(np.minimum(a,np.minimum(b,c))).astype(int),0);hi=np.minimum(np.ceil(np.maximum(a,np.maximum(b,c))).astype(int),res-1)
  if (hi<lo).any():continue
  x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));e0=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]);e1=(c[0]-b[0])*(y-b[1])-(c[1]-b[1])*(x-b[0]);e2=(a[0]-c[0])*(y-c[1])-(a[1]-c[1])*(x-c[0]);area=np.cross(b-a,c-a)
  if abs(area)>1e-7:out[lo[1]:hi[1]+1,lo[0]:hi[0]+1]|=((e0>=0)&(e1>=0)&(e2>=0))|((e0<=0)&(e1<=0)&(e2<=0))
 return out
