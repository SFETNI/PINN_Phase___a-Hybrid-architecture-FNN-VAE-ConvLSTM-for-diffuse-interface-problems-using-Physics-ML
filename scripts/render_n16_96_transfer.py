#!/usr/bin/env python3
"""Deterministically render the all-six N16/96 cohort from accepted arrays only.

The renderer makes paired custom affine/parallel voxel cutaways from raw argmax labels.
It never scores, aligns labels, interpolates frames, or selects a subset of cases.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE=np.array([(47,112,168),(247,135,37),(56,148,68),(154,98,204),(224,184,22),(61,174,191),(181,81,161),(83,130,42),(141,160,235),(199,140,115),(107,214,178),(146,109,156),(169,178,104),(173,231,255),(39,28,112),(192,242,179)],dtype=np.uint8)
STEPS=list(range(0,3201,200)); CASES=('C1','C2','C3','C4','C5','C6')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def camera_metadata(n):
    """Public record of the affine voxel mapping implemented below."""
    return {'projection':'custom affine parallel voxel mapping',
            'image_u':'x - y + (n - 1)',
            'image_v':'integer rasterization of (x + y - z) / 2 + floor(n / 2)',
            'raster_overwrite_order':'ascending x + y + z before raster overwrite',
            'cutaway':f'removed octant x,y,z >= {n//2}'}
def cutaway(labels, px=225):
    n=labels.shape[0]; x,y,z=np.indices(labels.shape); keep=~((x>=n//2)&(y>=n//2)&(z>=n//2))
    # The public camera metadata records this exact affine mapping and overwrite order.
    u=((x-y)+(n-1)).astype(np.int16); v=((x+y)/2-z/2+n//2).astype(np.int16); depth=x+y+z
    u=u[keep]; v=v[keep]; d=depth[keep]; lab=labels[keep]
    order=np.argsort(d); u=u[order];v=v[order];lab=lab[order]
    canvas=np.full((2*n,2*n,3),255,dtype=np.uint8); canvas[v,u]=PALETTE[lab]
    im=Image.fromarray(canvas).resize((px,px),Image.Resampling.NEAREST)
    return im
def font_file():
    path=Path(matplotlib.get_data_path())/'fonts'/'ttf'/'DejaVuSans.ttf'
    if not path.is_file(): raise RuntimeError(f'exact media font unavailable: {path}')
    return path
def frame(refs, models, index, status=False):
    w,h=1500,860 if status else 800; im=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(im)
    font=ImageFont.truetype(font_file(),size=22); small=ImageFont.truetype(font_file(),size=18)
    title=f'Terminal evaluation - Step 1600' if status else (f'Step {STEPS[index]}' + (' - Monitored continuation' if STEPS[index]>1600 else ''))
    d.text((20,14),title,fill='black',font=font)
    for i,(r,m) in enumerate(zip(refs,models)):
        col=i%3; row=i//3; x=col*500+12; y=52+row*370
        im.paste(cutaway(r[index]),(x,y+32)); im.paste(cutaway(m[index]),(x+250,y+32))
        d.text((x,y),f'Unseen microstructure {i+1}',fill='black',font=font)
        d.text((x+42,y+270),'Reference',fill='black',font=small); d.text((x+280,y+270),'PINN-Phase',fill='black',font=small)
    if status:
        d.rectangle((20,770,1480,842),outline='black',width=2)
        d.text((38,782),'Exact terminal topology and extinction identities: 6/6',fill='black',font=font)
        d.text((38,812),'Complete predefined qualification: 5/6',fill='black',font=font)
    return im
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--science-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    refs=[];models=[];inputs=[]
    for cid in CASES:
        rp=a.science_root/f'references/{cid}/ref_frames.npz'; mp=a.science_root/f'rollouts/{cid}/model_frames.npz'
        with np.load(rp,allow_pickle=False) as z: refs.append(np.argmax(z['frames'],axis=1).astype(np.uint8))
        with np.load(mp,allow_pickle=False) as z: models.append(np.argmax(z['frames'],axis=1).astype(np.uint8))
        inputs += [{'case':cid,'role':'reference','sha256':digest(rp)},{'case':cid,'role':'model','sha256':digest(mp)}]
    frames=[frame(refs,models,i) for i in range(17)]; gif=a.out/'n16_96_unseen_cohort_reference_vs_pinn_phase.gif'; poster=a.out/'n16_96_unseen_cohort_reference_vs_pinn_phase_poster.png'
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=550,loop=0,disposal=2,optimize=False); frame(refs,models,8,status=True).save(poster)
    lock=Path(__file__).resolve().parents[1]/'environment-media.yml'
    score=a.science_root/'score'/'COHORT_SCORE.json'; cohort=a.science_root/'cohort'/'COHORT_MANIFEST.json'; ft=font_file()
    rec={'schema':'pinn-phase-n16-96-media-v1','alt_text':'Six unseen 96-cubed microstructures arranged in a three-by-two grid, each comparing the reference simulation with the same frozen PINN-Phase model over saved steps 0 to 3200.','renderer_sha256':digest(Path(__file__)),'environment_lock_sha256':digest(lock),'renderer_environment':{'python':'3.11.15','numpy':'2.2.6','matplotlib':'3.10.7','freetype':'2.14.3','pillow':'12.2.0','scipy':'1.16.2','imageio':'2.37.3','font_file':ft.name,'font_sha256':digest(ft)},'cohort_score_sha256':digest(score),'cohort_manifest_sha256':digest(cohort),'checkpoint_sha256':json.loads(cohort.read_text())['expected_checkpoint_sha256'],'inputs':inputs,'camera':camera_metadata(refs[0].shape[1]),'steps':STEPS,'no_interpolation':True,'outputs':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in (gif,poster)}}
    (a.out/'asset_manifest.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
