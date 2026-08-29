#!/usr/bin/env python3
"""Deterministically render one N16/96 unseen case as paired exterior and interior views.

The renderer reads accepted arrays only. It never scores, never aligns labels, never
resamples between saved steps, and never selects a subset of saved steps.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, sys
from pathlib import Path
import matplotlib
import matplotlib.ft2font
import numpy as np
import PIL
import scipy
from PIL import Image, ImageDraw, ImageFont

PALETTE=np.array([(47,112,168),(247,135,37),(56,148,68),(154,98,204),(224,184,22),(61,174,191),(181,81,161),(83,130,42),(141,160,235),(199,140,115),(107,214,178),(146,109,156),(169,178,104),(173,231,255),(39,28,112),(192,242,179)],dtype=np.uint8)
STEPS=list(range(0,3201,200))
CASE='C3'; PUBLIC='Unseen microstructure 3'
DYING=[(8,'Grain 1'),(9,'Grain 2'),(15,'Grain 3')]
DYING_CHANNELS=[8,9,15]
N=96; TERMINAL_IDX=8
PANEL=384; W=816; H=940; POSTER_EXTRA=80

EXPECTED_CHECKPOINT_SHA256='9e15e6b74f4f92f37d5709a242e436f721ec5d1306a30e05e5f39794de4459dd'
EXPECTED_D_REF={8:1000,9:1400,15:1400}
PINS={'python':'3.11.15','numpy':'2.2.6','matplotlib':'3.10.7','pillow':'12.2.0','scipy':'1.16.2','imageio':'2.37.3','freetype':'2.14.3'}

EDGE_GRAY=(205,205,205); FRONT_EDGE_GRAY=(150,150,150)
CUBE_EDGES=(
    ((0,0,0),(N-1,0,0)),((0,0,0),(0,N-1,0)),((0,0,0),(0,0,N-1)),
    ((N-1,0,0),(N-1,N-1,0)),((N-1,0,0),(N-1,0,N-1)),
    ((0,N-1,0),(N-1,N-1,0)),((0,N-1,0),(0,N-1,N-1)),
    ((0,0,N-1),(N-1,0,N-1)),((0,0,N-1),(0,N-1,N-1)),
    ((N-1,N-1,0),(N-1,N-1,N-1)),((N-1,0,N-1),(N-1,N-1,N-1)),((0,N-1,N-1),(N-1,N-1,N-1)),
)
FRONT_EDGES=(
    ((0,N-1,N-1),(N-1,N-1,N-1)),((N-1,0,N-1),(N-1,N-1,N-1)),((N-1,N-1,0),(N-1,N-1,N-1)),
)

TITLE_Y=10; COLHDR_Y=44
ROW_LABEL_Y=(72,486); ROW_PANEL_Y=(94,508)
COL_X=(12,420)
LEGEND_TEXT_Y=903; LEGEND_SWATCH_Y=905; LEGEND_SWATCH=16
LEGEND_SWATCH_X=(12,115,218); LEGEND_LABEL_X=(34,137,240); LEGEND_NOTE_X=336
LEGEND_NOTE='Grain 1-3 denotes reference disappearance order'
STATUS_LINES=(
    'Unseen microstructure 3 - terminal agreement 95.83% at step 1600',
    'Cohort: exact terminal topology and extinction identities: 6/6',
    'Cohort: complete predefined qualification: 5/6',
)
STATUS_TEXT_X=38; STATUS_TEXT_Y=(947,969,991)

ALT_TEXT=('One unseen 96-cubed microstructure, Unseen microstructure 3, shown as paired reference and '
          'PINN-Phase rows over saved steps 0 to 3200. Each row pairs an exterior view of the intact cube '
          'with an interior view in which only the three disappearing grains are rendered inside a wireframe '
          'outline. The three grains shrink and vanish inside the volume, and at its last saved appearance '
          'each grain has no voxels on any rendered face of the cube. Step 1600 is the terminal evaluation '
          'and later frames are monitored continuation.')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def camera_metadata(n):
    """Public record of the affine voxel mapping implemented below."""
    return {'projection':'custom affine parallel voxel mapping',
            'image_u':'x - y + (n - 1)',
            'image_v':'integer rasterization of (x + y - z) / 2 + floor(n / 2)',
            'raster_overwrite_order':'ascending x + y + z before raster overwrite',
            'exterior_view':'intact cube, no cutaway',
            'interior_view':'three disappearing grains as solid voxel bodies inside the full-cube wireframe; all other grains suppressed; no cutaway'}

def assert_environment():
    seen={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
          'pillow':PIL.__version__,'scipy':scipy.__version__,
          'imageio':importlib.metadata.version('imageio'),
          'freetype':matplotlib.ft2font.__freetype_version__}
    for k,v in PINS.items():
        if seen[k]!=v: raise RuntimeError(f'exact media environment required: {k}=={v}, found {seen[k]}')
    return seen

def font_file():
    path=Path(matplotlib.get_data_path())/'fonts'/'ttf'/'DejaVuSans.ttf'
    if not path.is_file(): raise RuntimeError(f'exact media font unavailable: {path}')
    return path

def guard_out(out: Path, science_root: Path):
    out_real=os.path.realpath(str(out))
    root_real=os.path.realpath(str(science_root))
    if os.path.commonpath([out_real,root_real])==root_real:
        print(f'refusing to write inside the science root: {out_real} is under {root_real}',file=sys.stderr)
        sys.exit(2)

# The public camera metadata records this exact affine mapping and overwrite order.
_X,_Y,_Z=np.indices((N,N,N))
_U=((_X-_Y)+(N-1)).astype(np.int16)
_V=((_X+_Y)/2-_Z/2+N//2).astype(np.int16)
_DEPTH=_X+_Y+_Z
_ORDER=np.argsort(_DEPTH.reshape(-1))
U_ORD=_U.reshape(-1)[_ORDER]; V_ORD=_V.reshape(-1)[_ORDER]

def project_point(x,y,z):
    u=np.asarray((x-y)+(N-1)).astype(np.int16)
    v=np.asarray((np.asarray(x)+np.asarray(y))/2-np.asarray(z)/2+N//2).astype(np.int16)
    return u,v

def draw_edges(canvas, edges, colour):
    for (a,b) in edges:
        axis=[i for i in range(3) if a[i]!=b[i]][0]
        pts=[np.full(PANEL,float(a[i])) for i in range(3)]
        pts[axis]=np.linspace(float(a[axis]),float(b[axis]),PANEL)
        u,v=project_point(pts[0],pts[1],pts[2])
        canvas[v,u]=colour

def upscale(canvas):
    return Image.fromarray(canvas).resize((PANEL,PANEL),Image.Resampling.NEAREST)

def exterior(labels_3d):
    canvas=np.full((2*N,2*N,3),255,dtype=np.uint8)
    lab=labels_3d.reshape(-1)[_ORDER]
    canvas[V_ORD,U_ORD]=PALETTE[lab]
    return upscale(canvas)

def interior(labels_3d):
    canvas=np.full((2*N,2*N,3),255,dtype=np.uint8)
    draw_edges(canvas,CUBE_EDGES,EDGE_GRAY)
    lab=labels_3d.reshape(-1)[_ORDER]
    keep=np.isin(lab,DYING_CHANNELS)
    canvas[V_ORD[keep],U_ORD[keep]]=PALETTE[lab[keep]]
    draw_edges(canvas,FRONT_EDGES,FRONT_EDGE_GRAY)
    return upscale(canvas)

def step_title(index):
    text=f'Step {STEPS[index]}'
    if index==TERMINAL_IDX: return text+' - Terminal evaluation'
    if STEPS[index]>1600: return text+' - Monitored continuation'
    return text

def frame(ref_labels, model_labels, index):
    im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
    font=ImageFont.truetype(font_file(),size=22); small=ImageFont.truetype(font_file(),size=18)
    d.text((COL_X[0],TITLE_Y),PUBLIC,fill='black',font=font)
    label=step_title(index)
    d.text((W-12-int(d.textlength(label,font=font)),TITLE_Y),label,fill='black',font=font)
    d.text((COL_X[0],COLHDR_Y),'Exterior view',fill='black',font=small)
    d.text((COL_X[1],COLHDR_Y),'Interior - disappearing grains only',fill='black',font=small)
    for row,(name,labels) in enumerate((('Reference',ref_labels),('PINN-Phase',model_labels))):
        d.text((COL_X[0],ROW_LABEL_Y[row]),name,fill='black',font=small)
        im.paste(exterior(labels[index]),(COL_X[0],ROW_PANEL_Y[row]))
        im.paste(interior(labels[index]),(COL_X[1],ROW_PANEL_Y[row]))
    for i,(channel,name) in enumerate(DYING):
        x=LEGEND_SWATCH_X[i]
        d.rectangle((x,LEGEND_SWATCH_Y,x+LEGEND_SWATCH-1,LEGEND_SWATCH_Y+LEGEND_SWATCH-1),
                    fill=tuple(int(c) for c in PALETTE[channel]))
        d.text((LEGEND_LABEL_X[i],LEGEND_TEXT_Y),name,fill='black',font=small)
    d.text((LEGEND_NOTE_X,LEGEND_TEXT_Y),LEGEND_NOTE,fill='black',font=small)
    return im

def poster(ref_labels, model_labels):
    base=frame(ref_labels,model_labels,TERMINAL_IDX)
    im=Image.new('RGB',(W,H+POSTER_EXTRA),'white'); im.paste(base,(0,0))
    d=ImageDraw.Draw(im); small=ImageFont.truetype(font_file(),size=18)
    d.rectangle((20,H+6,W-20,H+74),outline='black',width=2)
    for y,line in zip(STATUS_TEXT_Y,STATUS_LINES,strict=True):
        d.text((STATUS_TEXT_X,y),line,fill='black',font=small)
    return im

def load_labels(path):
    with np.load(path,allow_pickle=False) as z:
        frames=z['frames']; steps=z['save_steps'].tolist()
        if steps!=STEPS: raise AssertionError(f'unexpected save_steps in {path}: {steps}')
        if frames.shape!=(17,16,N,N,N): raise AssertionError(f'unexpected shape in {path}: {frames.shape}')
        return np.argmax(frames,axis=1).astype(np.uint8)

def outer_face_count(volume_mask):
    m=volume_mask.copy(); m[1:-1,1:-1,1:-1]=False
    return int(m.sum())

def visibility(labels):
    per_channel={}
    for channel in DYING_CHANNELS:
        mask=(labels==channel)
        totals=mask.reshape(17,-1).sum(axis=1)
        if not totals.any(): raise AssertionError(f'channel {channel} never present')
        last=int(np.max(np.nonzero(totals)[0]))
        faces=[outer_face_count(mask[i]) for i in range(17)]
        if faces[last]!=0:
            raise AssertionError(f'channel {channel} has {faces[last]} outer-face voxels at its last saved appearance')
        per_channel[str(channel)]=faces
    return per_channel

def main():
    seen=assert_environment()
    ap=argparse.ArgumentParser()
    ap.add_argument('--science-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    guard_out(a.out,a.science_root)
    a.out.mkdir(parents=True,exist_ok=True)

    rp=a.science_root/f'references/{CASE}/ref_frames.npz'
    mp=a.science_root/f'rollouts/{CASE}/model_frames.npz'
    ref=load_labels(rp); model=load_labels(mp)

    score=a.science_root/'score'/'per_case'/f'{CASE}.json'
    events=json.loads(score.read_text())['G5_events']
    d_ref={int(k):v for k,v in events['d_ref'].items()}
    d_model={int(k):v for k,v in events['d_model'].items()}
    for channel,step in EXPECTED_D_REF.items():
        if d_ref.get(channel)!=step:
            raise AssertionError(f'reference first-absent step for channel {channel} is {d_ref.get(channel)}, expected {step}')
    cohort=a.science_root/'cohort'/'COHORT_MANIFEST.json'
    checkpoint=json.loads(cohort.read_text())['expected_checkpoint_sha256']
    if checkpoint!=EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f'unexpected checkpoint digest {checkpoint}')

    ref_faces=visibility(ref); model_faces=visibility(model)

    frames=[frame(ref,model,i) for i in range(17)]
    gif=a.out/'n16_96_unseen_microstructure_3_interior_reference_vs_pinn_phase.gif'
    png=a.out/'n16_96_unseen_microstructure_3_interior_reference_vs_pinn_phase_poster.png'
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=550,loop=0,disposal=2,optimize=False)
    poster(ref,model).save(png)

    lock=Path(__file__).resolve().parents[1]/'environment-media.yml'
    ft=font_file()
    rec={
        'schema':'pinn-phase-n16-96-interior-media-v1',
        'alt_text':ALT_TEXT,
        'camera':camera_metadata(N),
        'selection':{'rule':'maximise the number of disappearing grains with zero outer-face voxels at their last-present saved step; tie-break by largest pooled median distance of dying-grain voxels from the nearest rendered face',
                     'selected_case':CASE,'public_name':PUBLIC,'zero_outer_face_grains':'3 of 3',
                     'runner_up_note':'C3 is the only case with 3 of 3; every other case has 2 of 3'},
        'dying_grains':{name:{'channel':channel,
                              'reference_first_absent_step':d_ref[channel],
                              'model_first_absent_step':d_model[channel]} for channel,name in DYING},
        'interior_visibility':{'outer_face_voxels_per_saved_step':{'reference':ref_faces,'model':model_faces},
                               'zero_at_last_present':True,
                               'note':'channels 8 and 15 are visible on the rendered faces at early saved steps; each disappearing grain has zero rendered-face voxels at its last saved appearance'},
        'claims_source':{'terminal_agreement_percent':95.831412,'displayed_as':'95.83%',
                         'source':'benchmarks/n16_96_transfer/expected_score.json',
                         'source_sha256':'9f12027fa05063df7e9dd0f6abcb5135a302fd07fe66b304905cf25fa30aaaed'},
        'case_score_sha256':digest(score),
        'checkpoint_sha256':checkpoint,
        'cohort_manifest_sha256':digest(cohort),
        'cohort_score_sha256':digest(a.science_root/'score'/'COHORT_SCORE.json'),
        'environment_lock_sha256':digest(lock),
        'inputs':[{'case':CASE,'role':'reference','sha256':digest(rp)},
                  {'case':CASE,'role':'model','sha256':digest(mp)}],
        'no_interpolation':True,
        'outputs':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in (gif,png)},
        'renderer_environment':{'python':seen['python'],'numpy':seen['numpy'],'matplotlib':seen['matplotlib'],
                                'freetype':seen['freetype'],'pillow':seen['pillow'],'scipy':seen['scipy'],
                                'imageio':seen['imageio'],'font_file':ft.name,'font_sha256':digest(ft)},
        'renderer_sha256':digest(Path(__file__)),
        'steps':STEPS,
    }
    rec['dying_grains']['reference_tie_note']=('Grain 2 and Grain 3 are first absent at the same saved step (1400); '
                                               'numbering within the tie follows ascending channel index')
    (a.out/'asset_manifest.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
