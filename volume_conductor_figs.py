"""volume_conductor 결과 시각화. 한글 폰트가 없으므로 라벨은 ASCII."""
import sys, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import volume_conductor as V

OUT = 'result/volume_conductor'
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 130})

# ---------- Fig 1. geometry ----------
g = V.Geometry()
lab = V.build_labels(g)
yc = g.ny // 2
names = list(V.CONDUCTIVITY)
cols = ['#ffffff','#f2c9a0','#ffe680','#e8807f','#d9d9d9','#bfbfbf','#7fc7e8','#4da6ff']
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios':[2,1]})
sl = lab[:, yc, :].T
ax[0].imshow(sl, origin='upper', cmap=ListedColormap(cols),
             norm=BoundaryNorm(np.arange(-0.5, len(names)), len(names)),
             extent=[0, g.nx*g.h, g.nz*g.h, 0], aspect='equal', interpolation='nearest')
ax[0].axvline(g.x_joint, color='k', ls='--', lw=.8)
ax[0].text(g.x_joint+2, 6, 'joint line', fontsize=8)
ax[0].plot(g.x_joint, g.z_bone+6, 'r*', ms=13); ax[0].text(g.x_joint+3, g.z_bone+9,'source',color='r',fontsize=8)
ax[0].set_xlabel('x  proximal-distal (mm)'); ax[0].set_ylabel('z  depth from skin (mm)')
ax[0].set_title('(a) model cross-section   [skin/fat/muscle/bone/marrow/cartilage/synovial]')
handles=[plt.Rectangle((0,0),1,1,fc=cols[i]) for i in range(1,len(names))]
ax[1].legend(handles,[f'{n}  {V.CONDUCTIVITY[n]:g} S/m' for n in names[1:]],loc='center',frameon=False)
ax[1].axis('off'); ax[1].set_title('(b) conductivity (placeholder)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_geometry.png'); plt.close(fig)

# ---------- Fig 2. surface map + PSF ----------
A,_ = V._prepare(g)
res = {}
for axis,name in ((0,'tangential'),(2,'normal')):
    q = V.dipole_source(g, g.x_joint, g.ny*g.h/2, g.z_bone+6.0, axis=axis)
    phi,_ = V.solve_potential(A,q,g.shape); res[name]=V.surface_map(phi)
fig, ax = plt.subplots(2, 2, figsize=(9, 6))
for r,(name,surf) in enumerate(res.items()):
    m = np.abs(surf).max()
    im=ax[r,0].imshow(surf.T, origin='lower', cmap='RdBu_r', vmin=-m, vmax=m,
                      extent=[0,g.nx*g.h,0,g.ny*g.h], aspect='equal')
    ax[r,0].axvline(g.x_joint,color='k',ls='--',lw=.8)
    ax[r,0].set_title(f'({"ac"[r]}) skin surface potential  |  {name} dipole')
    ax[r,0].set_xlabel('x (mm)'); ax[r,0].set_ylabel('y  medial-lateral (mm)')
    plt.colorbar(im,ax=ax[r,0],fraction=.04)
    prof = surf[:, g.ny//2]
    ax[r,1].plot(np.arange(g.nx)*g.h, prof, 'k'); ax[r,1].axvline(g.x_joint,color='r',ls='--',lw=.8)
    ax[r,1].axhline(0,color='gray',lw=.5)
    pk=np.abs(prof).max(); ax[r,1].axhspan(-pk/2,pk/2,color='orange',alpha=.15)
    w=(np.abs(prof)>=pk/2).sum()*g.h
    ax[r,1].set_title(f'({"bd"[r]}) profile along limb axis   half-max width = {w:.0f} mm')
    ax[r,1].set_xlabel('x (mm)'); ax[r,1].set_ylabel('potential (a.u.)')
fig.suptitle('Point-spread: a focal cartilage source blurs to tens of mm at the skin', y=1.0)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_psf.png'); plt.close(fig)

# ---------- Fig 3. bone shielding ----------
fig, ax = plt.subplots(2, 2, figsize=(9, 6))
store={}
for r,(axis,name) in enumerate(((0,'tangential'),(2,'normal'))):
    for c,nb in enumerate((False,True)):
        gg = V.Geometry(no_bone=nb); AA,_ = V._prepare(gg)
        q = V.dipole_source(gg, gg.x_joint, gg.ny*gg.h/2, gg.z_bone+6.0, axis=axis)
        phi,_ = V.solve_potential(AA,q,gg.shape); s=V.surface_map(phi); store[(name,nb)]=s
for r,name in enumerate(('tangential','normal')):
    vmax=max(np.abs(store[(name,nb)]).max() for nb in (False,True))
    for c,nb in enumerate((False,True)):
        s=store[(name,nb)]
        im=ax[r,c].imshow(s.T,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax,
                          extent=[0,g.nx*g.h,0,g.ny*g.h],aspect='equal')
        ax[r,c].axvline(g.x_joint,color='k',ls='--',lw=.8)
        ax[r,c].set_title(f'{name} | {"bone REPLACED by muscle" if nb else "bone present"}\npeak |V| = {np.abs(s).max():.3g}')
        ax[r,c].set_xlabel('x (mm)'); ax[r,c].set_ylabel('y (mm)')
        plt.colorbar(im,ax=ax[r,c],fraction=.04)
fig.suptitle('Bone shielding: removing bone raises the potential over bone, not at the joint line', y=1.0)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_shield.png'); plt.close(fig)

# ---------- Fig 4. fat sweep ----------
fats=[2,4,6,8,10]; peaks=[]
for f in fats:
    gg=V.Geometry(d_fat=f); AA,_=V._prepare(gg)
    q=V.dipole_source(gg,gg.x_joint,gg.ny*gg.h/2,gg.z_bone+6.0)
    phi,_=V.solve_potential(AA,q,gg.shape); peaks.append(np.abs(V.surface_map(phi)).max())
fig,ax=plt.subplots(figsize=(5,3.4))
ax.plot(fats,np.array(peaks)/peaks[0],'o-k')
ax.axhline(1,color='gray',lw=.5,ls=':')
ax.set_xlabel('subcutaneous fat thickness (mm)'); ax.set_ylabel('peak surface potential (relative)')
ax.set_title('Thicker fat RAISES the potential\n(opposite to the sEMG intuition; matches the n=41 data)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_fat.png'); plt.close(fig)

# ---------- Fig 5. rank by layout ----------
gg=V.Geometry(); AA,_=V._prepare(gg)
par=V.cartilage_parcels(gg,4)
cols_=[]
for (x,y,z,side) in par:
    q=V.dipole_source(gg,x,y,z); phi,_=V.solve_potential(AA,q,gg.shape); cols_.append(V.surface_map(phi))
layouts=[(4,2),(2,4),(4,4),(8,2),(6,3)]
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
bars=[]
for (nx_e,ny_e) in layouts:
    el=V.electrode_grid(gg,nx_e,ny_e)
    L=np.stack([V.sample_at(s,gg,el) for s in cols_],axis=1); L=L-L.mean(0,keepdims=True)
    r=V.effective_rank(L,20.0); s=np.array(r['singular_values']); s=s/s[0]
    lbl=f'{nx_e}x{ny_e} (n={len(el)})'
    ax[0].semilogy(np.arange(1,len(s)+1),s,'o-',label=lbl,ms=4)
    bars.append((lbl,r['effective_rank'],r['cond']))
ax[0].axhline(10**(-20/20),color='r',ls='--',lw=.8); ax[0].text(6.2,10**(-20/20)*1.2,'noise floor (20 dB)',color='r',fontsize=8)
ax[0].set_xlabel('singular value index'); ax[0].set_ylabel('sigma_i / sigma_1')
ax[0].set_title('(a) lead-field spectrum: 8 cartilage parcels'); ax[0].legend(fontsize=8)
lbls=[b[0] for b in bars]; rk=[b[1] for b in bars]
bb=ax[1].bar(range(len(bars)),rk,color=['#bbb']*len(bars))
for i,b in enumerate(bars):
    if b[0].startswith('4x4'): bb[i].set_color('#d62728')
    ax[1].text(i,b[1]+.12,f'cond\n{b[2]:.0f}' if b[2]<1e3 else 'cond\n~1e16',ha='center',fontsize=7)
ax[1].set_xticks(range(len(bars))); ax[1].set_xticklabels(lbls,rotation=20,fontsize=8)
ax[1].set_ylabel('effective rank'); ax[1].set_ylim(0,9.5)
ax[1].axhline(8,color='k',ls=':',lw=.8); ax[1].text(0,8.15,'all 8 parcels',fontsize=8)
ax[1].set_title('(b) layout matters more than electrode count')
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_rank.png'); plt.close(fig)
print("saved:", *[f for f in sorted(os.listdir(OUT)) if f.endswith('.png')])
