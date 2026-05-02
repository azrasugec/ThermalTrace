"""
Pipeline:
  1. Anchor detection       : hot/cold pixel per frame
  2. Frame registration     : align via hot anchor (UAV drift correction)
  3. Extreme pixel masking  : NaN-mask anchors
  4. Temporal median stack  : noise suppression (~10x SNR improvement)
  5. Histogram Projection   : stretch occupied bins to 0-255 (Zhang et al. 2014)
  6. CLAHE                  : local contrast enhancement (Pizer et al. 1987)
  7. Anomaly detection      : Gaussian background subtraction
  8. Naive baseline         : comparison without registration
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.ndimage import gaussian_filter

# ── Ayarlar ───────────────────────────────────────────────────────────────────
IMAGE_DIR  = r'C:/Users/azras/OneDrive/Masaüstü/sayısal 1/2220674062/2220674062/'
OUTPUT_DIR = r'C:/Users/azras/OneDrive/Masaüstü/sayısal 1/outputs/'

HOT_THRESH  = 240   # bu değerin üzeri = hot anchor artifact
COLD_THRESH = 20    # bu değerin altı  = cold anchor artifact

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Görüntüleri yükle ─────────────────────────────────────────────────────────
# glob yerine os.listdir kullan — Türkçe klasör ismi sorun çıkarmaması için
paths = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.startswith('thermal_image_') and f.endswith('.png')
])
print(f'{len(paths)} görüntü yüklendi')

# hepsini grayscale float32 olarak yükle → (N, H, W) array
frames = np.stack([
    np.array(Image.open(p).convert('L')).astype(np.float32)
    for p in paths
], axis=0)

N, H, W = frames.shape
print(f'Stack boyutu: {frames.shape}')

# ham frame'i göster
plt.figure(figsize=(5, 5))
plt.imshow(frames[0], cmap='gray', vmin=0, vmax=255)
plt.title('Ham Frame — hedef görünmüyor')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_raw_frame.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Ham frame histogram analizi ───────────────────────────────────────────────
# neden hedef görünmüyor? histogram çok seyrek!
hist_raw, _ = np.histogram(frames[0], bins=256, range=(0, 255))
occupied = np.sum(hist_raw > 0)

plt.figure(figsize=(10, 4))
plt.bar(range(256), hist_raw, color='steelblue', width=1)
plt.axvline(HOT_THRESH,  color='red',  lw=2, ls='--', label=f'Hot eşik ({HOT_THRESH})')
plt.axvline(COLD_THRESH, color='cyan', lw=2, ls='--', label=f'Cold eşik ({COLD_THRESH})')
plt.title(f'Ham Frame Histogramı — {occupied}/256 bin dolu, gerisi boş!')
plt.xlabel('Gri Seviye')
plt.ylabel('Piksel Sayısı')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_histogram_raw.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f'Dolu bin: {occupied} | Boş bin: {256 - occupied}')

# ── Anchor piksellerini bul ───────────────────────────────────────────────────
# her frame'deki hot ve cold anchor pikselinin konumunu tespit et
hot_locs, cold_locs = [], []

for i in range(N):
    hot  = np.argwhere(frames[i] > HOT_THRESH)
    cold = np.argwhere(frames[i] < COLD_THRESH)
    # piksel bulunamazsa ortalama konumu kullan (fallback)
    hot_locs.append(hot[0]   if len(hot)  else np.array([80, 79]))
    cold_locs.append(cold[0] if len(cold) else np.array([593, 592]))

hot_locs  = np.array(hot_locs,  dtype=float)
cold_locs = np.array(cold_locs, dtype=float)

print(f'Hot anchor merkezi : ({hot_locs[:,0].mean():.0f}, {hot_locs[:,1].mean():.0f})')
print(f'Cold anchor merkezi: ({cold_locs[:,0].mean():.0f}, {cold_locs[:,1].mean():.0f})')

# 100 frame boyunca anchor'ların kaymasını görselleştir → UAV hareketi
plt.figure(figsize=(6, 6))
plt.gca().set_facecolor('black')
plt.scatter(hot_locs[:,1],  hot_locs[:,0],  c='red',  s=30, alpha=0.8, label='Hot anchor')
plt.scatter(cold_locs[:,1], cold_locs[:,0], c='cyan', s=30, alpha=0.8, label='Cold anchor')
plt.xlim(0, W)
plt.ylim(H, 0)
plt.gca().set_aspect('equal')
plt.title('100 Frame Boyunca Anchor Kayması (UAV hareketi)')
plt.xlabel('X px')
plt.ylabel('Y px')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_anchor_drift.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Frame registration ────────────────────────────────────────────────────────
# hot anchor'ı referans alarak tüm frame'leri öteleme ile hizala
# Δy = r0 - ri,  Δx = c0 - ci
ref_hot    = hot_locs[0]       # frame 0 referans
registered = np.zeros_like(frames)
shifts     = []

for i in range(N):
    dy = int(round(ref_hot[0] - hot_locs[i][0]))  # dikey öteleme miktarı
    dx = int(round(ref_hot[1] - hot_locs[i][1]))  # yatay öteleme miktarı
    shifts.append((dy, dx))
    shifted = np.roll(frames[i], dy, axis=0)       # frame'i dikey ötele
    shifted = np.roll(shifted,   dx, axis=1)       # frame'i yatay ötele
    registered[i] = shifted

shifts = np.array(shifts)
max_dy = int(np.abs(shifts[:,0]).max()) + 1
max_dx = int(np.abs(shifts[:,1]).max()) + 1

# np.roll kenarlarda sarma yapar, o kısımları kırp
registered = registered[:, max_dy:-max_dy, max_dx:-max_dx]
print(f'Hizalama tamamlandı — geçerli boyut: {registered.shape[1]}x{registered.shape[2]}')

# ── Maskeleme + Median stack ──────────────────────────────────────────────────
# hot/cold anchor piksellerini NaN yap — stack hesabını kirletmesin
masked = registered.astype(float)
masked[registered > HOT_THRESH] = np.nan
masked[registered < COLD_THRESH] = np.nan

# 100 frame'in median'ı: rastgele gürültü iptal olur, sabit hedef kalır
# SNR kazanımı ≈ √N = √100 = 10×
print('Median stack hesaplanıyor...')
median_stack = np.nanmedian(masked, axis=0)

# eğer bir piksel tüm framelerde NaN idiyse ortalama ile doldur
fill_val = np.nanmean(median_stack)
median_stack = np.where(np.isnan(median_stack), fill_val, median_stack)

print(f'Median stack aralığı: {median_stack.min():.1f} — {median_stack.max():.1f} GL')
print(f'Dinamik aralık: {median_stack.max()-median_stack.min():.1f} GL '
      f'(ham framedeydi ~160 GL, gürültü temizlendi)')

plt.figure(figsize=(5, 5))
plt.imshow(median_stack, cmap='gray')
plt.title('Registered Median Stack — Y şekli beliriyor!')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_median_stack.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Histogram Projection ──────────────────────────────────────────────────────
# Zhang et al. (2014): sadece dolu gri seviyeleri 0-255 aralığına eşit paylaştır
# boş bin'leri silerek HDR kırpıntı problemini çözer
def histogram_projection(image_float):
    img_u8     = np.clip(image_float, 0, 255).astype(np.uint8)
    hist, _    = np.histogram(img_u8, bins=256, range=(0, 255))
    occupancy  = (hist > 0).astype(np.int32)   # 1=dolu bin, 0=boş bin
    S          = occupancy.sum()               # kaç bin dolu
    cumulative = np.cumsum(occupancy)
    lut        = np.floor(255 * cumulative / S).astype(np.uint8)  # lookup table
    return lut[img_u8].astype(np.float32), S

hp_result, n_valid = histogram_projection(median_stack)
print(f'Dolu bin: {n_valid} → 0-255 aralığına yayıldı')

# histogramların karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
h1, _ = np.histogram(median_stack, bins=256, range=(0, 255))
h2, _ = np.histogram(hp_result,    bins=256, range=(0, 255))
axes[0].bar(range(256), h1, color='darkorange', width=1)
axes[0].set_title(f'Median Stack ({n_valid} bin dolu)')
axes[0].set_xlabel('Gri Seviye')
axes[0].set_ylabel('Piksel Sayısı')
axes[1].bar(range(256), h2, color='green', width=1)
axes[1].set_title('Histogram Projection sonrası (tam aralık)')
axes[1].set_xlabel('Gri Seviye')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6_histogram_projection.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── CLAHE ─────────────────────────────────────────────────────────────────────
# Pizer et al. (1987): 8x8 kutucuklarda histogram eşitleme
# clipLimit gürültü patlamasını önler
clahe_obj    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_result = clahe_obj.apply(hp_result.astype(np.uint8))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(frames[0], cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Ham Frame')
axes[0].axis('off')
axes[1].imshow(clahe_result, cmap='inferno')
axes[1].set_title('Son Sonuç — Hedef Görünür!')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '7_clahe_result.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Anomali tespiti ───────────────────────────────────────────────────────────
# büyük sigma ile çok yumuşak arka plan modeli
# arka plandan çıkar → yerelleşmiş sıcak/soğuk sapmalar = hedef sinyali
background = gaussian_filter(median_stack, sigma=20)
detail     = median_stack - background

m = 30  # kenar marjı
interior  = detail[m:-m, m:-m]
warm_pos  = np.unravel_index(interior.argmax(), interior.shape)
cold_pos  = np.unravel_index(interior.argmin(), interior.shape)
warm_y, warm_x = warm_pos[0]+m, warm_pos[1]+m
cold_y,  cold_x  = cold_pos[0]+m,  cold_pos[1]+m

print(f'Sıcak anomali: piksel ({warm_x}, {warm_y}), güç = +{interior.max():.2f} GL')
print(f'Soğuk anomali: piksel ({cold_x}, {cold_y}),  güç = {interior.min():.2f} GL')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.imshow(clahe_result, cmap='inferno')
circ = plt.Circle((warm_x, warm_y), 20, color='lime', fill=False, lw=2.5)
ax.add_patch(circ)
ax.annotate('HEDEF', xy=(warm_x, warm_y),
            xytext=(warm_x+60, warm_y-60), color='lime',
            fontsize=12, fontweight='bold',
            arrowprops=dict(color='lime', arrowstyle='->', lw=2))
ax.set_title('Son Sonuç + Hedef İşareti')
ax.axis('off')

ax = axes[1]
vmax = np.abs(detail).max()
im   = ax.imshow(detail, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.plot(warm_x, warm_y, 'g+', ms=18, mew=2.5, label='Sıcak hedef')
ax.plot(cold_x,  cold_y,  'k+', ms=18, mew=2.5, label='Soğuk hedef')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ΔT (GL)')
ax.set_title('Arka Plan Çıkarılmış Anomali Haritası')
ax.legend(fontsize=9)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '8_anomaly_map.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Tam pipeline özet figürü ──────────────────────────────────────────────────
# rapora koymak için 6 panelli özet
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('GMT234 Assignment I — Gizli Termal Hedefin Ortaya Çıkarılması\n'
             'Yöntem: Anchor Registration → Median Stack → Hist. Projection → CLAHE',
             fontsize=13, fontweight='bold')

ax = axes[0, 0]
ax.imshow(frames[0], cmap='gray', vmin=0, vmax=255)
hot0  = np.argwhere(frames[0] > HOT_THRESH)
cold0 = np.argwhere(frames[0] < COLD_THRESH)
if len(hot0):  ax.plot(hot0[0,1],  hot0[0,0],  'r*', ms=14, label='Hot anchor')
if len(cold0): ax.plot(cold0[0,1], cold0[0,0], 'b*', ms=14, label='Cold anchor')
ax.set_title('(a) Ham Tek Frame\n(hedef görünmüyor)', fontsize=10)
ax.legend(fontsize=8)
ax.axis('off')

ax = axes[0, 1]
ax.set_facecolor('black')
ax.scatter(hot_locs[:,1],  hot_locs[:,0],  c='red',  s=20, alpha=0.8, label='Hot anchor')
ax.scatter(cold_locs[:,1], cold_locs[:,0], c='cyan', s=20, alpha=0.8, label='Cold anchor')
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect('equal')
ax.set_title('(b) 100 Framede Anchor Kayması\n(UAV hareketi)', fontsize=10)
ax.legend(fontsize=8)
ax.set_xlabel('X px')
ax.set_ylabel('Y px')

ax = axes[0, 2]
ax.imshow(median_stack, cmap='gray')
ax.set_title('(c) Hizalanmış Median Stack\n(gürültü iptal, sinyal korundu)', fontsize=10)
ax.axis('off')

ax = axes[1, 0]
ax.imshow(hp_result, cmap='gray')
ax.set_title(f'(d) + Histogram Projection\n({n_valid} bin → 0-255)', fontsize=10)
ax.axis('off')

ax = axes[1, 1]
ax.imshow(clahe_result, cmap='inferno')
circ = plt.Circle((warm_x, warm_y), 18, color='lime', fill=False, lw=2.5)
ax.add_patch(circ)
ax.annotate('HEDEF', xy=(warm_x, warm_y),
            xytext=(warm_x+55, warm_y-55), color='lime',
            fontsize=12, fontweight='bold',
            arrowprops=dict(color='lime', arrowstyle='->', lw=1.5))
ax.set_title('(e) + CLAHE — SON SONUÇ\n(gizli hedef ortaya çıktı!)', fontsize=10)
ax.axis('off')

ax = axes[1, 2]
vmax = np.abs(detail).max()
im = ax.imshow(detail, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.plot(warm_x, warm_y, 'g+', ms=15, mew=2, label='Sıcak hedef')
ax.plot(cold_x,  cold_y,  'k+', ms=15, mew=2, label='Soğuk hedef')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ΔT (GL)')
ax.set_title('(f) Arka Plan Çıkarılmış Harita\n(kırmızı=sıcak, mavi=soğuk)', fontsize=10)
ax.legend(fontsize=8)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '9_pipeline_result.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Pipeline özet figürü kaydedildi ✓')

# ── Temel yaklaşım karşılaştırması ───────────────────────────────────────────
# registration olmadan mean + clip + HE — ödev ipucu: hot/cold objeler histogramı bozuyor
mean_img = np.mean(frames, axis=0)          # 100 frame ortalaması
clipped  = np.clip(mean_img, 20, 240)       # extreme pikselleri kırp
naive    = ((clipped - clipped.min()) /
            (clipped.max() - clipped.min()) * 255).astype('uint8')  # normalize
he_naive = cv2.equalizeHist(naive)          # histogram eşitleme

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Temel Yaklaşım vs Geliştirilen Yöntem', fontsize=13, fontweight='bold')

axes[0].imshow(frames[0], cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Ham Frame\n(hedef görünmüyor)')
axes[0].axis('off')

axes[1].imshow(he_naive, cmap='gray')
axes[1].set_title('Temel: Ortalama + Clip + HE\n(registration yok → bulanık, hedef belirsiz)')
axes[1].axis('off')

axes[2].imshow(clahe_result, cmap='inferno')
axes[2].set_title('Geliştirilen: Registration + Median\n+ Hist.Projection + CLAHE → Hedef Net!')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_naive_vs_advanced.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Sonuçları kaydet ─────────────────────────────────────────────────────────
Image.fromarray(clahe_result).save(os.path.join(OUTPUT_DIR, 'final_result.png'))

print('=' * 50)
print('TAMAMLANDI')
print('=' * 50)
print(f'İşlenen frame sayısı : {N}')
print(f'Gürültü azalma oranı : ~√{N} = {N**0.5:.1f}x')
print(f'Sıcak hedef konumu   : piksel ({warm_x}, {warm_y})')
print(f'Soğuk hedef konumu   : piksel ({cold_x}, {cold_y})')
print(f'Sıcak anomali gücü   : +{interior.max():.2f} GL')
print(f'Dosyalar kaydedildi  : {OUTPUT_DIR}')
