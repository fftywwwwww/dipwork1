import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


img = cv2.imread('/remote-home/yjk/dip_work3/face256.BMP', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image path error! Cannot read BMP file.")


def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

noisy_img = add_gaussian_noise(img, mean=0, sigma=25)

denoised_gaussian = cv2.GaussianBlur(noisy_img, (5,5), 1)
denoised_median = cv2.medianBlur(noisy_img, 5)
denoised_bilateral = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)


def wavelet_bayes(image, wavelet='db8', level=3, sigma=25):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, cDs = coeffs[0], coeffs[1:]
    new_cDs = []
    for (cH, cV, cD) in cDs:
        cH_t = pywt.threshold(cH, sigma, mode='soft')
        cV_t = pywt.threshold(cV, sigma, mode='soft')
        cD_t = pywt.threshold(cD, sigma, mode='soft')
        new_cDs.append((cH_t, cV_t, cD_t))
    denoised = pywt.waverec2([cA] + new_cDs, wavelet)
    denoised = np.clip(denoised, 0, 255)
    return denoised.astype(np.uint8)

denoised_wavelet = wavelet_bayes(noisy_img)


denoised_nlm = cv2.fastNlMeansDenoising(noisy_img, None, h=15, templateWindowSize=7, searchWindowSize=21)


try:
    from bm3d import bm3d
    denoised_bm3d = bm3d(noisy_img.astype(np.float32)/255.0, sigma_psd=25/255.0)
    denoised_bm3d = (denoised_bm3d*255).astype(np.uint8)
except Exception as e:
    print("BM3D not available:", e)
    denoised_bm3d = np.zeros_like(img)


plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.3, bottom=0.15)  # 底部留出空间防止标号被截掉

images = [img, noisy_img]
labels = ['(a)', '(b)']

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    plt.text(0.5, 0, labels[i], transform=plt.gca().transAxes,
             ha='center', va='top', fontsize=24)

plt.tight_layout()
plt.savefig('/remote-home/yjk/dip_work3/original_vs_noisy_letters.png', dpi=300)
plt.show()


plt.figure(figsize=(12,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.1)

denoised_images = [
    denoised_gaussian,
    denoised_median,
    denoised_bilateral,
    denoised_wavelet,
    denoised_nlm,
    denoised_bm3d
]

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(denoised_images[i], cmap='gray')
    plt.axis('off')
    plt.text(0.5, 0, letters[i], transform=plt.gca().transAxes,
             ha='center', va='top', fontsize=14)

plt.tight_layout()
plt.savefig('/remote-home/yjk/dip_work3/denoised_2x3_letters.png', dpi=300)
plt.show()

print("Saved figures:\n- original_vs_noisy_letters.png\n- denoised_2x3_letters.png")
