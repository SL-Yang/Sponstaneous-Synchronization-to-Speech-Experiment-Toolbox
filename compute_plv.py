from scipy.signal import lfilter,filtfilt,hilbert,resample,detrend,butter,sosfilt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile, loadmat
from scipy.stats import norm
from matplotlib import pyplot as plt
import numpy as np
import math

def plot_result(subj,signal,sig_env,fft,afft,plv,tplv,Fs,Fs_new,band,save_fig=False,save_path='',block=1):

	fig,ax = plt.subplots(3,1,figsize=(10,9)) #generate  subplots

	#plot signal with envelope, grey:signal, red:envelope
	signal_time = [i / Fs for i in range(len(signal))]
	env_time = [i / Fs_new for i in range(len(sig_env))]
	ax[0].plot(signal_time,signal,'grey',label='signal')
	ax[0].plot(env_time,sig_env,'r',label='envelope')
	ax[0].set_title(f'{subj} Speech')
	ax[0].legend()

	#plot spectrum
	df = f[2] - f[1]
	n_ave = int(round(0.2 / df))
	cof = np.ones((1,n_ave)) / n_ave
	cof = np.squeeze(cof)
	smooth_fft = lfilter(cof,1,afft)
	smooth_fft = smooth_fft / smooth_fft.max()
	ax[1].plot(fft,smooth_fft,'black',label='spectrum')

	#find the max freq within bandpass
	ind = np.where((fft > band[0]) & (fft < band[1]))
	temp = smooth_fft[ind].max()
	ind = np.squeeze(np.where(smooth_fft == temp))
	
	ax[1].set_title(f'Pico = {round(fft[ind],3)}')
	ax[1].legend()

	#plot plv
	ax[2].plot(tplv,plv,'b',label='plv')
	ax[2].set_title(f'PLV = {round(plv.mean(),3)}')
	ax[2].legend()

	plt.tight_layout()

	if save_fig:
		fig.savefig(f'{save_path}{subj}.jpg',dpi=360)
	else:

		pass

	plt.show()


def freqfiltbp(env,band,Fs,numdev,dim):
	# get frequency axis
	f = np.array([i for i in range(len(env))])
	f = f / len(f)
	f = f * Fs

	#build frequency filter
	freqfilt = norm(band.mean(),np.diff(band)/(2*numdev)).pdf(f[:int(len(f)/2)])
	freqfilt = np.concatenate((freqfilt,np.flip(freqfilt)))

	#apply filter to freq data
	fqdata = fft(env)
	fqfiltdata = fqdata * freqfilt

	#convert back to time domain
	output = ifft(fqfiltdata)

	return output.real

def compute_plv(phase_dif):

	T = phase_dif.shape[0]
	
	sums = []
	for t in range(T):

		sums.append(np.exp(1j*(phase_dif[t])))

	sums = np.array(sums)

	return abs(sums.sum()) / T

# Set up variables	
fs_new = 100 
bandpass = np.array([3.5,5.5]) 

file_name ='test2'
time_limit = 30

# Load the produced speech audio file
fs, Ps = wavfile.read(f'results/{file_name}.wav') 
Ps = Ps / 32767 # change from Int16 to Float64
Ps = Ps[:time_limit*fs] 
Ps = Ps - Ps.mean() 
Ps_as = hilbert(Ps)
Ps_env = np.abs(Ps_as)

# Smooths and shows the envelope with the original signal
n_average = 0.01 * fs
coeff= np.ones((1, int(n_average))) / n_average
coeff = np.squeeze(coeff)
Ps_filt = filtfilt(coeff,1,Ps_env)
Ps_rsp = resample(Ps_filt,int(len(Ps) / (fs / fs_new)))
Ps_dtd = detrend(Ps_rsp)

# Compute the spectral content of the envelope
Ps_dtd = Ps_dtd - Ps_dtd.mean()
L = Ps_dtd.shape[0]
NFFT = 2 ** math.ceil(math.log(abs(L),2))
Y = fft(Ps_dtd,n = NFFT) / L
f = 100 / 2 * np.linspace(0,1,int(NFFT / 2) + 1)

# low pass filter at 10hz
f = np.array([i for i in f.clip(max = 10) if i < 10])
Y = Y[:f.shape[0]]
amp_fft = abs(Y) ** 2

# load audio
As = loadmat('sounds/main.mat')
As = np.squeeze(As['envelope'])
As = As[:time_limit*fs_new]
As = detrend(As)

# apply gaussian filter
As_ap = freqfiltbp(As,bandpass,fs_new,1,2)
Ps_ap = freqfiltbp(Ps_dtd,bandpass,fs_new,1,2)

# hilbert transforamtion, get phase
temp_1 = hilbert(As_ap)
temp_2 = hilbert(Ps_ap)
pin_1 = np.angle(temp_1)
pin_2 = np.angle(temp_2)
phase_diff = pin_1 - pin_2

# Compute PLV
T,shift,i = 5,2,1
PLVs,time_plv = [],[]

for i in range(0,time_limit,2):
	plv = compute_plv(phase_diff[i*fs_new:(i+T)*fs_new])
	PLVs.append(plv)
	time_plv.append((2*i+T)/2)

PLVs = np.array(PLVs)

# Plot all results
plot_result(file_name,Ps,Ps_dtd,f,amp_fft,PLVs,time_plv,
	fs,fs_new,bandpass,save_fig=True,save_path='results/')











