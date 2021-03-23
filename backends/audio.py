import numpy as np
try:
    import librosa
    has_librosa = True
except:
    has_librosa = False

try:
    import soundfile as sf
    has_snf = True
except:
    has_snf = False
    
__norm_types__ = ['linear','gaussian']
__mono_types__ = ['ch0','ch1','random','average']

from ipdb import set_trace
def audio_resample(y,src_sr, target_sr):
    if has_librosa:
        return librosa.resample(y,src_sr,target_sr)
    
    assert False, 'not implemented'
    
def audio_to_mono(y,mono_type='average'):

    if mono_type not in __mono_types__:
        assert False, 'Unsupported mono_type {}, available types are {}'.format(mono_type,__mono_types__)
        
    if y.ndim == 1:
        return y
    if y.ndim > 2:
        assert False, 'Unsupported audio array,  y.ndim > 2, the shape is {}'.format(y.shape)
    if mono_type == 'ch0':
        return y[0]
    if mono_type == 'ch1':
        return y[1]
    if mono_type == 'random':
        return y[np.random.randint(0,2)]
        
    if y.dtype=='float32':
        return (y[0]+y[1])*0.5
    if y.dtype=='int16':
        y1 = y.astype('int32')
        y1 = (y1[0]+y1[1])//2
        y1 = np.clip(y1,np.iinfo(y.dtype).min,np.iinfo(y.dtype).max).astype(y.dtype)
        return y1
    if y.dtype=='int8':
        y1 = y.astype('int16')
        y1 = (y1[0]+y1[1])//2
        y1 = np.clip(y1,np.iinfo(y.dtype).min,np.iinfo(y.dtype).max).astype(y.dtype)
        return y1
    
    assert False, 'Unsupported audio array type,  y.dtype={}'.format(y.dytpe)
        
    

    

def sound_file_load(file,offset=None,dtype='int16',duration=None):
        with sf.SoundFile(file) as sf_desc:
            sr_native = sf_desc.samplerate
            if offset:
                # Seek to the start of the target read
                sf_desc.seek(int(offset * sr_native))
            if duration is not None:
                frame_duration = int(duration * sr_native)
            else:
                frame_duration = -1

            # Load the target number of frames, and transpose to match librosa form
            y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T
       

        return y,sf_desc.samplerate

def audio_normalize(y,norm_type='linear',eps=1e-8,mul_factor=1.0): 
    if norm_type == 'linear':
        amin = np.min(y)
        amax = np.max(y)
        y = (y-amin)/(amax-amin+eps)*mul_factor
    elif norm_type == 'gaussian':
        amean = np.mean(y)
        mul_factor = max(0.01,min(mul_factor,0.2))
        astd = np.std(y)
        y = mul_factor*(y - amean)/(astd+eps)
    else:
         assert False, 'not implemented error, norm_type should be in {}'.format(__norm_types__)
            
            
    return y
        
def load(file, sr=None, 
         mono=True, 
         mono_type = 'avearge', # ch0,ch1,random,average
         normalize=True,
         norm_type='gaussian',
         norm_mul_factor = 1.0,
         offset=0.0, 
         duration=None,dtype='float32'):
    
    if has_librosa:
        y,r = librosa.load(file,sr=sr,
                           mono=False,
                           offset=offset,
                           duration=duration,
                           dtype=dtype)
    elif has_snf:
        y,r = sound_file_load(file,offset=offset,dypte=dtype,duration=duration)
        
    else: 
        assert False, 'not implemented error'
        
    if mono:
        print(mono_type)
        y = audio_to_mono(y,mono_type)
    
    if sr is not None and sr != r:
        y = resample(y,r,sr)
        r = sr
    
        
        
    if normalize:
        y = audio_normalize(y,norm_type,norm_mul_factor)
        
    return y,r
        
        
        
y,r = load(file, sr=44000, mono=True, 
         mono_type = 'aveadrge', # ch0,ch1,random,average
         normalize=True,
         norm_type='gaussian',
         offset=0.0,
         duration=None,dtype='float32')
print(r,y.shape)
from IPython.display import Audio
Audio(y,rate=r)
