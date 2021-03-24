# PaddleAudio
Unofficial  paddle audio codebase

## Install
```
git clone https://github.com/ranchlai/PaddleAudio.git
cd PaddleAudio
pip install .

```

## Usage
```
import paddleAudio as pa
s,r = pa.load(f)
mel = pa.features.mel_spect(s,r)
```
## to do 

- Add mixup loss
- add sound effects(tempo, mag, etc)
- add dataset support
- add models
- 
