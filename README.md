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

- Add mixup loss [yqlai]
- add sound effects(tempo, mag, etc) , sox supports[yqlai]
- add dataset support [xiaojie】
- add models 【audioset sound tagging[yqlai], DCASE classication[xiaojie], ASD[yqlai]，sound classification[xiaojie]】
- add demos (audio,video demos) [xiaojie]
