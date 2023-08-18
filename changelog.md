# Change Log

## 0.1.0
### Changed
* --ngf option now control the EncoderDecoder models
* --ngf option now control the input channel of MLP layers in cut models (lesion_cutC, lesion_cutGB)
* --c_mlp option now control the output channel of MLP layers in cut models (lesion_cutC, lesion_cutGB)
* default value of fDown in lesion_cutC is changed to 1
* fix the issue of netF not doing anything in test_lesion_contrastive.py