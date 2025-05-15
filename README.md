## Sana for webui ##
### only tested with Forge2 ###
I don't think there is anything Forge specific here.


### works for me <sup>TM</sup> on 8GB VRAM, 16GB RAM (GTX1070) ###

---
## Install ##
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.
### needs updated *diffusers* ###

Easiest way to ensure necessary versions are installed is to edit `requirements_versions.txt` in the webUI folder.
```
diffusers>=0.32.0
accelerate>=0.26.0
```

---
### downloads models on demand - minimum will be ~9GB ###

---
>[!NOTE]
> if **noUnload** is selected then models are kept in memory; otherwise reloaded for each run. The **unload models** button removes them from memory.

---
### current UI screenshot ###
![](screenshot2.png "UI screenshot")


---
<details>
<summary>Change log</summary>

#### 15/05/2025 ####
* use v1.1 VAE
* fp8 model storage option - needs diffusers >= 0.33.0

#### 10/04/2025 ####
* add Sana1.5 models - needs diffusers >= 0.33.0

#### 07/02/2025 ####
* improved image2image / inpainting

#### 31/01/2025 ####
* switched img2img to use ForgeCanvas, if installed in Forge2. Gradio4 ImageEditor is bugged, consumes GPU or CPU constantly.

#### 17/01/2025 ####
* add option for alternative CFG calculation. '0' button toggle, +50% inference time, PAG takes priority.

#### 11/01/2025 ####
* add 4K model (unlikely to work until diffusers adds VAE tiling);
* changes to model loading so VAE only downloaded once regardless of how many models are used;
* and should not download the fp32 transformer models anymore (not sure why pipeline.from_pretrained() loading ignored the variant specified, but loading the transformer separately avoids the issue).

#### 01/01/2025 ####
* add initial sampler selection, not sure how many will work yet. *Euler* and *Heun* need more steps than *DPM++ 2M*;
* add rescale CFG, can be *very* effective.

#### 26/12/2024 ####
* fixes for gallery, sending to i2i.

#### 25/12/2024 (2) ####
* add complex human instruction toggle (CHI button), for automatic prompt enhancement;
* avoid unnecessary text encoder load if prompt hasn't changed.

#### 25/12/2024 ####
* add control of shift parameter. From initial tests doesn't seem as useful as with Flux or SD3.

#### 24/12/2024 (2) ####
* added PAG and some sort of i2i.

#### 24/12/2024 ####
* first implemention. 2K models need ~16GB VRAM for VAE.

</details>


---
example:

![](example.png "11 steps with 1024 model")

