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

```


---
### downloads models on demand - minimum will be ~15GB ###

---
>[!NOTE]
> if **noUnload** is not selected then models are freed after use, reloaded for each run.


---
<details>
<summary>Change log</summary>

#### 24/12/2024 ####
* first implemention. 2K models need ~16GB VRAM for VAE.

</details>


---
prompt: portrait photograph, woman with red hair, wearing green blazer over yellow tshirt and blue trousers, on sunny beach with dark clouds on horizon

![portrait photograph, woman with red hair, wearing green blazer over yellow tshirt and blue trousers, on sunny beach with dark clouds on horizon](example.png "20 steps with 1024 model")

