from diffusers.utils import check_min_version
check_min_version("0.32.0")


class SanaStorage:
    ModuleReload = False
    usingGradio4 = False
    pipeTE = None
    pipeTR = None
    lastModel = None

    lastSeed = -1
    galleryIndex = 0
    lastPrompt = None
    lastNegative = None
    pos_embeds = None
    pos_attention = None
    neg_embeds = None
    neg_attention = None
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    sendAccessToken = False
    doneAccessTokenWarning = False

    locked = False     #   for preventing changes to the following volatile state while generating
    noUnload = False
    karras = False
    vpred = False
    resolutionBin = True
    sharpNoise = False
    i2iAllSteps = False

import gc
import gradio
if int(gradio.__version__[0]) == 4:
    SanaStorage.usingGradio4 = True
import math
import numpy
import os
import torch
import torchvision.transforms.functional as TF
try:
    from importlib import reload
    SanaStorage.ModuleReload = True
except:
    SanaStorage.ModuleReload = False

##   from webui
from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste

##   diffusers / transformers necessary imports
from diffusers import SanaPipeline as pipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN

from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging

##  for Florence-2, including workaround for unnecessary flash_attn requirement
from transformers import AutoProcessor, AutoModelForCausalLM 
##  for SuperPrompt
from transformers import T5TokenizerFast, T5ForConditionalGeneration

##   my extras
import customStylesListSana as styles


# modules/processing.py - don't use ',', '\n', ':' in values
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, steps, seed, width, height):
    karras = " : Karras" if SanaStorage.karras == True else ""
    vpred = " : V-Prediction" if SanaStorage.vpred == True else ""
    isDMD = "PixArt-Alpha-DMD" in model
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Steps": steps if not isDMD else None,
        "CFG": f"{guidance_scale}",
    }
#add i2i marker?
    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = "Parameters: " + ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    noise_text = f"\nInitial noise: {SanaStorage.noiseRGBA}" if SanaStorage.noiseRGBA[3] != 0.0 else ""

    return f"Model: {model}\n{prompt_text}{generation_params_text}{noise_text}"

def predict(positive_prompt, negative_prompt, model, width, height, guidance_scale, num_steps, sampling_seed, num_images, i2iSource, i2iDenoise, style, *args):
 
    logging.set_verbosity(logging.ERROR)        #   diffusers and transformers both enjoy spamming the console with useless info
 
    access_token = 0
    if SanaStorage.sendAccessToken == True:
        try:
            with open('huggingface_access_token.txt', 'r') as file:
                access_token = file.read().strip()
        except:
            if SanaStorage.doneAccessTokenWarning == False:
                print ("Sana: couldn't load 'huggingface_access_token.txt' from the webui directory. Will not be able to download/update gated models. Local cache will work.")
                SanaStorage.doneAccessTokenWarning = True

    torch.set_grad_enabled(False)


    dtype = torch.bfloat16 if "BF16" in model else torch.float16
    variant = "bf16" if "BF16" in model else "fp16"
    
    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = negative_prompt + styles.styles_list[style][2]
        
    ####    check img2img
    if i2iSource == None:
        i2iDenoise = 1
    if SanaStorage.i2iAllSteps == True:
        num_steps = int(num_steps / i2iDenoise)
            ####    end check img2img
 
    ####    enforce safe generation size
    if SanaStorage.resolutionBin == False:
        width  = (width  // 32) * 32
        height = (height // 32) * 32
    ####    end enforce safe generation size


    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    SanaStorage.lastSeed = fixed_seed


    ####    setup pipe for text encoding
    if SanaStorage.pipeTE == None:
        SanaStorage.pipeTE = pipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            transformer=None,
            vae=None,
            variant="bf16",
            torch_dtype=torch.bfloat16
        )

    useCachedEmbeds = (SanaStorage.lastPrompt == positive_prompt and SanaStorage.lastNegative == negative_prompt)
    if useCachedEmbeds:
        print ("Sana: Skipping tokenizer, text_encoder.")
    else:
        print ("Sana: encoding prompt ...", end="\r", flush=True)

        SanaStorage.pipeTE.to('cuda')
        SanaStorage.lastPrompt = positive_prompt
        SanaStorage.lastNegative = negative_prompt

        pos_embeds, pos_attention, neg_embeds, neg_attention = SanaStorage.pipeTE.encode_prompt(positive_prompt, negative_prompt=negative_prompt)

        print ("Sana: encoding prompt ... done")
        SanaStorage.pos_embeds    = pos_embeds.to('cuda').to(dtype)
        SanaStorage.neg_embeds    = neg_embeds.to('cuda').to(dtype)
        SanaStorage.pos_attention = pos_attention.to('cuda').to(dtype)
        SanaStorage.neg_attention = neg_attention.to('cuda').to(dtype)

        del pos_embeds, neg_embeds, pos_attention, neg_attention

        if SanaStorage.noUnload == False:
            SanaStorage.pipeTE.to('cpu')
        else:
            SanaStorage.pipeTE = None
    ####    end text encoding

    if SanaStorage.lastModel != model:
        SanaStorage.pipeTR = None
    
    ####    setup pipe for transformer + VAE
    if SanaStorage.pipeTR == None:
        SanaStorage.pipeTR = pipeline.from_pretrained(
            model,
#            tokenizer=None,
            text_encoder=None,
            variant=variant,
            torch_dtype=dtype
        )
        SanaStorage.lastModel = model
        SanaStorage.pipeTR.enable_model_cpu_offload()
#        SanaStorage.pipeTR.vae.enable_slicing() #ineffective?

    ####    end setup pipe for transformer + VAE

#seperate te and transformer?


    # gc.collect()
    # torch.cuda.empty_cache()



#   lora test - diffusers type, PEFT
#   seems to work, but base model better
#   possibly: model.load_adapter(loraRepo)
#             model.unload()                #   but are these in PixArtTransformer?
##    if isSigma and not is2Stage:
##        try:
##            loraRepo = './/models//diffusers//PixArtLora//pocketCreatures1024'
##            SanaStorage.pipeTR.transformer = PeftModel.from_pretrained(
##                SanaStorage.pipeTR.transformer,
##                loraRepo
##            )
##        except:
##            pass



    #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)

    if SanaStorage.resolutionBin:
        match SanaStorage.pipeTR.transformer.config.sample_size:
            case 64:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            case 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            case 16:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            case _:
                raise ValueError("Invalid sample size")

        ar = float(height / width)
        closest_ratio = min(aspect_ratio_bin.keys(), key=lambda ratio: abs(float(ratio) - ar))
        theight = int(aspect_ratio_bin[closest_ratio][0])
        twidth  = int(aspect_ratio_bin[closest_ratio][1])
    else:
        theight = height
        twidth  = width

    shape = (
        num_images,
        SanaStorage.pipeTR.transformer.config.in_channels,
        int(theight) // SanaStorage.pipeTR.vae_scale_factor,
        int(twidth) // SanaStorage.pipeTR.vae_scale_factor,
    )

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    latents = randn_tensor(shape, generator=generator).to('cuda')
    
    if SanaStorage.sharpNoise:
        minDim = 1 + (min(latents.size(2), latents.size(3)) // 2)
        for b in range(len(latents)):
            blurred = TF.gaussian_blur(latents[b], minDim)
            latents[b] = 1.02*latents[b] - 0.02*blurred
    
    
    #regen the generator to minimise differences between single/batch - might still be different - batch processing could use different pytorch kernels
    del generator
    generator = torch.Generator(device='cpu').manual_seed(14641)

    #   colour the initial noise
    if SanaStorage.noiseRGBA[3] != 0.0:
        nr = SanaStorage.noiseRGBA[0] ** 0.5
        ng = SanaStorage.noiseRGBA[1] ** 0.5
        nb = SanaStorage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(numpy.full((32,32), (nr), dtype=numpy.float32))
        imageG = torch.tensor(numpy.full((32,32), (ng), dtype=numpy.float32))
        imageB = torch.tensor(numpy.full((32,32), (nb), dtype=numpy.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)
        
        SanaStorage.pipeTR.vae.to(torch.float32)

        image = SanaStorage.pipeTR.image_processor.preprocess(image).to('cuda')
        image_latents = SanaStorage.pipeTR.vae.encode(image)[0]
        image_latents *= SanaStorage.pipeTR.vae.config.scaling_factor * SanaStorage.pipeTR.scheduler.init_noise_sigma
        image_latents = image_latents.to(latents.dtype)
        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

#        latents += image_latents * SanaStorage.noiseRGBA[3]
        torch.lerp (latents, image_latents, SanaStorage.noiseRGBA[3], out=latents)

        # NoiseScheduler = SanaStorage.pipeTR.scheduler
        # ts = torch.tensor([int(1000 * (1.0-(0.1*SanaStorage.noiseRGBA[3]))) - 1], device='cpu')
        # ts = ts[:1].repeat(num_images)
        # latents = NoiseScheduler.add_noise(image_latents, latents, ts)

        del imageR, imageG, imageB, image, image_latents#, NoiseScheduler
    #   end: colour the initial noise

    timesteps = None

#    if useCustomTimeSteps:
#    timesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]    #   AYS sdXL
    #loglin interpolate to number of steps

    SanaStorage.pipeTR.transformer.to('cuda')
    SanaStorage.pipeTR.vae.to('cpu')
    torch.cuda.empty_cache()

    with torch.inference_mode():
        output = SanaStorage.pipeTR(
            prompt                          = None,#positive,
            negative_prompt                 = None,#negative,
            generator                       = generator,
            latents                         = latents.to(dtype),   #   initial noise, possibly with colour biasing

            # image                           = i2iSource,
            # strength                        = i2iDenoise,

            num_inference_steps             = num_steps,
            num_images_per_prompt           = num_images,
            height                          = height,
            width                           = width,
            guidance_scale                  = guidance_scale,
            prompt_embeds                   = SanaStorage.pos_embeds,
            negative_prompt_embeds          = SanaStorage.neg_embeds,
            prompt_attention_mask           = SanaStorage.pos_attention,
            negative_prompt_attention_mask  = SanaStorage.neg_attention,
            use_resolution_binning          = SanaStorage.resolutionBin,
            
            output_type                     = "latent",
        ).images

    if SanaStorage.noUnload:
        SanaStorage.pipeTR.transformer.to('cpu')
        torch.cuda.empty_cache()
    else:
        SanaStorage.pipeTR.transformer = None
        SanaStorage.lastModel = None

    del generator, latents

    SanaStorage.pipeTR.vae.to('cuda')

    results = []
    total = len(output)
    for i in range (total):
        print (f'Sana: VAE: {i+1} of {total}', end='\r', flush=True)
        latent = output[i:i+1].to(SanaStorage.pipeTR.vae.dtype)
        image = SanaStorage.pipeTR.vae.decode(latent / SanaStorage.pipeTR.vae.config.scaling_factor, return_dict=False)[0]
        if SanaStorage.resolutionBin:
            image = SanaStorage.pipeTR.image_processor.resize_and_crop_tensor(image, width, height)
        image = SanaStorage.pipeTR.image_processor.postprocess(image, output_type="pil")[0]

        info=create_infotext(
            model,
            positive_prompt, negative_prompt,
            guidance_scale, 
            num_steps, 
            fixed_seed + i, 
            width, height)

        results.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed + i,
            positive_prompt,
            opts.samples_format,
            info
        )
    print ('Sana: VAE: done  ')

    del output

    if SanaStorage.noUnload:
        SanaStorage.pipeTR.vae.to('cpu')
    else:
        SanaStorage.pipeTR = None
        SanaStorage.lastModel = None

    gc.collect()
    torch.cuda.empty_cache()

    SanaStorage.locked = False
    return gradio.Button.update(value='Generate', variant='primary', interactive=True), gradio.Button.update(interactive=True), results


def on_ui_tabs():
    if SanaStorage.ModuleReload:
        reload(styles)
    
    models_list = ["Efficient-Large-Model/Sana_600M_512px_diffusers", "Efficient-Large-Model/Sana_600M_1024px_diffusers", "Efficient-Large-Model/Sana_1600M_512px_diffusers", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers"]
    defaultModel = models_list[3]
    defaultWidth = 1024
    defaultHeight = 1024
    
    def getGalleryIndex (evt: gradio.SelectData, gallery):
        SanaStorage.galleryIndex = evt.index
        return gallery[SanaStorage.galleryIndex][1]

    def reuseLastSeed ():
        return SanaStorage.lastSeed + SanaStorage.galleryIndex
        
    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = image.size[0]
            h = image.size[1]
        return [w, h]

    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', 
                                                         attn_implementation="sdpa", 
                                                         torch_dtype=torch.float16, 
                                                         trust_remote_code=True).to('cuda')
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
            inputs.to('cuda').to(torch.float16)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            del inputs
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            del generated_ids
            parsed_answer = processor.post_process_generation(generated_text, task=p, image_size=(image.width, image.height))
            del generated_text
            print (parsed_answer)
            result += parsed_answer[p]
            del parsed_answer
            if p != prompts[-1]:
                result += ' | \n'

        del model, processor

        if SanaStorage.captionToPrompt:
            return result
        else:
            return originalPrompt

    def i2iImageFromGallery (gallery):
        try:
            if SanaStorage.usingGradio4:
                newImage = gallery[SanaStorage.galleryIndex][0]
                return newImage
            else:
                newImage = gallery[SanaStorage.galleryIndex][0]['name'].rsplit('?', 1)[0]
                return newImage
        except:
            return None

    def toggleC2P ():
        SanaStorage.captionToPrompt ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SanaStorage.captionToPrompt])
    def toggleAccess ():
        SanaStorage.sendAccessToken ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SanaStorage.sendAccessToken])

    #   these are volatile state, should not be changed during generation
    def toggleNU ():
        if not SanaStorage.locked:
            SanaStorage.noUnload ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SanaStorage.noUnload])
    def unloadM ():
        if not SanaStorage.locked:
            SanaStorage.pipeTR = None
            SanaStorage.lastModel = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')
    def toggleKarras ():
        if not SanaStorage.locked:
            SanaStorage.karras ^= True
        return gradio.Button.update(variant='primary' if SanaStorage.karras == True else 'secondary',
                                value='\U0001D40A' if SanaStorage.karras == True else '\U0001D542')
    def toggleResBin ():
        if not SanaStorage.locked:
            SanaStorage.resolutionBin ^= True
        return gradio.Button.update(variant='primary' if SanaStorage.resolutionBin == True else 'secondary',
                                value='\U0001D401' if SanaStorage.resolutionBin == True else '\U0001D539')
#    def toggleVP ():
#        if not SanaStorage.locked:
#           SanaStorage.vpred ^= True
#        return gradio.Button.update(variant='primary' if SanaStorage.vpred == True else 'secondary',
#                                value='\U0001D415' if SanaStorage.vpred == True else '\U0001D54D')
    def toggleAS ():
        if not SanaStorage.locked:
            SanaStorage.i2iAllSteps ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SanaStorage.i2iAllSteps])


    def toggleSP ():
        if not SanaStorage.locked:
            return gradio.Button.update(variant='primary')
    def superPrompt (prompt, seed):
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')

        torch.manual_seed(get_fixed_seed(seed))
        device = superprompt.device
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(device)
        outputs = superprompt.generate(input_ids, max_new_tokens=256, repetition_penalty=1.2, do_sample=True)
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        return gradio.Button.update(variant='secondary'), result



    def toggleGenerate (R, G, B, A):
        SanaStorage.noiseRGBA = [R, G, B, A]
        SanaStorage.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)

    schedulerList = ["default", "DDPM", "DEIS", "DPM++ 2M", "DPM++ 2M SDE", "DPM", "DPM SDE",
                     "Euler", "Euler A", "LCM", "SA-solver", "UniPC", ]

    def parsePrompt (positive, negative, width, height, seed, steps, cfg, nr, ng, nb, ns):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Initial noise: " == str(p[l][0:15]):
                noiseRGBA = str(p[l][16:-1]).split(',')
                nr = float(noiseRGBA[0])
                ng = float(noiseRGBA[1])
                nb = float(noiseRGBA[2])
                ns = float(noiseRGBA[3])
            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = 16 * ((int(size[0]) + 8) // 16)
                            height = 16 * ((int(size[1]) + 8) // 16)
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Sampler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Scheduler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                cfg = float(pairs[2])
                        case "CFG:":
                            cfg = float(pairs[1])
                        case "width:":
                            width = 16 * ((int(pairs[1]) + 8) // 16)
                        case "height:":
                            height = 16 * ((int(pairs[1]) + 8) // 16)
        return positive, negative, width, height, seed, steps, cfg, nr, ng, nb, ns

    resolutionList512 = [
        (1024, 256),    (864, 288),     (704, 352),     (640, 384),     (608, 416),
        (512, 512), 
        (416, 608),     (384, 640),     (352, 704),     (288, 864),     (256, 1024)
    ]
    resolutionList1024 = [
        (2048, 512),    (1728, 576),    (1408, 704),    (1280, 768),    (1216, 832),
        (1024, 1024),
        (832, 1216),    (768, 1280),    (704, 1408),    (576, 1728),    (512, 2048)
    ]
    resolutionList2048 = [
        (4096, 1024),   (3456, 1152),   (2816, 1408),   (2560, 1536),   (2432, 1664),
        (2048, 2048),
        (1664, 2432),   (1536, 2560),   (1408, 2816),   (1152, 3456),   (1024, 4096)
    ]


    def updateWH (dims, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        #   passing by value because of odd gradio bug? when using index can either update displayed list correctly, or get values correctly, not both
        wh = dims.split('\u00D7')
        return None, int(wh[0]), int(wh[1])

    def processCN (image, method):
        if image:
            if method == 1:     # generate HED edge
                try:
                    from controlnet_aux import HEDdetector
                    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
                    hed_edge = hed(image)
                    return hed_edge
                except:
                    print ("Need controlAux package to preprocess.")
                    return image
        return image
    def toggleSharp ():
        SanaStorage.sharpNoise ^= True
        return gradio.Button.update(value=['s', 'S'][SanaStorage.sharpNoise],
                                variant=['secondary', 'primary'][SanaStorage.sharpNoise])


    with gradio.Blocks() as sana_block:
        with ResizeHandleRow():
            with gradio.Column():
                with gradio.Row():
                    access = ToolButton(value='\U0001F917', variant='secondary')
                    model = gradio.Dropdown(models_list, label='Model', value=defaultModel, type='value', scale=2)
#                    scheduler = gradio.Dropdown(schedulerList, label='Sampler', value="UniPC", type='value', scale=1)
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
#                    vpred = ToolButton(value="\U0001D54D", variant='secondary', tooltip="use v-prediction")

                with gradio.Row():
                    positive_prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here...', lines=2, show_label=False)
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='zero out negative embeds')

                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', placeholder='Negative prompt', lines=1.01, show_label=False)
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value="[style] (None)", type='index', scale=0, show_label=False)
                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=128, maximum=4096, step=32, value=defaultWidth, elem_id="PixArtSigma_width")
                    swapper = ToolButton(value="\U000021C4")
                    height = gradio.Slider(label='Height', minimum=128, maximum=4096, step=32, value=defaultHeight, elem_id="PixArtSigma_height")
                    resBin = ToolButton(value="\U0001D401", variant='primary', tooltip="use resolution binning")
                    dims = gradio.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList1024],
                                        label='Quickset', type='value', scale=0)

                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=8, step=0.1, value=4.0, scale=1, visible=True)
                    steps = gradio.Slider(label='Steps', minimum=1, maximum=60, step=1, value=20, scale=2, visible=True)
                with gradio.Row():
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gradio.Accordion(label='the colour of noise', open=False):
                    with gradio.Row():
                        initialNoiseR = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gradio.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')
                        sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gradio.Accordion(label='image to image', open=False, visible=False):
                    with gradio.Row():
                        i2iSource = gradio.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                    with gradio.Row():
                        with gradio.Column():
                            with gradio.Row():
                                i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                AS = ToolButton(value='AS')
                            with gradio.Row():
                                i2iFromGallery = gradio.Button(value='Get gallery image')
                                i2iSetWH = gradio.Button(value='Set size from image')
                            with gradio.Row():
                                i2iCaption = gradio.Button(value='Caption image (Florence-2)', scale=6)
                                toPrompt = ToolButton(value='P', variant='secondary')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if SanaStorage.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)

                ctrls = [positive_prompt, negative_prompt, model, width, height, guidance_scale, steps, sampling_seed, batch_size, i2iSource, i2iDenoise, style,]
                
                parseCtrls = [positive_prompt, negative_prompt, width, height, sampling_seed, steps, guidance_scale, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gradio.Gallery(label='Output', height="80vh", type='pil', interactive=False, 
                                            show_label=False, object_fit='contain', visible=True, columns=1, preview=True)
                image_infotext = gradio.Textbox(visible=False)

#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=image_infotext,#positive_prompt,
                        source_image_component=output_gallery,
                    ))

        def set_dims(model):
            if "512" in model:
                resList = resolutionList512
            elif "2K" in model or "2048" in model:
                resList = resolutionList2048
            else:
                resList = resolutionList1024

            choices = [f'{i} \u00D7 {j}' for i,j in resList]
            return gradio.update(choices=choices)

        model.change(
            fn=set_dims,
            inputs=model,
            outputs=dims,
            show_progress=False
        )


#        vpred.click(toggleVP, inputs=[], outputs=vpred)
        noUnload.click(toggleNU, inputs=[], outputs=noUnload)
        unloadModels.click(unloadM, inputs=[], outputs=[], show_progress=True)

        SP.click(toggleSP, inputs=[], outputs=SP)
        SP.click(superPrompt, inputs=[positive_prompt, sampling_seed], outputs=[SP, positive_prompt])
        sharpNoise.click(toggleSharp, inputs=[], outputs=sharpNoise)
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        parse.click(parsePrompt, inputs=parseCtrls, outputs=parseCtrls, show_progress=False)
        access.click(toggleAccess, inputs=[], outputs=access)
        karras.click(toggleKarras, inputs=[], outputs=karras)
        resBin.click(toggleResBin, inputs=[], outputs=resBin)
        swapper.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
        random.click(lambda : -1, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        AS.click(toggleAS, inputs=[], outputs=AS)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])
        i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])
        toPrompt.click(toggleC2P, inputs=[], outputs=[toPrompt])

        output_gallery.select (fn=getGalleryIndex, inputs=[output_gallery], outputs=[image_infotext])

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, SP, output_gallery]).then(fn=lambda: gradio.update(value='Generate', variant='primary', interactive=True), inputs=None, outputs=generate_button)
        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA], outputs=[generate_button, SP])

    return [(sana_block, "Sana", "sana_DoE")]

script_callbacks.on_ui_tabs(on_ui_tabs)

