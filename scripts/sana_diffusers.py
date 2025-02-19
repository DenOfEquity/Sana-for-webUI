from diffusers.utils import check_min_version
check_min_version("0.32.0")


class SanaStorage:
    ModuleReload = False
    forgeCanvas = False
    usingGradio4 = False
    pipeTE = None
    pipeTR = None
    lastModel = None

    lora = None
    lora_scale = 1.0
    loadedLora = False

    lastPrompt = None
    lastNegative = None
    pos_embeds = None
    pos_attention = None
    neg_embeds = None
    neg_attention = None
    nul_embeds = None
    nul_attention = None
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    sendAccessToken = False
    doneAccessTokenWarning = False
    randomSeed = True

    locked = False     #   for preventing changes to the following volatile state while generating
    noUnload = False
    biasCFG = False
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

try:
    from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
    SanaStorage.forgeCanvas = True
except:
    SanaStorage.forgeCanvas = False
    canvas_head = ""

from PIL import Image, ImageFilter

##   from webui
from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste

##   diffusers / transformers necessary imports
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from scripts.sana_pipeline import ASPECT_RATIO_4096_BIN

from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler#, SASolverScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging

##  for Florence-2
from transformers import AutoProcessor, AutoModelForCausalLM 
##  for SuperPrompt
from transformers import T5TokenizerFast, T5ForConditionalGeneration

##   my extras
import customStylesListSana as styles
import scripts.sana_pipeline as pipeline


# modules/processing.py - don't use ',', '\n', ':' in values
def create_infotext(model, sampler, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, steps, seed, width, height, PAG_scale, PAG_adapt, shift):
    bCFG = ", biasCorrectionCFG: enabled" if SanaStorage.biasCFG == True else ""
    generation_params = {
        "Steps": steps,
        "CFG scale": f"{guidance_scale}",
        "CFG rescale": f"{guidance_rescale}",
        "Seed": seed,
        "Size": f"{width}x{height}",
        "PAG": f"{PAG_scale} ({PAG_adapt})",
        "Shift": f"{shift}",
        "Sampler": f"{sampler}",
    }
#add i2i marker?
    prompt_text = f"{positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative prompt: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    noise_text = f", Initial noise: {SanaStorage.noiseRGBA}" if SanaStorage.noiseRGBA[3] != 0.0 else ""

    return f"{prompt_text}{generation_params_text}{noise_text}{bCFG}, Model (Sana): {model}"

def predict(positive_prompt, negative_prompt, model, sampler, width, height, guidance_scale, guidance_rescale, num_steps, sampling_seed, num_images, i2iSource, i2iDenoise, style, PAG_scale, PAG_adapt, maskType, maskSource, maskBlur, maskCutOff, shift, *args):
 
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

    if PAG_scale > 0.0:
        guidance_rescale = 0.0

    ####    check img2img
    if i2iSource == None:
        maskType = 0
        i2iDenoise = 1
    
    if maskSource == None:
        maskType = 0

    match maskType:
        case 0:     #   'none'
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0
        case 1:
            if SanaStorage.forgeCanvas: #  'inpaint mask'
                maskSource = maskSource.getchannel('A').convert('L')#.convert("RGB")#.getchannel('R').convert('L')
            else:                       #   'drawn'
                maskSource = maskSource['layers'][0]  if SanaStorage.usingGradio4 else maskSource['mask']
        case 2:
            if SanaStorage.forgeCanvas: #   sketch
                i2iSource = Image.alpha_composite(i2iSource, maskSource)
                maskSource = None
                maskBlur = 0
                maskCutOff = 1.0
            else:                       #   'image'
                maskSource = maskSource['background'] if SanaStorage.usingGradio4 else maskSource['image']
        case 3:
            if SanaStorage.forgeCanvas: #   inpaint sketch
                i2iSource = Image.alpha_composite(i2iSource, maskSource)
                mask = maskSource.getchannel('A').convert('L')
                short_side = min(mask.size)
                dilation_size = int(0.015 * short_side) * 2 + 1
                mask = mask.filter(ImageFilter.MaxFilter(dilation_size))
                maskSource = mask.point(lambda v: 255 if v > 0 else 0)
                maskCutoff = 0.0
            else:                       #   'composite'
                maskSource = maskSource['composite']  if SanaStorage.usingGradio4 else maskSource['image']
        case _:
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0

    if i2iSource:
        if SanaStorage.i2iAllSteps == True:
            num_steps = int(num_steps / i2iDenoise)

        if SanaStorage.forgeCanvas:
            i2iSource = i2iSource.convert('RGB')


    if maskBlur > 0:
        dilation_size = maskBlur * 2 + 1
        maskSource = TF.gaussian_blur(maskSource.filter(ImageFilter.MaxFilter(dilation_size)), dilation_size)
    ####    end check img2img
 
    ####    enforce safe generation size
    if SanaStorage.resolutionBin == False:
        width  = (width  // 32) * 32
        height = (height // 32) * 32
    ####    end enforce safe generation size

    fixed_seed = get_fixed_seed(-1 if SanaStorage.randomSeed else sampling_seed)

    ####    text encoding
    calcEmbeds = (SanaStorage.lastPrompt   != positive_prompt) or \
                 (SanaStorage.lastNegative != negative_prompt) or \
                 (SanaStorage.pos_embeds is None) or \
                 (SanaStorage.neg_embeds is None) or \
                 (SanaStorage.nul_embeds is None)
    if calcEmbeds:
        ####    setup pipe for text encoding
        if SanaStorage.pipeTE == None:
            SanaStorage.pipeTE = pipeline.Sana_Pipeline_DoE.from_pretrained(
                "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
                transformer=None,
                vae=None,
                scheduler=None,
                variant="bf16",
                torch_dtype=torch.bfloat16
            )

        SanaStorage.pipeTE.to('cuda')

        print ("Sana: encoding prompt ...", end="\r", flush=True)
        if SanaStorage.lastPrompt != positive_prompt or SanaStorage.pos_embeds is None:
            pos_embeds, pos_attention = SanaStorage.pipeTE.encode_prompt(
                positive_prompt,
            )
            SanaStorage.pos_embeds    = pos_embeds.to('cuda')
            SanaStorage.pos_attention = pos_attention.to('cuda')
            del pos_embeds, pos_attention
            SanaStorage.lastPrompt = positive_prompt

        if SanaStorage.lastNegative != negative_prompt or SanaStorage.neg_embeds is None:
            neg_embeds, neg_attention = SanaStorage.pipeTE.encode_prompt(
                negative_prompt,
            )
            SanaStorage.neg_embeds    = neg_embeds.to('cuda')
            SanaStorage.neg_attention = neg_attention.to('cuda')
            del neg_embeds, neg_attention
            SanaStorage.lastNegative = negative_prompt

        if SanaStorage.nul_embeds is None:
            nul_embeds, nul_attention = SanaStorage.pipeTE.encode_prompt(
                "",
            )
            SanaStorage.nul_embeds    = nul_embeds.to('cuda')
            SanaStorage.nul_attention = nul_attention.to('cuda')
            del nul_embeds, nul_attention

        print ("Sana: encoding prompt ... done")
    else:
        print ("Sana: Skipping tokenizer, text_encoder.")

    if SanaStorage.noUnload:
        if SanaStorage.pipeTE is not None:
            SanaStorage.pipeTE.to('cpu')
    else:
        SanaStorage.pipeTE = None
        gc.collect()
        torch.cuda.empty_cache()
    ####    end text encoding



    ####    setup pipe for transformer + VAE
    ### #   shared VAE, save ~1.16GB per model (after 1600M_1024px) - they are identical for fp16 and bf16 models
    if SanaStorage.lastModel != model:
        SanaStorage.pipeTR = None
        gc.collect()
        torch.cuda.empty_cache()

    if SanaStorage.pipeTR == None:
        from diffusers.models import AutoencoderDC, SanaTransformer2DModel

        SanaStorage.pipeTR = pipeline.Sana_Pipeline_DoE.from_pretrained(
            model,
            tokenizer=None,
            text_encoder=None,
            vae=AutoencoderDC.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", subfolder="vae", variant="bf16"),
            transformer=SanaTransformer2DModel.from_pretrained(model, subfolder="transformer", variant=variant),
            variant=variant,
            torch_dtype=dtype,
            pag_applied_layers=["transformer_blocks.8"],
        )
        SanaStorage.lastModel = model
        SanaStorage.pipeTR.enable_model_cpu_offload()

        if "2Kpx" in model or "4Kpx" in model:
            SanaStorage.pipeTR.vae.enable_tiling()
    ####    end setup pipe for transformer + VAE

    ####    shift, possibly other scheduler options, different sigmas?
    #flow, karras, exponential, beta
    schedulerConfig = dict(SanaStorage.pipeTR.scheduler.config)
    schedulerConfig['flow_shift'] = shift

    if sampler == "Euler":
        schedulerConfig.pop('algorithm_type', None) 
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(schedulerConfig)
    elif sampler == "Heun":
        schedulerConfig.pop('algorithm_type', None) 
        scheduler = FlowMatchHeunDiscreteScheduler.from_config(schedulerConfig)
    # elif sampler == "SA-solver":
        # schedulerConfig['algorithm_type'] = 'data_prediction'
        # scheduler = SASolverScheduler.from_config(schedulerConfig)
    else:
        scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)

    SanaStorage.pipeTR.scheduler = scheduler

#   load in LoRA
    if SanaStorage.lora and SanaStorage.lora != "(None)" and SanaStorage.lora_scale != 0.0:
        lorapath = ".//models/diffusers//SanaLora//"
        loraname = SanaStorage.lora + ".safetensors"
        try:
            SanaStorage.pipeTR.load_lora_weights(lorapath, weight_name=loraname, local_files_only=True, adapter_name="lora")
            SanaStorage.loadedLora = True
            scales = {
                "unet": SanaStorage.lora_scale
            }
            SanaStorage.pipeTR.set_adapters("lora", scales)
#            pipe.set_adapters(SanaStorage.lora, adapter_weights=SanaStorage.lora_scale)
        except:
            print ("Sana: failed LoRA: " + lorafile)
            #   no reason to abort, just carry on without LoRA


    #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)
    if SanaStorage.resolutionBin:
        match SanaStorage.pipeTR.transformer.config.sample_size:
            case 128:
                aspect_ratio_bin = ASPECT_RATIO_4096_BIN
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
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        output = SanaStorage.pipeTR(
            prompt                          = None,#positive,
            negative_prompt                 = None,#negative,
            generator                       = generator,
            latents                         = latents.to(dtype),   #   initial noise, possibly with colour biasing

            image                           = i2iSource,
            mask_image                      = maskSource,
            strength                        = i2iDenoise,
            mask_cutoff                     = maskCutOff,

            num_inference_steps             = num_steps,
            num_images_per_prompt           = num_images,
            height                          = height,
            width                           = width,
            guidance_scale                  = guidance_scale,
            guidance_rescale                = guidance_rescale,
            prompt_embeds                   = SanaStorage.pos_embeds,
            negative_prompt_embeds          = SanaStorage.neg_embeds,
            nul_prompt_embeds               = SanaStorage.nul_embeds,
            prompt_attention_mask           = SanaStorage.pos_attention,
            negative_prompt_attention_mask  = SanaStorage.neg_attention,
            nul_prompt_attention_mask       = SanaStorage.nul_attention,
            use_resolution_binning          = SanaStorage.resolutionBin,
            
            do_bias_CFG                     = SanaStorage.biasCFG,

            pag_scale                       = PAG_scale,
            pag_adaptive_scale              = PAG_adapt,
        ).images

    if SanaStorage.noUnload:
        if SanaStorage.loadedLora == True:
            SanaStorage.pipeTR.unload_lora_weights()
            SanaStorage.loadedLora = False
        SanaStorage.pipeTR.transformer.to('cpu')
    else:
        SanaStorage.pipeTR.transformer = None
        SanaStorage.lastModel = None

    del generator, latents

    gc.collect()
    torch.cuda.empty_cache()

    SanaStorage.pipeTR.vae.to('cuda')

    original_samples_filename_pattern = opts.samples_filename_pattern
    opts.samples_filename_pattern = "Sana_[datetime]"
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
            model, sampler,
            positive_prompt, negative_prompt,
            guidance_scale, guidance_rescale, 
            num_steps, 
            fixed_seed + i, 
            width, height,
            PAG_scale, PAG_adapt,
            shift)

        if maskType > 0 and maskSource is not None:
            # i2iSource = i2iSource.convert('RGB')
            # image = image.convert('RGB')
            image = Image.composite(image, i2iSource, maskSource)

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
    opts.samples_filename_pattern = original_samples_filename_pattern

    del output

    if SanaStorage.noUnload:
        SanaStorage.pipeTR.vae.to('cpu')
    else:
        SanaStorage.pipeTR = None
        SanaStorage.lastModel = None

    gc.collect()
    torch.cuda.empty_cache()

    return fixed_seed, gradio.Button.update(interactive=True), results


def on_ui_tabs():
    if SanaStorage.ModuleReload:
        reload(styles)
        reload(pipeline)
    
    models_list = ["Efficient-Large-Model/Sana_600M_512px_diffusers", "Efficient-Large-Model/Sana_600M_1024px_diffusers", "Efficient-Large-Model/Sana_1600M_512px_diffusers", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers", "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"]
    defaultModel = models_list[3]
    defaultWidth = 1024
    defaultHeight = 1024
 
    def buildLoRAList ():
        loras = ["(None)"]
        
        import glob
        customLoRA = glob.glob(".\models\diffusers\SanaLora\*.safetensors")

        for i in customLoRA:
            filename = i.split('\\')[-1]
            loras.append(filename[0:-12])

        return loras

    loras = buildLoRAList ()

    def refreshLoRAs ():
        loras = buildLoRAList ()
        return gradio.Dropdown.update(choices=loras)
 
    def getGalleryIndex (index):
        if index < 0:
            index = 0
        return index

    def getGalleryText (gallery, index, seed):
        return gallery[index][1], seed+index

    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = image.size[0]
            h = image.size[1]
        return [w, h]

    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt
        image = image.convert('RGB')

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

    def i2iImageFromGallery (gallery, index):
        try:
            if SanaStorage.usingGradio4:
                newImage = gallery[index][0]
                return newImage
            else:
                newImage = gallery[index][0]['name'].rsplit('?', 1)[0]
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
            SanaStorage.pipeTE = None
            SanaStorage.pipeTR = None
            SanaStorage.lastModel = None
            shared.SuperPrompt_tokenizer = None
            shared.SuperPrompt_model = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')

    def toggleRandom ():
        SanaStorage.randomSeed ^= True
        return gradio.Button.update(variant='primary' if SanaStorage.randomSeed == True else 'secondary')

    def toggleBiasCFG ():
        if not SanaStorage.locked:
            SanaStorage.biasCFG ^= True
        return gradio.Button.update(variant='primary' if SanaStorage.biasCFG == True else 'secondary')

    def toggleResBin ():
        if not SanaStorage.locked:
            SanaStorage.resolutionBin ^= True
        return gradio.Button.update(variant='primary' if SanaStorage.resolutionBin == True else 'secondary',
                                value='\U0001D401' if SanaStorage.resolutionBin == True else '\U0001D539')

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



    def toggleGenerate (R, G, B, A, lora, scale):
        SanaStorage.noiseRGBA = [R, G, B, A]
        SanaStorage.lora = lora
        SanaStorage.lora_scale = scale# if lora != "(None)" else 1.0
        SanaStorage.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)

    def afterGenerate ():
        SanaStorage.locked = False
        return gradio.Button.update(value='Generate', variant='primary', interactive=True)

    schedulerList = ["default", "DDPM", "DEIS", "DPM++ 2M", "DPM++ 2M SDE", "DPM", "DPM SDE",
                     "Euler", "Euler A", "LCM", "SA-solver", "UniPC", ]

    def parsePrompt (positive, negative, sampler, width, height, seed, steps, cfg, rescale, nr, ng, nb, ns, PAG_scale, PAG_adapt, shift):
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
                if ": " == p[l][6:8]:                                   #   mine (old)
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
                if ": " == p[l][8:10]:                                  #   mine (old)
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
                            width = 32 * ((int(size[0]) + 16) // 32)
                            height = 32 * ((int(size[1]) + 16) // 32)
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                cfg = float(pairs[2])
                            elif "rescale:" == pairs[1]:
                                rescale = float(pairs[2])
                            else:
                                cfg = float(pairs[1])
                        case "width:":
                            width = 32 * ((int(pairs[1]) + 16) // 32)
                        case "height:":
                            height = 32 * ((int(pairs[1]) + 16) // 32)
                        case "PAG:":
                            if len(pairs) == 3:
                                PAG_scale = float(pairs[1])
                                PAG_adapt = float(pairs[2].strip('\(\)'))
                        case "Shift:":
                            shift = float(pairs[1])
                        case "Sampler:":
                            if len(pairs) == 3:
                                sampler = f"{pairs[1]} {pairs[2]}"
                            else:
                                sampler = pairs[1]

        return positive, negative, sampler, width, height, seed, steps, cfg, rescale, nr, ng, nb, ns, PAG_scale, PAG_adapt, shift

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
    resolutionList4096 = [
        (8192, 2048),   (6912, 2304),   (5632, 2816),   (5120, 3072),   (4864, 3328),
        (4096, 4096),
        (3328, 4864),   (3072, 5120),   (2816, 5632),   (2304, 6912),   (2048, 8192)
    ]


    def updateWH (dims, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        #   passing by value because of odd gradio bug? when using index can either update displayed list correctly, or get values correctly, not both
        wh = dims.split('\u00D7')
        return None, int(wh[0]), int(wh[1])

    def toggleSharp ():
        SanaStorage.sharpNoise ^= True
        return gradio.Button.update(value=['s', 'S'][SanaStorage.sharpNoise],
                                variant=['secondary', 'primary'][SanaStorage.sharpNoise])

    def maskFromImage (image):
        if image:
            return image, 'drawn'
        else:
            return None, 'none'


    with gradio.Blocks(analytics_enabled=False, head=canvas_head) as sana_block:
        with ResizeHandleRow():
            with gradio.Column():
                with gradio.Row():
                    access = ToolButton(value='\U0001F917', variant='secondary', visible=False)
                    model = gradio.Dropdown(models_list, label='Model', value=defaultModel, type='value', scale=2)
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='prompt enhancement')
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    sampler = gradio.Dropdown(["DPM++ 2M", "Euler", "Heun"], label='Sampler', value="DPM++ 2M", type='value', scale=0)

                with gradio.Row():
                    positive_prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here...', lines=2)
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)

                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', lines=1, value="")
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)
                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=128, maximum=4096, step=32, value=defaultWidth)
                    swapper = ToolButton(value="\U000021C4")
                    height = gradio.Slider(label='Height', minimum=128, maximum=4096, step=32, value=defaultHeight)
                    resBin = ToolButton(value="\U0001D401", variant='primary', tooltip="use resolution binning")
                    dims = gradio.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList1024],
                                        label='Quickset', type='value', scale=0)

                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=4.0, scale=1)
                    CFGrescale = gradio.Slider(label='rescale CFG', minimum=0.00, maximum=1.0, step=0.01, value=0.0, scale=1)
                    bCFG = ToolButton(value="0", variant='secondary', tooltip="use bias CFG correction")
                    shift = gradio.Slider(label='Shift', minimum=1, maximum=12.0, step=0.01, value=3.0, scale=0)
                with gradio.Row():
                    PAG_scale = gradio.Slider(label='Perturbed-Attention Guidance scale', minimum=0, maximum=8, step=0.1, value=0.0, scale=1)
                    PAG_adapt = gradio.Slider(label='PAG adaptive scale', minimum=0.00, maximum=0.1, step=0.001, value=0.0, scale=1)
                with gradio.Row():
                    steps = gradio.Slider(label='Steps', minimum=1, maximum=60, step=1, value=11, scale=1, visible=True)
                    random = ToolButton(value="\U0001f3b2\ufe0f", variant="primary")
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=0)

                with gradio.Row(equal_height=True):
                    lora = gradio.Dropdown([x for x in loras], label='LoRA (place in models/diffusers/SanaLora)', value="(None)", type='value', multiselect=False, scale=1)
                    refreshL = ToolButton(value='\U0001f504')
                    scale = gradio.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01, scale=1)

                with gradio.Accordion(label='the colour of noise', open=False):
                    with gradio.Row():
                        initialNoiseR = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gradio.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')
                        sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gradio.Accordion(label='image to image', open=False):
                    if SanaStorage.forgeCanvas:
                        i2iSource = ForgeCanvas(elem_id="Sana_img2img_image", height=320, scribble_color=opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=False, scribble_alpha=100, scribble_alpha_fixed=False, scribble_softness_fixed=False)
                        with gradio.Row():
                            i2iFromGallery = gradio.Button(value='Get gallery image')
                            i2iSetWH = gradio.Button(value='Set size from image')
                            i2iCaption = gradio.Button(value='Caption image')
                            toPrompt = ToolButton(value='P', variant='secondary')
                        
                        with gradio.Row():
                            i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                            AS = ToolButton(value='AS')
                            maskType = gradio.Dropdown(['i2i', 'inpaint mask', 'sketch', 'inpaint sketch'], value='i2i', label='Type', type='index')
                        with gradio.Row():
                            maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=64, step=1, value=0)
                            maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                 
                    else:
                        with gradio.Row():
                            i2iSource = gradio.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                            if SanaStorage.usingGradio4:
                                maskSource = gradio.ImageEditor(label='mask source', sources=['upload'], type='pil', interactive=True, show_download_button=False, layers=False, brush=gradio.Brush(colors=['#FFFFFF'], color_mode='fixed'))
                            else:
                                maskSource = gradio.Image(label='mask source', sources=['upload'], type='pil', interactive=True, show_download_button=False, tool='sketch', image_mode='RGB', brush_color='#F0F0F0')#opts.img2img_inpaint_mask_brush_color)
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

                            with gradio.Column():
                                maskType = gradio.Dropdown(['none', 'drawn', 'image', 'composite'], value='none', label='Mask', type='index')
                                maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=25, step=1, value=0)
                                maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                                maskCopy = gradio.Button(value='use i2i source as template')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if SanaStorage.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)

                if SanaStorage.forgeCanvas:
                    ctrls = [positive_prompt, negative_prompt, model, sampler, width, height, guidance_scale, CFGrescale, steps, sampling_seed, batch_size, i2iSource.background, i2iDenoise, style, PAG_scale, PAG_adapt, maskType, i2iSource.foreground, maskBlur, maskCut, shift]
                else:
                    ctrls = [positive_prompt, negative_prompt, model, sampler, width, height, guidance_scale, CFGrescale, steps, sampling_seed, batch_size, i2iSource, i2iDenoise, style, PAG_scale, PAG_adapt, maskType, maskSource, maskBlur, maskCut, shift]
                
                parseCtrls = [positive_prompt, negative_prompt, sampler, width, height, sampling_seed, steps, guidance_scale, CFGrescale, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, PAG_scale, PAG_adapt, shift]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gradio.Gallery(label='Output', height="80vh", type='pil', interactive=False, elem_id="Sana_gallery",
                                            show_label=False, object_fit='contain', visible=True, columns=3, rows=3, preview=True)

#   caption not displaying linebreaks, alt text does
                gallery_index = gradio.Number(value=0, visible=False)
                infotext = gradio.Textbox(value="", visible=False)
                base_seed = gradio.Number(value=0, visible=False)

                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=infotext,
                        source_image_component=output_gallery,
                    ))

        def set_dims(model):
            if "512" in model:
                resList = resolutionList512
            elif "2K" in model or "2048" in model:
                resList = resolutionList2048
            elif "4K" in model or "4096" in model:
                resList = resolutionList4096
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

        if SanaStorage.forgeCanvas:
            i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource.background, width, height], outputs=[width, height], show_progress=False)
            i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery, gallery_index], outputs=[i2iSource.background])
            i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource.background, positive_prompt], outputs=[positive_prompt])
        else:
            maskCopy.click(fn=maskFromImage, inputs=[i2iSource], outputs=[maskSource, maskType])
            i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
            i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery, gallery_index], outputs=[i2iSource])
            i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])

        noUnload.click(toggleNU, inputs=None, outputs=noUnload)
        unloadModels.click(unloadM, inputs=None, outputs=None, show_progress=True)

        SP.click(toggleSP, inputs=None, outputs=SP).then(superPrompt, inputs=[positive_prompt, sampling_seed], outputs=[SP, positive_prompt])
        sharpNoise.click(toggleSharp, inputs=None, outputs=sharpNoise)
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        parse.click(parsePrompt, inputs=parseCtrls, outputs=parseCtrls, show_progress=False)
        access.click(toggleAccess, inputs=None, outputs=access)
        bCFG.click(toggleBiasCFG, inputs=None, outputs=bCFG)
        resBin.click(toggleResBin, inputs=None, outputs=resBin)
        swapper.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
        random.click(toggleRandom, inputs=None, outputs=random, show_progress=False)
        AS.click(toggleAS, inputs=None, outputs=AS)
        refreshL.click(refreshLoRAs, inputs=None, outputs=[lora])

        toPrompt.click(toggleC2P, inputs=None, outputs=[toPrompt])

        output_gallery.select(fn=getGalleryIndex, js="selected_gallery_index", inputs=gallery_index, outputs=gallery_index, show_progress=False).then(fn=getGalleryText, inputs=[output_gallery, gallery_index, base_seed], outputs=[infotext, sampling_seed], show_progress=False)

        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button, SP]).then(predict, inputs=ctrls, outputs=[base_seed, SP, output_gallery], show_progress='full').then(fn=afterGenerate, inputs=None, outputs=generate_button).then(fn=getGalleryIndex, js="selected_gallery_index", inputs=gallery_index, outputs=gallery_index, show_progress=False).then(fn=getGalleryText, inputs=[output_gallery, gallery_index, base_seed], outputs=[infotext, sampling_seed], show_progress=False)

    return [(sana_block, "Sana", "sana_DoE")]

script_callbacks.on_ui_tabs(on_ui_tabs)

