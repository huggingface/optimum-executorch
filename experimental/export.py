import subprocess
import tempfile

transformer_text_models =  ['bert', 'cohere', 'cohere2', 'distilbert', 'gemma', 'gemma2', 'ibert', 'llama', 
 'mistral', 'mobilebert', 'olmo', 'qwen', 'qwen2', 'roberta', 'smollm', 'bamba', 
 'bart', 'barthez', 'bartpho', 'bert_generation', 'bert_japanese', 'bertweet', 
 'big_bird', 'bigbird_pegasus', 'biogpt', 'blenderbot', 'blenderbot_small', 
 'bloom', 'byt5', 'camembert', 'canine', 'codegen', 'colpali', 'convbert', 'cpm', 
 'cpmant', 'ctrl', 'deberta', 'deberta_v2', 'dpr', 'electra', 'ernie', 'ernie_m', 
 'esm', 'falcon', 'flan_t5', 'flan_ul2', 'flaubert', 'fnet', 'fsmt', 'funnel', 
 'gpt2', 'gpt_bigcode', 'gpt_neo', 'gpt_neox', 'gptj', 'gptsan_japanese', 'led', 
 'lilt', 'longformer', 'longt5', 'luke', 'm2m_100', 'mamba', 'marian', 'mbart', 
 'mega', 'megatron_bert', 'mixtral', 'mpnet', 'mpt', 'mra', 'mt5', 'mvp', 'nat', 
 'nezha', 'nllb', 'nllb_moe', 'nystromformer', 'one_piece', 'openai_gpt', 'opt', 
 'palm', 'pegasus', 'pegasus_x', 'persimmon', 'phi', 'phi3', 'phobert', 'plbart', 
 'prophetnet', 'qdqbert', 'rag', 'realm', 'reformer', 'rembert', 'retribert', 
 'roberta_prelayernorm', 'roc_bert', 'roformer', 'rwkv', 'splinter', 
 'squeezebert', 'stablelm', 'starcoder', 'starcoder2', 'switch_transformers', 
 't5', 'umt5', 'xglm', 'xlm', 'xlm_prophetnet', 'xlm_roberta', 'xlm_roberta_xl', 
 'xlnet', 'xmod', 'yoso', 'zebranet', 'zephyr']


EXPORT_TEXT_GENERATION = {
    "align": "kakaobrain/align-base",
    "altclip": "BAAI/AltCLIP",
    "aria": "aria-text/aria",
    "audio_spectrogram_transformer": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "auto": "google/auto",
    "autoformer": "huggingface/autoformer",
    "bamba": "ibm-fms/Bamba-9B",
    "bark": "suno/bark",
    "barthez": "moussaKam/barthez",
    "bartpho": "vinai/bartpho-syllable",
    "bert": "bert-base-uncased",
    "bert_generation": "google/bert_for_seq_generation_L-24_bbc_encoder",
    "bert_japanese": "cl-tohoku/bert-base-japanese",
    "bertweet": "vinai/bertweet-base",
    "big_bird": "google/bigbird-roberta-base",
    "bigbird_pegasus": "google/bigbird-pegasus-large-arxiv",
    "biogpt": "microsoft/biogpt",
    "bit": "google/bit",
    "blenderbot_small": "facebook/blenderbot_small-90M",
    "blip": "Salesforce/blip-image-captioning-base",
    "blip_2": "Salesforce/blip2-flan-t5-xl",
    "bridgetower": "microsoft/bridgetower-base",
    "bros": "naver-clova-ocr/bros-base-uncased",
    "canine": "google/canine-s",
    "chameleon": "chameleon/chameleon",
    "chinese_clip": "OFA-Sys/chinese-clip-vit-base-patch16",
    "clap": "laion/clap-htsat-fused",
    "clipseg": "CIDAS/clipseg-rd64-refined",
    "clvp": "openai/clvp",
    "cohere": "CohereForAI/c4ai-command-r-v01",
    "cohere2": "CohereForAI/c4ai-command-r7b-12-2024",
    "colpali": "vidore/colpali-v1.2-hf",
    "conditional_detr": "microsoft/conditional-detr-resnet-50",
    "convnextv2": "facebook/convnextv2-base-22k-224",
    "cpm": "mymusise/CPM-GPT2-FP16",
    "cpmant": "openbmb/cpm-ant-10b",
    "ctrl": "salesforce/ctrl",
    "dab_detr": "facebook/dab-detr-resnet-50",
    "data2vec": "facebook/data2vec-vision-base",
    "decision_transformer": "edbeeching/decision-transformer-gym-hopper-medium",
    "deformable_detr": "SenseTime/deformable-detr",
    "depth_anything": "Intel/depth-anything",
    "deta": "microsoft/deta",
    "dinat": "shi-labs/dinat-mini-in1k-224",
    "dinov2": "facebook/dinov2-base",
    "dit": "microsoft/dit-base",
    "dpr": "facebook/dpr-question_encoder-single-nq-base",
    "efficientformer": "snap-research/efficientformer-l1",
    "efficientnet": "google/efficientnet-b0",
    "ernie": "nghuyong/ernie-2.0-en",
    "esm": "facebook/esm2_t6_8M_UR50D",
    "falcon": "tiiuae/falcon-7b",
    "fastspeech2": "facebook/fastspeech2-en-ljspeech",
    "flava": "facebook/flava-full",
    "focalnet": "microsoft/focalnet-tiny-srf",
    "fuyu": "fuyu/fuyu",
    "gemma": "google/gemma-7b",
    "gemma2": "google/gemma2-7b",
    "git": "microsoft/git-base",
    "gpt_bigcode": "bigcode/gpt_bigcode-santacoder",
    "gpt_neo": "EleutherAI/gpt-neo-125M",
    "gpt_neox": "EleutherAI/gpt-neox-20b",
    "gptsan_japanese": "rinna/japanese-gpt-neox-3.6b-instruction-sft",
    "graphormer": "microsoft/graphormer-base",
    "hubert": "facebook/hubert-base-ls960",
    "idefics": "HuggingFaceM4/idefics-80b",
    "ijepa": "facebook/ijepa-base-400w",
    "informer": "huggingface/informer",
    "instructblip": "Salesforce/instructblip-flan-t5-xl",
    "jukebox": "openai/jukebox-1b-lyrics",
    "kosmos2": "microsoft/kosmos-2",
    "layoutlmv2": "microsoft/layoutlmv2-base-uncased",
    "llava": "liuhaotian/llava-v1-13b",
    "lxmert": "unc-nlp/lxmert-base-uncased",
    "mamba": "state-spaces/mamba-130m-hf",
    "markuplm": "microsoft/markuplm-base",
    "mask2former": "facebook/mask2former-swin-base-coco",
    "maskformer": "facebook/maskformer-swin-base-coco",
    "mega": "mnaylor/mega-base-wikitext",
    "mgp_str": "tesseract-ocr/str",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "mobilenet_v1": "google/mobilenet_v1_1.0_224",
    "mobilenet_v2": "google/mobilenet_v2_1.0_224",
}

EXPORT_TEXT2TEXT_GENERATION = {
    "flan_t5": "google/flan-t5-small",
    "byt5": "google/byt5-small",
    "flan_ul2": "google/flan-ul2",
    "fsmt": "facebook/wmt19-en-ru",
    "led": "allenai/led-base-16384",
    "longformer": "allenai/longformer-base-4096",
    "m2m_100": "facebook/m2m100_418M",
}

EXPORT_MASKED_LM_MODELS = {
    "deberta_v2": "microsoft/deberta-v2-xlarge",
    "ernie_m": "PaddlePaddle/ernie-m-base",
    "fnet": "google/fnet-base",
    "funnel": "huggingface/funnel-transformer",
    "luke": "studio-ousia/luke-base",
    "megatron_bert": "nvidia/megatron-bert-uncased-345m",
}

succeeded = 0
failed = 0
for k, v in {**EXPORT_TEXT_GENERATION, **EXPORT_TEXT2TEXT_GENERATION, **EXPORT_MASKED_LM_MODELS}.items():
    if k in transformer_text_models:
        print(f"exporting {k}")
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                if k in EXPORT_TEXT_GENERATION:
                    subprocess.run(
                        f"optimum-cli export executorch --model {v} --task \"text-generation\" --recipe xnnpack --output_dir {tempdir}/executorch",
                        shell=True,
                        check=True,
                    )
                if k in EXPORT_TEXT2TEXT_GENERATION:
                    subprocess.run(
                        f"optimum-cli export executorch --model {v} --task \"text2text-generation\" --recipe xnnpack --output_dir {tempdir}/executorch",
                        shell=True,
                        check=True,
                    )
                if k in EXPORT_MASKED_LM_MODELS:
                    subprocess.run(
                        f"optimum-cli export executorch --model {v} --task \"fill-mask\" --recipe xnnpack --output_dir {tempdir}/executorch",
                        shell=True,
                        check=True,
                    )
            except Exception as e:
                print(f"Failed to export {k} with error {e}")
                failed += 1
                continue
            else:
                succeeded += 1
                print(f"Successfully exported {k}")

print(f"Successfully exported {succeeded} models, failed to export {failed} models")
