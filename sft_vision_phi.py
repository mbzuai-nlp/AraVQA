"""
finetune Phi-4-multimodal-instruct on an image task
code source : https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_finetune_vision.py
Note : We are not from microsoft team and thus we are not breaking the anonymity policy by sharing this link. We are just citing the source for the code
scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.47.0
accelerate==1.3.0
"""


from transformers.trainer_utils import get_last_checkpoint
import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
import torch
from vllm.lora.request import LoRARequest
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Trainer,
    TrainingArguments,
)
import ast
import requests


MODEL_ID="microsoft/Phi-4-multimodal-instruct"
PROCESSOR_ID="microsoft/Phi-4-multimodal-instruct"
OUTPUT_DIR="Phi4-SFT"

#TASK="ours"
TASK="pearl"

DEFAULT_INSTSRUCTION =""
_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 16_384
TOTAL_BATCH_SIZE=16
BATCH_SIZE=16
EPOCH=1
LEARNING_RATE=1.0e-5
WD=0.01
USE_FLASH_ATTENTION=False
DYNAMIC_HD=36
_TRAIN_SIZE = 10000000
_EVAL_SIZE = 100000000
result_list=[]


def isevaluatable(s):
    try:
        ast.literal_eval(s)
        return True
    except ValueError:
        return False

def resize_img(img):

        original_width, original_height = img.size
        if original_width > original_height:
            scale_factor = 224 / original_width
        else:
            scale_factor = 224 / original_height

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img


class GenTrainDataset(Dataset):
    def __init__(self, processor, data_size, instruction=DEFAULT_INSTSRUCTION):

        self.train_data = load_dataset("csv", data_files={"train": "AraVQA-Train.csv"}, keep_default_na=False, split="train")

        self.processor = processor
        self.instruction = instruction

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        annotation = self.train_data[idx]
        url = annotation["img_url"]
        image = Image.open(requests.get(url, stream=True).raw)
        image = resize_img(image)
        question=annotation["question"]
        A=annotation["A"]
        B=annotation["B"]
        C=annotation["C"]
        D=annotation["D"]
        answer=annotation["answer"]
        answer_letter=annotation["answer_letter"]
        prompt= annotation['question']+"\nA. "+A+"\nB. "+B+"\nC. "+C+"\nD. "+D
        prompt= "Answer the following question by writing the answer letter and answer between <answer> tags (e.g, <answer>A. Paris</answer>)\n"+prompt
        output= "<answer>" + annotation["answer_letter"]+". "+annotation["answer"]+"</answer>"
        user_message = {'role': 'user','content': '<|image_1|>' + "\n"+ prompt}
        prompt = self.processor.tokenizer.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        answer = f'{output}<|end|><|endoftext|>'
        inputs = self.processor(prompt, images=[image], return_tensors='pt')
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                # workaround to make sure loss compute won't fail
                labels[:, -1] = self.processor.tokenizer.eos_token_id

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_image_embeds': inputs.input_image_embeds,
            'image_attention_mask': inputs.image_attention_mask,
            'image_sizes': inputs.image_sizes,
        }



class GenEvalDataset(Dataset):
    def __init__(self, processor, data_size, instruction=DEFAULT_INSTSRUCTION, rank=0, world_size=1):
        # Download the file
        if TASK=="pearl":
             self.eval_data = load_dataset("UBC-NLP/PEARL", split="test")
        if TASK=="ours":
             self.eval_data = load_dataset("csv", data_files={"test": "AraVQA-Test.csv"}, keep_default_na=False, split="test")
        self.processor = processor
        self.instruction = instruction

    def __len__(self):
        return len(self.eval_data)


    def __getitem__(self, idx):
        annotation = self.eval_data[idx]

        if TASK=="pearl":
             img_path = annotation["image"]
             image = resize_img(image)
             question= annotation['question']
             choices = eval(annotation['choices']) ## eval will convert the list represented as text to an actual list with eval. if it throw an error use isevaluatable function
             prompt= question + "\n".join(choices)
             output = annotation['answer_letter'] + ". "+ annotation["answer"]

        if TASK=="ours":
             url = annotation["img_url"]
             image = Image.open(requests.get(url, stream=True).raw)
             image = resize_img(image)
             question=annotation["question"]
             A=annotation["A"]
             B=annotation["B"]
             C=annotation["C"]
             D=annotation["D"]
             answer=annotation["answer"]
             answer_letter=annotation["answer_letter"]
             prompt= annotation['question']+"\nA. "+A+"\nB. "+B+"\nC. "+C+"\nD. "+D
             output= annotation["answer_letter"]+". "+annotation["answer"]
        prompt = "Answer the following question by writing the answer letter and answer between <answer> tags (e.g, <answer>A. Paris</answer>)\n"+prompt
        output="<answer>" +output +"</answer>"
        user_message = {'role': 'user','content': '<|image_1|>' + "\n"+ prompt}
        prompt = self.processor.tokenizer.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        answer = f'{output}<|end|><|endoftext|>'

        inputs = self.processor(prompt, images=[image], return_tensors='pt')
        eval_idx=random.randint(0, 100000000000000000)
        unique_id = f'{eval_idx:010d}'
        return {
            'id': unique_id,
            'input_ids': inputs.input_ids,
            'input_image_embeds': inputs.input_image_embeds,
            'image_attention_mask': inputs.image_attention_mask,
            'image_sizes': inputs.image_sizes,
            'answer': answer,
        }



def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def vision_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_image_embeds_list.append(inputs['input_image_embeds'])
        image_attention_mask_list.append(inputs['image_attention_mask'])
        image_sizes_list.append(inputs['image_sizes'])

    input_ids = pad_sequence(input_ids_list, padding_side='right', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='right', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_image_embeds': input_image_embeds,
            'image_attention_mask': image_attention_mask,
            'image_sizes': image_sizes,
            'input_mode': 1,  # vision mode
        }
    )


def vision_eval_collate_fn(batch):
    input_ids_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    all_unique_ids = []
    all_answers = []
    for inputs in batch:
        all_unique_ids.append(inputs['id'])
        input_ids_list.append(inputs['input_ids'][0])
        input_image_embeds_list.append(inputs['input_image_embeds'])
        image_attention_mask_list.append(inputs['image_attention_mask'])
        image_sizes_list.append(inputs['image_sizes'])
        all_answers.append(inputs['answer'])

    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    return (
        all_unique_ids,
        all_answers,
        BatchFeature(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'input_image_embeds': input_image_embeds,
                'image_attention_mask': image_attention_mask,
                'image_sizes': image_sizes,
                'input_mode': 1,  # vision mode
            }
        ),
    )


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')
    model.config.use_cache = False
    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    # TODO remove unused vision layers?
    return model


@torch.no_grad()



def get_answer_letter(s):

  for i in range(len(s)):
        text=s[:i]
        if "A." in text or "a." in text:
           if "B." and "C." and "D." and "b." and "c." and "d."  not in text:
                return "A"
        elif "B." in text or "b." in text:
           if "A." and "C." and "D." and "a." and "c." and "d."  not in text:
                return "B"
        elif "C." in text or "c." in text:
           if "A." and "B." and "D." and "a." and "b." and "d."  not in text:
                return "C"
        elif "D." in text or "d." in text:
           if "A." and "B." and "C." and "a." and "b." and "c."  not in text:
                return "D"
  return s

def calc_acc(pred_list,actual_list):

        correct_answer=0
        total_answer=len(actual_list)
        for pred,actual in zip(pred_list,actual_list):
                pred=pred.strip()
                pred=pred.replace("<answer>","").replace("</answer>","")
                actual=actual.strip()
                ref_pred=get_answer_letter(pred)
                ref_actual=get_answer_letter(actual)
                if ref_pred.lower()==ref_actual.lower():
                        correct_answer+=1
        acc_score=correct_answer/total_answer
        return acc_score


def evaluate(model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_answers = []
    all_generated_texts = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=vision_eval_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    for ids, answers, inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'
    ):
        all_answers.extend({'id': i, 'answer': a.strip().lower()} for i, a in zip(ids, answers))

        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=16)

        input_len = inputs.input_ids.size(1)
        generated_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        all_generated_texts.extend(
            {'id': i, 'generated_text': g.strip().lower()} for i, g in zip(ids, generated_texts)
        )

    # gather outputs from all ranks
    all_answers = gather_object(all_answers)
    all_generated_texts = gather_object(all_generated_texts)
    if rank == 0:
        assert len(all_answers) == len(all_generated_texts)
        actual_list=[]
        pred_list=[]
        for actual,predict in zip(all_answers, all_generated_texts):
            current_actual=actual["answer"].replace("<|end|><|endoftext|>","")
            current_predict=predict["generated_text"]
            actual_list.append(current_actual)
            pred_list.append(current_predict)
        result=calc_acc(pred_list,actual_list)
        print(f"Current Acc. : {result}")
    result_list.append(str(result))

    return str(result)
    #return 0


def main(eval_before=True,train_model=True,eval_after=True):

    accelerator = Accelerator()
    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            PROCESSOR_ID,
            trust_remote_code=True,
            dynamic_hd=DYNAMIC_HD,
        )
        model = create_model(
            MODEL_ID,
            USE_FLASH_ATTENTION,
        )
    # tune vision encoder and lora
    #model.config.use_cache = False

    if MODEL_ID=="microsoft/Phi-4-multimodal-instruct":
        model.set_lora_adapter('vision')
        for param in model.model.embed_tokens_extend.image_embed.parameters():
            param.requires_grad = True

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    train_dataset = GenTrainDataset(processor, data_size=None if FULL_RUN else _TRAIN_SIZE)
    eval_dataset = GenEvalDataset(
        processor,
        data_size=None if FULL_RUN else _EVAL_SIZE,
        rank=rank,
        world_size=world_size,
    )

    num_gpus = accelerator.num_processes

    print(f'training on {num_gpus} GPUs')
    assert (TOTAL_BATCH_SIZE % (num_gpus * BATCH_SIZE) == 0)

    gradient_accumulation_steps = TOTAL_BATCH_SIZE // (num_gpus * BATCH_SIZE)
    if USE_FLASH_ATTENTION:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=EPOCH,
	max_steps=-1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=LEARNING_RATE,
        weight_decay=WD,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        save_strategy='no',
        save_steps=20,
        save_total_limit=1,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=False,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
    )

    # eval before fine-tuning
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = False

    if eval_before==True:
        acc = evaluate(
              model,
              processor,
              eval_dataset,
              save_path=out_path / 'eval_before.json',
              disable_tqdm=False,
              eval_batch_size=BATCH_SIZE,
         )
        if accelerator.is_main_process:
                 print(f'Accuracy before finetuning: {acc}')
 


    if train_model==True:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=vision_collate_fn,
            train_dataset=train_dataset,
        )
        trainer.train()
        trainer.save_model()
        accelerator.wait_for_everyone()

        del model
        del trainer
        __import__('gc').collect()
        torch.cuda.empty_cache()

 

    if eval_after==True:
        model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,
            torch_dtype=torch.bfloat16 if USE_FLASH_ATTENTION else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if USE_FLASH_ATTENTION else 'sdpa',
        ).to('cuda')
        processor = AutoProcessor.from_pretrained(PROCESSOR_ID,trust_remote_code=True,dynamic_hd=DYNAMIC_HD)
        acc = evaluate(
            model,
            processor,
            eval_dataset,
            save_path=out_path / 'eval_after.json',
            disable_tqdm=False,
            eval_batch_size=BATCH_SIZE,
        )

        if accelerator.is_main_process:
              print(f'Accuracy after finetuning: {acc}')
        print(result_list)

if __name__ == '__main__':
    main(eval_before=True,train_model=False,eval_after=True)
