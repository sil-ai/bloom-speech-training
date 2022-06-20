import transformers
from transformers import AutoModelForCTC, Wav2Vec2Processor
import datasets
import time
import jiwer
import statistics
import huggingface_hub
from datasets import load_dataset, load_metric, Audio
import json
import torch
from clearml import Task
from clearml.config import config_obj 
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # ClearML stuff
    Task.add_requirements("-rrequirements.txt")
    task = Task.init(
      project_name='IDX-Bloom-Speech',    # project name of at least 3 characters
      task_name='ASR-model-scoring-' + str(int(time.time())), # task name of at least 3 characters
      task_type="testing",
      tags=None,
      reuse_last_task_id=True,
      continue_last_task=False,
      output_uri="s3://bloom-speech/models/",
      auto_connect_arg_parser=True,
      auto_connect_frameworks=True,
      auto_resource_monitoring=True,
      auto_connect_streams=True,    
    )

    # HF login
    token = config_obj.get("huggingface.token")
    huggingface_hub.hf_api.set_access_token(token)
    huggingface_hub.HfFolder.save_token(token)

    langs= ['bam', 'boz', 'bzi', 'cak', 'ceb', 'chd', 'eng', 'fra', 'hbb', 'jra', 'kan', 'kek', 'kjb', 'mam', 'mya', 'myk', 'quc', 'sdk', 'snk', 'spa', 'stk', 'tgl', 'tpi']

    with open('scores.txt', 'w') as f:
        f.write('lang\twer\tcer\n')

    for language in langs:
        repo = f'jnemecek/wav2vec2-bloom-speech-{language}'

        model = AutoModelForCTC.from_pretrained(repo, use_auth_token=True).to("cuda")
        #processor = Wav2Vec2Processor.from_pretrained(repo, use_auth_token=True)
        data = load_dataset('sil-ai/bloom-speech', language, split='test', use_auth_token=True)
        processor = Wav2Vec2Processor.from_pretrained(repo, use_auth_token=True)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(repo, use_auth_token=True)

        wer_metric = load_metric("wer")
        cer_metric = load_metric("cer")

        def prepare_dataset(batch):
            audio = batch["audio"]

            # batched output is "un-batched"
            #warnings.simplefilter("ignore") #catch librosa warnings for mp3 processing
            batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            batch["input_length"] = len(batch["input_values"])
            
            with processor.as_target_processor():
                batch["labels"] = processor(batch["text"]).input_ids
            return batch

        def clear_unks(text):
            text = text.replace('[UNK]', '')
            text = text.replace('  ', ' ')
            return(text)

        data = data.map(prepare_dataset)

        preds = []
        refs = []

        for item in tqdm(data):
            input_dict = processor(item["input_values"], return_tensors="pt", padding=True, sampling_rate=16000)
            if clear_unks(processor.decode(item["labels"])).strip() != '' and len(item['input_values']) < (25*16000):
                try:
                    logits = model(input_dict.input_values.to("cuda")).logits
                    pred_ids = torch.argmax(logits, dim=-1)[0]

                    preds.append(processor.decode(pred_ids))
                    refs.append(clear_unks(processor.decode(item["labels"])))
                except:
                    torch.cuda.empty_cache()
                    pass

        wer = wer_metric.compute(predictions=preds, references=refs)
        cer = cer_metric.compute(predictions=preds, references=refs)

        with open('scores.txt', 'a') as f:
            f.write(f'{language}:\t{wer}\t{cer}\n')

        with open('sample_predictions.txt', 'a') as f:
            f.write(f'Predictions for {language}:\n')
            f.write('Predictions\tReferences\n')
            for idx in range(0,10):
                f.write(f'{preds[idx]}\t{refs[idx]}\n')

        task.upload_artifact('local file','scores.txt')
        task.upload_artifact('local file', 'sample_predictions.txt')

if __name__ == "__main__":
    main()
