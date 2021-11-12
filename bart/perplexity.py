import transformers as t

import torch
from nlp import load_dataset

def calc():
    model = t.AutoModelForSeq2SeqLM.from_pretrained('results2/checkpoint-31500/')
    #model_checkpoint = 'results2/checkpoint-31500/scheduler.pt'
    tokenizer = t.BartTokenizerFast.from_pretrained('results2/checkpoint-31500')
    #model = t.BartForConditionalGeneration.from_pretrained("facebook/bart-base", return_dict=True)
    #model.load_state_dict(torch.load("results2/checkpoint-31500/rng_state.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = load_dataset("xsum", split='test[:20]')
    encodings =  tokenizer(test['document'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

    model = model.to(device)
    model.eval()
    number_beams = 8
    result = model.generate(encodings['input_ids'],  num_beams=number_beams,  max_length=model.config.max_length) #, output_scores=True, output_attentions=True)
    
    log_sent = []

    for batch_num in range(0, result.scores[0].shape[0], number_beams):
        max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
        for beam_num in range(number_beams):
            max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
        log_sent.append(max_score)
        
    print("Perplexity:", torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))

calc()
