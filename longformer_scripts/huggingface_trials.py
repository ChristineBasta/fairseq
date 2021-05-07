import torch
from transformers import LongformerModel, LongformerTokenizer
from transformers import LongformerForMaskedLM, LongformerForSequenceClassification
class LongformerUsage:
    def __init__(self):
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    # LongformerModel output
    def get_output(self, doc_string):
        # the documents change
        #SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        SAMPLE_TEXT = doc_string
        print('TOKENIZED for doc by self.tokenizer.tokenize:')
        print(self.tokenizer.tokenize(SAMPLE_TEXT))
        print('encodings for doc by self.tokenizer.encode:')
        print(self.tokenizer.encode(SAMPLE_TEXT))

        input_ids = torch.tensor(self.tokenizer.encode(SAMPLE_TEXT, add_special_tokens=True)).unsqueeze(0)  # batch of size 1
        print(input_ids)
        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
        # TODO(Christine) learn what to change here according to your problem
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                    device=input_ids.device)  # initialize to local attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long,
                                            device=input_ids.device)  # initialize to global attention to be deactivated for all tokens
        # TODO(Christine) compute the global attention depending on <S> and <P>

        # the gloabl attention set to the places needed
        global_attention_mask[:, [0]] = 1   # Set global attention to random tokens for the sake of this example
        # Usually, set global attention based on the task. For example,
        # classification: the <s> token
        # QA: question tokens
        # LM: potentially on the beginning of sentences and paragraphs

        outputs = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, output_attentions=True, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        print('hidden_states:')
        print(len(outputs.hidden_states))
        print('golbal_attentions:')
        print(len(outputs.global_attentions))
        print(outputs.global_attentions[0].shape) #batch, tokens, embedding, global_attention_tokens

        print(sequence_output)
        return outputs

    # LongformerForMaskedLM
    def get_output_maskedLM(self):
        model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
        attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
        # check ``LongformerModel.forward`` for more details how to set `attention_mask`
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        prediction_logits = outputs.logits

    #LongFormer sequence classification
    def get_output_seq_classification(self):
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

    def get_end_sentences(self, sentence):
        ids=self.tokenizer.encode(sentence)
        print(ids)
        index=0
        list_indices_end=[]
        list_indices_end.append(0)
        list_indices_start=[]
        list_indices_start.append(0)
        for id in ids:
            if (id==4): #end of se
                list_indices_end.append(index)
                list_indices_start.append(index+1)
            index=index+1
        return list_indices_start, list_indices_end

class BertUsage:
    def __init__(self):
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        # LongformerModel output

    def get_output(self, doc_string):
        return doc_string

if __name__ == "__main__":
    print ('main')
    longFormer = LongformerUsage()
    longFormer.get_output('There is hope. There is love. Hello all from the biggest company.')
    #end, beg=longFormer.get_end_sentences('There is hope. There is love. Hello all from the biggest company.')
    #print(end)
    #print(beg)





