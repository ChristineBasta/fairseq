import torch
import logging
from transformers import LongformerModel, LongformerTokenizer
from transformers import LongformerForMaskedLM, LongformerForSequenceClassification
from transformers import LEDTokenizer, LEDModel
import numpy

class LongformerUsage:
    def __init__(self):
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.led_tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
        self.led_model = LEDModel.from_pretrained('allenai/led-base-16384')
    # LongformerModel output
    #classify_model is set to ture if classification is targeted, and false if modeling
    #to change the
    def get_output(self, doc_string, classify_model):
        # the documents change
        #SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        SAMPLE_TEXT = doc_string
        # getting start ids of sentences so we set the attention to them
        start, end =self.get_end_sentences(doc_string)
        input_ids = torch.tensor(self.tokenizer.encode(SAMPLE_TEXT, add_special_tokens=True)).unsqueeze(0)  # batch of size 1

        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
        # TODO(Christine) learn what to change here according to your problem
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                    device=input_ids.device)  # initialize to local attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long,
                                            device=input_ids.device)  # initialize to global attention to be deactivated for all tokens
        # TODO(Christine) compute the global attention depending on <S> and <P>

        # the global attention set to the places needed
        # Usually, set global attention based on the task. For example,
        # classification: the <s> token
        # QA: question tokens
        # LM: potentially on the beginning of sentences and paragraphs
        if classify_model:
            global_attention_mask[:, [0]] = 1
        else:
            global_attention_mask[:, start] = 1   # Set global attention to random tokens for the sake of this example

        outputs = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, output_attentions=True, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state

        mean_sequence_output=torch.mean(outputs.last_hidden_state, dim=1)
        logging.info(mean_sequence_output.shape)
        pooled_output = outputs.pooler_output


        return outputs, mean_sequence_output, pooled_output

    def get_output_certain_tokens(self, doc_string, classify_model):
        outputs, _, _=self.get_output(doc_string, classify_model)
        first_token_last_hidden = outputs.last_hidden_state[0][0].view(1, -1)
        start, end = self.get_end_sentences(doc_string)
        start_numpy = numpy.array(start)
        start_tensor = torch.from_numpy(start_numpy)
        global_attention_tokens = torch.index_select(outputs.last_hidden_state[0], 0, start_tensor)
        mean_global_attentions=torch.mean(global_attention_tokens, dim=0)
        mean_global_attentions = mean_global_attentions.view(1, -1)
        return first_token_last_hidden, global_attention_tokens, mean_global_attentions

    # LongformerForMaskedLM
    def get_output_maskedLM(self, sample_text):

        model_maskedmodel = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
        #SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        input_ids = torch.tensor(self.tokenizer.encode(sample_text)).unsqueeze(0)  # batch of size 1
        attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
        # check ``LongformerModel.forward`` for more details how to set `attention_mask`
        outputs = model_maskedmodel(input_ids, attention_mask=attention_mask, labels=input_ids, output_hidden_states=True, output_attentions=True)
        loss = outputs.loss
        prediction_logits = outputs.logits
        mean_last_hidden=torch.mean(outputs.hidden_states[len( outputs.hidden_states)-1], dim=1)
        return outputs, mean_last_hidden

    #LongFormer sequence classification
    def get_output_seq_classification(self,sentence):
        sequence_classify_model = LongformerForSequenceClassification.from_pretrained(
            'allenai/longformer-base-4096')
        #inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        inputs = self.tokenizer(sentence, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = sequence_classify_model(**inputs, labels=labels, output_hidden_states=True, output_attentions=True)
        loss = outputs.loss
        logits = outputs.logits
        mean_last_hidden = torch.mean(outputs.hidden_states[len(outputs.hidden_states) - 1], dim=1)
        return outputs, mean_last_hidden

        # LongFormer sequence classification

    def get_led_representation(self, sentence):
        inputs = self.led_tokenizer(sentence, return_tensors="pt")
        outputs = self.led_model(**inputs, decoder_input_ids=inputs.input_ids)

        last_hidden_states = outputs.last_hidden_state
        mean_last_hidden=torch.mean(outputs.last_hidden_state, dim=1)
        mean_encoder_last_hidden = torch.mean(outputs.encoder_last_hidden_state, dim=1)
        first_token_encoder_last=outputs.encoder_last_hidden_state[0][0].view(1, -1)
        first_token_last = outputs.last_hidden_state[0][0].view(1, -1)
        print(outputs.encoder_last_hidden_state[0][0].shape)
        print(outputs.encoder_last_hidden_state[0][0].view(1, -1).shape)
        return mean_last_hidden, mean_encoder_last_hidden, first_token_encoder_last, first_token_last

    def get_end_sentences(self, sentence):
        ids=self.tokenizer.encode(sentence)

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


if __name__ == "__main__":
    print ('main')
    longFormer = LongformerUsage()

    first_token_last_hidden, global_attention_tokens, mean_global_attentions=longFormer.get_output_certain_tokens('There is hope. There is love. Hello all from the biggest company.', classify_model=True)

    print(first_token_last_hidden)
    print(global_attention_tokens)
    print(mean_global_attentions)




