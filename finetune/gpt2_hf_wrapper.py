import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from gpt2_min import GPT2, GPTConfig

class GPT2ConfigHF(PretrainedConfig):
    model_type = "gpt2_custom"
    
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class GPT2HF(PreTrainedModel):
    config_class = GPT2ConfigHF
    
    def __init__(self, config):
        super().__init__(config)
        # Convert HF config to our internal GPTConfig dataclass
        self.internal_config = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout
        )
        self.model = GPT2(self.internal_config)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through your custom model
        # Note: Your GPT2 handles 'targets' for loss calculation internally if provided
        results = self.model(
            idx=input_ids, 
            targets=labels, 
            output_hidden_states=output_hidden_states
        )
        
        # Unpack results based on what was returned
        if output_hidden_states:
            logits, loss, hidden_states = results
        else:
            logits, loss = results
            hidden_states = None

        if not return_dict:
            return (loss, logits, hidden_states) if loss is not None else (logits, hidden_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None, # Your model doesn't support KV-cache yet
            hidden_states=hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Minimal implementation to support generate()
        return {"input_ids": input_ids}