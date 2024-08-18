from transformers import BertConfig, XLMRobertaConfig


class BertNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertNerConfig, self).__init__(**kwargs)
        self.model_dropout = kwargs.get("model_dropout", 0.1)
        
class XLMRNerConfig(XLMRobertaConfig):
    def __init__(self, **kwargs):
        super(XLMRNerConfig, self).__init__(**kwargs)
        self.model_dropout = kwargs.get("model_dropout", 0.1)

