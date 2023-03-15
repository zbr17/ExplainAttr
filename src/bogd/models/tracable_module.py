import torch.nn as nn
import torch 


class TracableModule(nn.Module):
    tracing: bool = False
    
    def trace(self, tracing: bool = True):
        self.tracing = tracing
        self.trace_setting()
        for sub_module in self.children():
            if isinstance(sub_module, TracableModule):
                sub_module.trace(tracing)
    
    def trace_setting(self):
        """
        Default no operation.
        """
        pass

    @torch.no_grad()
    def trace_back(self):
        raise NotImplementedError()
