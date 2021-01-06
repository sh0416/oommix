class Collector:
    def __init__(self):
        self.activations = {}

    def create_hook_fn(self, name, fn):
        def hook(model, input, output):
            self.activations[name] = fn(output)
        return hook


def collect_representation(model, collector):
    target_names = ["embedding_norm"]
    target_names += ["encoder.%d.ff_norm" % i for i in range(12)]
    for name, module in model.named_modules():
        if name in target_names:
            hook_fn = collector.create_hook_fn(name, lambda x: x.detach().cpu().numpy())
            module.register_forward_hook(hook_fn)


def collect_attention(model, collector):
    target_names = ["encoder.%d.mhsa" % i for i in range(12)]
    for name, module in model.named_modules():
        if name in target_names:
            hook_fn = collector.create_hook_fn(name+"_attn", lambda x: x[1].detach().cpu().numpy())
            module.register_forward_hook(hook_fn)
