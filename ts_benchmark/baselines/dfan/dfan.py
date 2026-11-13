from ts_benchmark.baselines.dfan.models.dfan_model import DFANModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    # "e_layers": 2,
    # "n_heads": 8,
    # "output_attention": 0,
    # "patch_len": 16,
    # "dropout": 0.1,
    # "fc_dropout": 0.2,
    # "moving_avg": 25,
    # "batch_size": 256,
    # "lradj": "type1",
    # "lr": 0.02,
    # "num_epochs": 20,
    # "num_workers": 0,
    # "loss": "MAE",
    # "patience": 3,
    # "num_experts": 2,
    "noisy_gating": True,
    # "k": 1,
    # "hidden_size": 256,
    "k_ratio": 0.5,
}


class DFAN(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "DFAN"

    def _init_model(self):
        return DFANModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, loss_importance = self.model(input)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss
