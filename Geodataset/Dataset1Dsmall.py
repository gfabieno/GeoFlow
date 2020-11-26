# -*- coding: utf-8 -*-

from GeoFlow.Dataset import Dataset

class Dataset1Dsmall(Dataset):
    name = "Dataset1Dsmall"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True

        return model, acquire, inputs, outputs
