# -*- coding: utf-8 -*-

from GeoFlow.GeoDataset import GeoDataset


class Dataset1Dsmall(GeoDataset):
    name = "Dataset1Dsmall"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True

        return model, acquire, inputs, outputs


if __name__ == "__main__":
    dataset = Dataset1Dsmall()
    dataset.model.animated_dataset()
