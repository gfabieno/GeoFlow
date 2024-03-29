# -*- coding: utf-8 -*-

from DefinedDataset.Dataset1Dsmall import Dataset1Dsmall
from GeoFlow.GraphIO import make_output_from_shotgather


class Dataset1Dautoencode(Dataset1Dsmall):
    def set_dataset(self):
        model, acquire, inputs, _ = super().set_dataset()

        self.trainsize = 100
        self.validatesize = 0
        self.testsize = 50

        reconstructed = make_output_from_shotgather(inputs['shotgather'])
        outputs = {reconstructed.name: reconstructed}
        for name in inputs:
            inputs[name].train_on_shots = True
        for name in outputs:
            outputs[name].train_on_shots = True

        return model, acquire, inputs, outputs


if __name__ == "__main__":
    dataset = Dataset1Dautoencode()
    dataset.model.animated_dataset()
