from typing import List
from openai import OpenAI
import json
import hashlib


class Shot:
    def __init__(self, inputs: dict, outputs: dict, namespace: str):
        self.inputs = inputs
        self.outputs = outputs
        self.namespace = namespace
        self._id = self._generate_id()

    @property
    def id(self):
        return self._id

    @property
    def embedding(self):
        # Calculate the embedding of the shot
        pass

    def _generate_id(self):
        # Generate a unique ID for the shot using hash of the inputs
        inputs_str = json.dumps(self.inputs, sort_keys=True)
        return hashlib.sha256(inputs_str.encode()).hexdigest()



class BestShots:
    def __init__(self, api_key):
        self.api_key = api_key
        self.datasets = {}
        self.openai = OpenAI(api_key=api_key)

    def list(self, inputs, namespace, limit, format):
        # Get the list of the best shots
        pass

    def add(self, shots: List[Shot]):
        # Adding data to the best shots dataset
        for shot in shots:
            if not shot.namespace in self.datasets:
                self.datasets[shot.namespace] = {}
            if shot.id in self.datasets[shot.namespace]:
                continue

        pass

    def remove(self, inputs=None, namespace=None):
        # Removing data from the best shots dataset
        pass
