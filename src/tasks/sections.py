"""
The luigi parameter that carries one validated configuration section into a task.

Structure:
1. SectionParameter
"""

import luigi
from pydantic import BaseModel


class SectionParameter(luigi.Parameter):
    """A task parameter holding one validated section of the configuration, an instance of the model it declares.

    luigi normalizes every constructor argument (normalize) and serializes the values for the task id (serialize).
    normalize returns an instance of the declared model unchanged, validates JSON text through model_validate_json and
    anything else through model_validate, which refuses an instance of another model and a mapping that does not fit,
    so a section passed under the wrong keyword never reaches a task. serialize is model_dump_json(); parse is
    model_validate_json.
    """

    def __init__(self, model: type[BaseModel], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def normalize(self, x):
        if isinstance(x, self.model):
            return x
        if isinstance(x, str):
            return self.model.model_validate_json(x)
        return self.model.model_validate(x)

    def serialize(self, x) -> str:
        return x.model_dump_json()

    def parse(self, x: str):
        return self.model.model_validate_json(x)
