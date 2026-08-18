from pydantic import BaseModel, Field


class Example(BaseModel):
    tokens_acts_list: list[float]
    train_token_ind: int
    is_repeated_datapoint: bool
    tokens: list[str]
    image_references: list[str] = Field(default_factory=list)
    example_id: str | None = None


class ExamplesQuantile(BaseModel):
    quantile_name: str
    examples: list[Example]


class Model(BaseModel):
    transcoder_id: str
    index: int
    examples_quantiles: list[ExamplesQuantile]
    top_logits: list[str]
    bottom_logits: list[str]
    act_min: float
    act_max: float
    quantile_values: list[float] = Field(default_factory=list)
    histogram: list[float] = Field(default_factory=list)
    activation_frequency: float
    firing_count: int | None = None
    isDead: bool = False
