from collections.abc import Mapping
from typing import ClassVar, Self

import narwhals as nw
from pydantic import BaseModel, ConfigDict, model_validator

from assetlife.typing import TabularDataset


class BaseTabularDataset(BaseModel):
    """Base model for tabular data."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, extra="ignore"
    )

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        lengths = {name: len(v) for name, v in self if v is not None}
        if len(set(lengths.values())) > 1:
            msg = f"All columns must have the same length, got {lengths}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_dataset(
        cls, dataset: TabularDataset, *, columns: Mapping[str, str] | None = None
    ) -> Self:
        # pick columns, try to convert to df
        columns = columns or {}
        if isinstance(dataset, Mapping):
            renamed = {columns.get(k, k): v for k, v in dataset.items()}
            dict_dataset = {c: renamed[c] for c in cls.model_fields if c in renamed}
        else:
            try:
                df = nw.from_native(dataset, eager_only=True)
            except TypeError as err:
                msg = f"""
                Expected a DataFrame or a mapping of columns, got
                {type(dataset).__name__}
                """
                raise TypeError(msg) from err
            df = df.rename(dict(columns))
            dict_dataset = {
                c: df[c].to_numpy() for c in df.columns if c in cls.model_fields
            }
        return cls.model_validate(dict_dataset)
