import json
from typing import Union


class Predicates:
    """Predicados formais usados na saída OutputType.PREDICATES."""

    @staticmethod
    def percentile(feature: str,
                   percentile: Union[int, float],
                   lower: Union[int, float],
                   upper: Union[int, float]):
        return f"<{feature}, {percentile}-between, [{lower}, {upper}]>"

    @staticmethod
    def between(feature: str, lower: Union[int, float], upper: Union[int, float]):
        return f"<{feature}, between, [{lower}, {upper}]>"

    @staticmethod
    def contains(feature: str, value: Union[int, str, list]):
        return f"<{feature}, contains, {json.dumps(value)}>"
