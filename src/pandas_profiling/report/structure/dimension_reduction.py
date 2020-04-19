from typing import Optional, List

from pandas_profiling.config import config
from pandas_profiling.report.presentation.abstract.renderable import Renderable
from pandas_profiling.report.presentation.core import (
    Sequence,
    HTML,
    Image,
    ToggleButton,
    Collapse,
)
from pandas_profiling.visualisation import plot

def get_dim_reduction_items(summary):
    image_format = config["plot"]["image_format"].get(str)
    dim_reduction = summary['dim_reduction']

    items = []
    for tech, array_2d_plot in dim_reduction.items():
        items.append(
            Image(
                array_2d_plot,
                image_format=image_format,
                alt=tech,
                anchor_id=tech,
                name=tech.upper()
            )
        )

    return Sequence(
        items,
        sequence_type='tabs',
        name='Dimension Reduction',
        anchor_id='dim_reduction'
    )
