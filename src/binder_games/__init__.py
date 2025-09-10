from .optimizers import (
    minmax_logits,
    alternating_br_logits,
    stackelberg_logits,
    extragradient_minmax_logits,
)

from .builders import (
    build_minmax_phase,
    build_stackelberg_phase,
    build_multi_adversary_phase,
)

from .analyzers import (
    saddle_gap_estimate,
    value_components,
    decode_sequences_xy,
    off_target_weights_summary,
)

from .validators import (
    gap_threshold,
    worst_case_threshold,
)

from .transforms import (
    temperature_on_logits_xy,
    gradient_normalizer_xy,
    hard_one_hot_xy,
)

from .losses import (
    make_minmax_loss,
    make_multi_adversary_loss,
    make_dro_loss,
    worst_case_panel_loss,
    make_dro_two_player_loss,
    bayesian_stackelberg_closed_form,
    bayesian_stackelberg_two_player_loss,
)

__all__ = [
    # optimizers
    "minmax_logits",
    "alternating_br_logits",
    "stackelberg_logits",
    "extragradient_minmax_logits",
    # builders
    "build_minmax_phase",
    "build_stackelberg_phase",
    "build_multi_adversary_phase",
    # analyzers
    "saddle_gap_estimate",
    "value_components",
    "decode_sequences_xy",
    "off_target_weights_summary",
    # validators
    "gap_threshold",
    "worst_case_threshold",
    # transforms
    "temperature_on_logits_xy",
    "gradient_normalizer_xy",
    "hard_one_hot_xy",
    # losses
    "make_minmax_loss",
    "make_multi_adversary_loss",
    "make_dro_loss",
    "worst_case_panel_loss",
    "make_dro_two_player_loss",
    "bayesian_stackelberg_closed_form",
    "bayesian_stackelberg_two_player_loss",
]


