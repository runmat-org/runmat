use super::*;

noarg_policy!(
    response_relative_displacement,
    RESPONSE_RD_DESCRIPTOR,
    RESPONSE_RD_SIGNATURES,
    "fea.SpectrumResponse.relativeDisplacement",
    "response = fea.SpectrumResponse.relativeDisplacement()",
    "Select relative-displacement response.",
    RESPONSE_CLASS,
    SpectrumResponse::RelativeDisplacement
);
noarg_policy!(
    response_relative_velocity,
    RESPONSE_RV_DESCRIPTOR,
    RESPONSE_RV_SIGNATURES,
    "fea.SpectrumResponse.relativeVelocity",
    "response = fea.SpectrumResponse.relativeVelocity()",
    "Select relative-velocity response.",
    RESPONSE_CLASS,
    SpectrumResponse::RelativeVelocity
);
noarg_policy!(
    response_relative_acceleration,
    RESPONSE_RA_DESCRIPTOR,
    RESPONSE_RA_SIGNATURES,
    "fea.SpectrumResponse.relativeAcceleration",
    "response = fea.SpectrumResponse.relativeAcceleration()",
    "Select relative-acceleration response.",
    RESPONSE_CLASS,
    SpectrumResponse::RelativeAcceleration
);
noarg_policy!(
    response_absolute_acceleration,
    RESPONSE_AA_DESCRIPTOR,
    RESPONSE_AA_SIGNATURES,
    "fea.SpectrumResponse.absoluteAcceleration",
    "response = fea.SpectrumResponse.absoluteAcceleration()",
    "Select absolute-acceleration response.",
    RESPONSE_CLASS,
    SpectrumResponse::AbsoluteAcceleration
);
noarg_policy!(
    response_pseudo_velocity,
    RESPONSE_PV_DESCRIPTOR,
    RESPONSE_PV_SIGNATURES,
    "fea.SpectrumResponse.pseudoVelocity",
    "response = fea.SpectrumResponse.pseudoVelocity()",
    "Select pseudo-velocity response.",
    RESPONSE_CLASS,
    SpectrumResponse::PseudoVelocity
);
noarg_policy!(
    response_pseudo_acceleration,
    RESPONSE_PA_DESCRIPTOR,
    RESPONSE_PA_SIGNATURES,
    "fea.SpectrumResponse.pseudoAcceleration",
    "response = fea.SpectrumResponse.pseudoAcceleration()",
    "Select pseudo-acceleration response.",
    RESPONSE_CLASS,
    SpectrumResponse::PseudoAcceleration
);
noarg_policy!(
    envelope_positive,
    ENVELOPE_POS_DESCRIPTOR,
    ENVELOPE_POS_SIGNATURES,
    "fea.SpectrumEnvelope.positive",
    "envelope = fea.SpectrumEnvelope.positive()",
    "Select the positive response envelope.",
    ENVELOPE_CLASS,
    SpectrumEnvelope::Positive
);
noarg_policy!(
    envelope_negative,
    ENVELOPE_NEG_DESCRIPTOR,
    ENVELOPE_NEG_SIGNATURES,
    "fea.SpectrumEnvelope.negative",
    "envelope = fea.SpectrumEnvelope.negative()",
    "Select the negative response envelope.",
    ENVELOPE_CLASS,
    SpectrumEnvelope::Negative
);
noarg_policy!(
    envelope_maximax,
    ENVELOPE_MAX_DESCRIPTOR,
    ENVELOPE_MAX_SIGNATURES,
    "fea.SpectrumEnvelope.maximax",
    "envelope = fea.SpectrumEnvelope.maximax()",
    "Select the maximax response envelope.",
    ENVELOPE_CLASS,
    SpectrumEnvelope::Maximax
);
noarg_policy!(
    interpolation_frequency,
    INTERP_FREQ_DESCRIPTOR,
    INTERP_FREQ_SIGNATURES,
    "fea.SpectrumInterpolation.linearFrequency",
    "interpolation = fea.SpectrumInterpolation.linearFrequency()",
    "Select linear interpolation in frequency.",
    INTERPOLATION_CLASS,
    SpectrumInterpolation::LinearFrequency
);
noarg_policy!(
    interpolation_period,
    INTERP_PERIOD_DESCRIPTOR,
    INTERP_PERIOD_SIGNATURES,
    "fea.SpectrumInterpolation.linearPeriod",
    "interpolation = fea.SpectrumInterpolation.linearPeriod()",
    "Select linear interpolation in period.",
    INTERPOLATION_CLASS,
    SpectrumInterpolation::LinearPeriod
);
noarg_policy!(
    interpolation_loglog,
    INTERP_LOG_DESCRIPTOR,
    INTERP_LOG_SIGNATURES,
    "fea.SpectrumInterpolation.logLog",
    "interpolation = fea.SpectrumInterpolation.logLog()",
    "Select log-log spectrum interpolation.",
    INTERPOLATION_CLASS,
    SpectrumInterpolation::LogLog
);
noarg_policy!(
    extrapolation_reject,
    EXTRAP_REJECT_DESCRIPTOR,
    EXTRAP_REJECT_SIGNATURES,
    "fea.SpectrumExtrapolation.reject",
    "extrapolation = fea.SpectrumExtrapolation.reject()",
    "Reject response-spectrum extrapolation.",
    EXTRAPOLATION_CLASS,
    SpectrumExtrapolation::Reject
);
noarg_policy!(
    extrapolation_hold,
    EXTRAP_HOLD_DESCRIPTOR,
    EXTRAP_HOLD_SIGNATURES,
    "fea.SpectrumExtrapolation.holdEndpoint",
    "extrapolation = fea.SpectrumExtrapolation.holdEndpoint()",
    "Hold the response-spectrum endpoint when extrapolating.",
    EXTRAPOLATION_CLASS,
    SpectrumExtrapolation::HoldEndpoint
);
noarg_policy!(
    modal_srss,
    MODAL_SRSS_DESCRIPTOR,
    MODAL_SRSS_SIGNATURES,
    "fea.ModalCombination.srss",
    "combination = fea.ModalCombination.srss()",
    "Select square-root-of-sum-of-squares modal combination.",
    MODAL_COMBINATION_CLASS,
    ModalCombination::Srss
);
noarg_policy!(
    modal_cqc,
    MODAL_CQC_DESCRIPTOR,
    MODAL_CQC_SIGNATURES,
    "fea.ModalCombination.completeQuadratic",
    "combination = fea.ModalCombination.completeQuadratic()",
    "Select complete-quadratic modal combination.",
    MODAL_COMBINATION_CLASS,
    ModalCombination::CompleteQuadraticCombination
);
noarg_policy!(
    modal_ten_percent,
    MODAL_TEN_DESCRIPTOR,
    MODAL_TEN_SIGNATURES,
    "fea.ModalCombination.tenPercent",
    "combination = fea.ModalCombination.tenPercent()",
    "Select the ten-percent modal combination rule.",
    MODAL_COMBINATION_CLASS,
    ModalCombination::TenPercent
);
noarg_policy!(
    modal_absolute_sum,
    MODAL_ABS_DESCRIPTOR,
    MODAL_ABS_SIGNATURES,
    "fea.ModalCombination.absoluteSum",
    "combination = fea.ModalCombination.absoluteSum()",
    "Select absolute modal summation.",
    MODAL_COMBINATION_CLASS,
    ModalCombination::AbsoluteSum
);
noarg_policy!(
    directional_srss,
    DIR_SRSS_DESCRIPTOR,
    DIR_SRSS_SIGNATURES,
    "fea.DirectionalCombination.srss",
    "combination = fea.DirectionalCombination.srss()",
    "Select SRSS directional combination.",
    DIRECTIONAL_COMBINATION_CLASS,
    DirectionalCombination::Srss
);
noarg_policy!(
    directional_absolute_sum,
    DIR_ABS_DESCRIPTOR,
    DIR_ABS_SIGNATURES,
    "fea.DirectionalCombination.absoluteSum",
    "combination = fea.DirectionalCombination.absoluteSum()",
    "Select absolute directional summation.",
    DIRECTIONAL_COMBINATION_CLASS,
    DirectionalCombination::AbsoluteSum
);
noarg_policy!(
    truncation_correction_none,
    CORRECTION_NONE_DESCRIPTOR,
    CORRECTION_NONE_SIGNATURES,
    "fea.ModalTruncationCorrection.none",
    "correction = fea.ModalTruncationCorrection.none()",
    "Disable high-frequency truncation correction.",
    TRUNCATION_CORRECTION_CLASS,
    ModalTruncationCorrection::None
);
noarg_policy!(
    truncation_correction_residual_mode,
    CORRECTION_RESIDUAL_DESCRIPTOR,
    CORRECTION_RESIDUAL_SIGNATURES,
    "fea.ModalTruncationCorrection.residualMode",
    "correction = fea.ModalTruncationCorrection.residualMode()",
    "Request a load-dependent residual mode.",
    TRUNCATION_CORRECTION_CLASS,
    ModalTruncationCorrection::ResidualMode
);
noarg_policy!(
    rigid_response_disabled,
    RIGID_DISABLED_DESCRIPTOR,
    RIGID_DISABLED_SIGNATURES,
    "fea.RigidResponse.disabled",
    "policy = fea.RigidResponse.disabled()",
    "Disable rigid-response separation.",
    RIGID_RESPONSE_CLASS,
    RigidResponsePolicy::Disabled
);
