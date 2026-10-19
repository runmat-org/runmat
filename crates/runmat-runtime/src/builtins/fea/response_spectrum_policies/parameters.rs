use super::*;

const AXIS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "axis = fea.SpectrumAxis.frequencyHertz(values)",
    inputs: &VECTOR_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const AXIS_FREQUENCY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AXIS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.SpectrumAxis.frequencyHertz",
    category = "fea",
    summary = "Create a frequency response-spectrum axis in hertz.",
    keywords = "fea,response,spectrum,axis,frequency,typed",
    descriptor(
        crate::builtins::fea::response_spectrum_policies::parameters::AXIS_FREQUENCY_DESCRIPTOR
    ),
    integer_capabilities(AXIS_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn frequency_axis(values: Value) -> BuiltinResult<Value> {
    let values = finite_numeric_vector("fea.SpectrumAxis.frequencyHertz", &values, "values")?;
    validate_positive_axis("fea.SpectrumAxis.frequencyHertz", &values)?;
    typed_value(
        "fea.SpectrumAxis.frequencyHertz",
        AXIS_CLASS,
        SpectrumAbscissa::FrequencyHertz(values),
    )
}

const PERIOD_AXIS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "axis = fea.SpectrumAxis.periodSeconds(values)",
    inputs: &VECTOR_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const AXIS_PERIOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERIOD_AXIS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.SpectrumAxis.periodSeconds",
    category = "fea",
    summary = "Create a response-spectrum period axis in seconds.",
    keywords = "fea,response,spectrum,axis,period,typed",
    descriptor(
        crate::builtins::fea::response_spectrum_policies::parameters::AXIS_PERIOD_DESCRIPTOR
    ),
    integer_capabilities(AXIS_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn period_axis(values: Value) -> BuiltinResult<Value> {
    let values = finite_numeric_vector("fea.SpectrumAxis.periodSeconds", &values, "values")?;
    validate_positive_axis("fea.SpectrumAxis.periodSeconds", &values)?;
    typed_value(
        "fea.SpectrumAxis.periodSeconds",
        AXIS_CLASS,
        SpectrumAbscissa::PeriodSeconds(values),
    )
}

fn validate_positive_axis(builtin: &'static str, values: &[f64]) -> BuiltinResult<()> {
    if values.is_empty()
        || values.iter().any(|value| *value <= 0.0)
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "spectrum axis must be positive and strictly increasing",
        ))
    } else {
        Ok(())
    }
}

unary_policy!(
    damping_ratio,
    DAMPING_RATIO_DESCRIPTOR,
    DAMPING_RATIO_SIGNATURES,
    "fea.SpectrumDamping.ratio",
    "damping = fea.SpectrumDamping.ratio(value)",
    "Create a response-spectrum damping ratio.",
    DAMPING_CLASS,
    |value: f64| if (0.0..1.0).contains(&value) {
        Ok(SpectrumDamping::Ratio(value))
    } else {
        Err(builtin_error(
            "fea.SpectrumDamping.ratio",
            &ERROR_INPUT,
            "damping ratio must be at least zero and less than one",
        ))
    }
);
unary_policy!(
    damping_quality_factor,
    DAMPING_QUALITY_DESCRIPTOR,
    DAMPING_QUALITY_SIGNATURES,
    "fea.SpectrumDamping.qualityFactor",
    "damping = fea.SpectrumDamping.qualityFactor(value)",
    "Create a response-spectrum quality factor.",
    DAMPING_CLASS,
    |value: f64| if value > 0.0 {
        Ok(SpectrumDamping::QualityFactor(value))
    } else {
        Err(builtin_error(
            "fea.SpectrumDamping.qualityFactor",
            &ERROR_INPUT,
            "quality factor must be positive",
        ))
    }
);

// The remaining value-bearing constructors and registration are kept below so the closed no-argument
// policy vocabulary above stays mechanically auditable.

const DIRECTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "direction = fea.SpectrumDirection.new(frame, vector)",
    inputs: &DIRECTION_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const DIRECTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIRECTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.SpectrumDirection.new",
    category = "fea",
    summary = "Create a typed response-spectrum direction.",
    keywords = "fea,response,spectrum,direction,frame,typed",
    descriptor(crate::builtins::fea::response_spectrum_policies::parameters::DIRECTION_DESCRIPTOR),
    integer_capabilities(DIRECTION_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn spectrum_direction(frame: Value, vector: Value) -> BuiltinResult<Value> {
    let vector = finite_numeric_vector3("fea.SpectrumDirection.new", &vector, "vector")?;
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(builtin_error(
            "fea.SpectrumDirection.new",
            &ERROR_INPUT,
            "direction vector must be nonzero",
        ));
    }
    typed_value(
        "fea.SpectrumDirection.new",
        DIRECTION_CLASS,
        SpectrumDirection {
            frame: coordinate_frame_from_value("fea.SpectrumDirection.new", &frame)?,
            vector: vector.map(|value| value / norm),
        },
    )
}

const PERCENTAGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "combination = fea.DirectionalCombination.percentage(primary, secondary)",
    inputs: &PERCENTAGE_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const PERCENTAGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERCENTAGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.DirectionalCombination.percentage",
    category = "fea",
    summary = "Create a directional percentage combination rule.",
    keywords = "fea,response,spectrum,directional,percentage,typed",
    descriptor(
        crate::builtins::fea::response_spectrum_policies::parameters::PERCENTAGE_DESCRIPTOR
    ),
    integer_capabilities(PERCENTAGE_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn directional_percentage(primary: Value, secondary: Value) -> BuiltinResult<Value> {
    let primary =
        finite_numeric_scalar("fea.DirectionalCombination.percentage", &primary, "primary")?;
    let secondary = finite_numeric_scalar(
        "fea.DirectionalCombination.percentage",
        &secondary,
        "secondary",
    )?;
    if !(0.0..=1.0).contains(&primary) || !(0.0..=1.0).contains(&secondary) || primary < secondary {
        return Err(builtin_error(
            "fea.DirectionalCombination.percentage",
            &ERROR_INPUT,
            "fractions must lie in [0, 1] with primary not less than secondary",
        ));
    }
    typed_value(
        "fea.DirectionalCombination.percentage",
        DIRECTIONAL_COMBINATION_CLASS,
        DirectionalCombination::PercentageRule { primary, secondary },
    )
}

const TRUNCATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "policy = fea.ModalTruncation.new(minimumMassRatio, correction)",
    inputs: &TRUNCATION_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const TRUNCATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRUNCATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.ModalTruncation.new",
    category = "fea",
    summary = "Create a typed response-spectrum modal truncation policy.",
    keywords = "fea,response,spectrum,modal,truncation,typed",
    descriptor(
        crate::builtins::fea::response_spectrum_policies::parameters::TRUNCATION_DESCRIPTOR
    ),
    integer_capabilities(TRUNCATION_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn modal_truncation(
    minimum_mass_ratio: Value,
    correction: Value,
) -> BuiltinResult<Value> {
    let ratio = finite_numeric_scalar(
        "fea.ModalTruncation.new",
        &minimum_mass_ratio,
        "minimumMassRatio",
    )?;
    if !(0.0..=1.0).contains(&ratio) {
        return Err(builtin_error(
            "fea.ModalTruncation.new",
            &ERROR_INPUT,
            "minimum mass ratio must lie in [0, 1]",
        ));
    }
    typed_value(
        "fea.ModalTruncation.new",
        TRUNCATION_CLASS,
        ModalTruncationPolicy {
            minimum_cumulative_mass_ratio: ratio,
            correction: typed_value_from(
                "fea.ModalTruncation.new",
                &correction,
                TRUNCATION_CORRECTION_CLASS,
            )?,
        },
    )
}

unary_policy!(
    truncation_missing_mass,
    CORRECTION_MISSING_MASS_DESCRIPTOR,
    CORRECTION_MISSING_MASS_SIGNATURES,
    "fea.ModalTruncationCorrection.missingMass",
    "correction = fea.ModalTruncationCorrection.missingMass(zeroPeriodAcceleration)",
    "Request missing-mass correction with an explicit zero-period acceleration.",
    TRUNCATION_CORRECTION_CLASS,
    |value| if value >= 0.0 {
        Ok(ModalTruncationCorrection::MissingMass {
            zero_period_acceleration: value,
        })
    } else {
        Err(builtin_error(
            "fea.ModalTruncationCorrection.missingMass",
            &ERROR_INPUT,
            "zero-period acceleration must be nonnegative",
        ))
    }
);

unary_policy!(
    rigid_response_lindley_yow,
    RIGID_LINDLEY_DESCRIPTOR,
    RIGID_LINDLEY_SIGNATURES,
    "fea.RigidResponse.lindleyYow",
    "policy = fea.RigidResponse.lindleyYow(zeroPeriodAcceleration)",
    "Use Lindley-Yow periodic/rigid separation with an explicit ZPA.",
    RIGID_RESPONSE_CLASS,
    |value| if value >= 0.0 {
        Ok(RigidResponsePolicy::LindleyYow {
            zero_period_acceleration: value,
        })
    } else {
        Err(builtin_error(
            "fea.RigidResponse.lindleyYow",
            &ERROR_INPUT,
            "zero-period acceleration must be nonnegative",
        ))
    }
);

const GUPTA_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "periodicFrequencyHertz",
        BuiltinParamType::NumericScalar,
        "Upper bound of the wholly periodic frequency region.",
    ),
    required(
        "rigidFrequencyHertz",
        BuiltinParamType::NumericScalar,
        "Lower bound of the wholly rigid frequency region.",
    ),
];
const GUPTA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "policy = fea.RigidResponse.gupta(periodicFrequencyHertz, rigidFrequencyHertz)",
    inputs: &GUPTA_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const GUPTA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GUPTA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.RigidResponse.gupta",
    category = "fea",
    summary = "Use Gupta periodic/rigid response separation.",
    keywords = "fea,response,spectrum,rigid,gupta,typed",
    descriptor(crate::builtins::fea::response_spectrum_policies::parameters::GUPTA_DESCRIPTOR),
    integer_capabilities(GUPTA_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn rigid_response_gupta(periodic: Value, rigid: Value) -> BuiltinResult<Value> {
    let periodic = finite_numeric_scalar(
        "fea.RigidResponse.gupta",
        &periodic,
        "periodicFrequencyHertz",
    )?;
    let rigid = finite_numeric_scalar("fea.RigidResponse.gupta", &rigid, "rigidFrequencyHertz")?;
    if periodic <= 0.0 || rigid <= periodic {
        return Err(builtin_error(
            "fea.RigidResponse.gupta",
            &ERROR_INPUT,
            "frequencies must be positive with rigid frequency greater than periodic frequency",
        ));
    }
    typed_value(
        "fea.RigidResponse.gupta",
        RIGID_RESPONSE_CLASS,
        RigidResponsePolicy::Gupta {
            periodic_frequency_hz: periodic,
            rigid_frequency_hz: rigid,
        },
    )
}

const CONSTRUCTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "construction = fea.SpectrumConstruction.authoredDesign()",
    inputs: &NO_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const CONSTRUCTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONSTRUCTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.SpectrumConstruction.authoredDesign",
    category = "fea",
    summary = "Mark a response spectrum as an authored design spectrum.",
    keywords = "fea,response,spectrum,construction,design,typed",
    descriptor(
        crate::builtins::fea::response_spectrum_policies::parameters::CONSTRUCTION_DESCRIPTOR
    ),
    integer_audit(POLICY_AUDIT),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn authored_design() -> BuiltinResult<Value> {
    typed_value(
        "fea.SpectrumConstruction.authoredDesign",
        CONSTRUCTION_CLASS,
        SpectrumConstruction::AuthoredDesign,
    )
}

macro_rules! derived_construction {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $method:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor { label: concat!("construction = ", $builtin, "(accelerationSignal)"), inputs: &CONSTRUCTION_SIGNAL_INPUTS, outputs: &OUT_ANY }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor { signatures: &$signatures, output_mode: BuiltinOutputMode::Fixed, completion_policy: BuiltinCompletionPolicy::Public, errors: &ERRORS };
        #[runtime_builtin(name = $builtin, category = "fea", summary = "Derive a response spectrum from a typed acceleration history.", keywords = "fea,response,spectrum,construction,acceleration,typed", descriptor($descriptor), integer_audit(POLICY_AUDIT), builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters")]
        pub async fn $function(signal: Value) -> BuiltinResult<Value> {
            let signal = super::signals::physical_signal_from_value($builtin, &signal)?;
            if signal.quantity() != runmat_analysis_core::PhysicalQuantityKind::Acceleration || signal.domain() != runmat_analysis_core::SignalDomainKind::Time {
                return Err(builtin_error($builtin, &ERROR_INPUT, "construction requires a time-domain acceleration signal"));
            }
            typed_value($builtin, CONSTRUCTION_CLASS, SpectrumConstruction::DerivedFromAccelerationHistory { source_signal_id: signal.signal_id().clone(), method: $method })
        }
    };
}
derived_construction!(
    derived_exact,
    DERIVED_EXACT_DESCRIPTOR,
    DERIVED_EXACT_SIGNATURES,
    "fea.SpectrumConstruction.fromAccelerationHistoryExact",
    ShockResponseSpectrumMethod::ExactPiecewiseLinear
);
derived_construction!(
    derived_recursive,
    DERIVED_RECURSIVE_DESCRIPTOR,
    DERIVED_RECURSIVE_SIGNATURES,
    "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
    ShockResponseSpectrumMethod::RecursiveFilter
);

const EVIDENCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label:
        "evidence = fea.SpectrumEvidence.bounded(samplingAdequacyRatio, maximumInterpolationError)",
    inputs: &EVIDENCE_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const EVIDENCE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EVIDENCE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
#[runtime_builtin(
    name = "fea.SpectrumEvidence.bounded",
    category = "fea",
    summary = "Create response-spectrum sampling and interpolation evidence.",
    keywords = "fea,response,spectrum,evidence,error,typed",
    descriptor(crate::builtins::fea::response_spectrum_policies::parameters::EVIDENCE_DESCRIPTOR),
    integer_capabilities(EVIDENCE_NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn spectrum_evidence(
    sampling_ratio: Value,
    maximum_error: Value,
) -> BuiltinResult<Value> {
    let sampling_adequacy_ratio = finite_numeric_scalar(
        "fea.SpectrumEvidence.bounded",
        &sampling_ratio,
        "samplingAdequacyRatio",
    )?;
    let max_interpolation_error = finite_numeric_scalar(
        "fea.SpectrumEvidence.bounded",
        &maximum_error,
        "maximumInterpolationError",
    )?;
    if !(0.0..=1.0).contains(&sampling_adequacy_ratio) || max_interpolation_error < 0.0 {
        return Err(builtin_error("fea.SpectrumEvidence.bounded", &ERROR_INPUT, "sampling adequacy must be a fraction in [0, 1] and interpolation error must be nonnegative"));
    }
    typed_value(
        "fea.SpectrumEvidence.bounded",
        EVIDENCE_CLASS,
        SpectrumEvidence {
            sampling_adequacy_ratio,
            max_interpolation_error: Some(max_interpolation_error),
        },
    )
}
