use super::*;

#[runtime_builtin(
    name = "fea.SpectrumAxis.frequencyHertz",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn frequency_axis(values: Value) -> BuiltinResult<Value> {
    let values = finite_numeric_vector("fea.SpectrumAxis.frequencyHertz", &values, "values")?;
    validate_positive_axis("fea.SpectrumAxis.frequencyHertz", &values)?;
    typed_value(
        "fea.SpectrumAxis.frequencyHertz",
        AXIS_CLASS,
        SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(values)),
    )
}

#[runtime_builtin(
    name = "fea.SpectrumAxis.periodSeconds",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters"
)]
pub async fn period_axis(values: Value) -> BuiltinResult<Value> {
    let values = finite_numeric_vector("fea.SpectrumAxis.periodSeconds", &values, "values")?;
    validate_positive_axis("fea.SpectrumAxis.periodSeconds", &values)?;
    typed_value(
        "fea.SpectrumAxis.periodSeconds",
        AXIS_CLASS,
        SpectrumAbscissa::PeriodSeconds(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Time,
            _,
        >::new(values)),
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
    "fea.SpectrumDamping.ratio",
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
    "fea.SpectrumDamping.qualityFactor",
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

#[runtime_builtin(
    name = "fea.SpectrumDirection.new",
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.DirectionalCombination.percentage",
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.ModalTruncation.new",
    binding_variant = "default",
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
    "fea.ModalTruncationCorrection.missingMass",
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
    "fea.RigidResponse.lindleyYow",
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

#[runtime_builtin(
    name = "fea.RigidResponse.gupta",
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.SpectrumConstruction.authoredDesign",
    binding_variant = "default",
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
    ($function:ident, $builtin:literal, $method:expr) => {
        #[runtime_builtin(name = $builtin, binding_variant = "default", builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters")]
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
    "fea.SpectrumConstruction.fromAccelerationHistoryExact",
    ShockResponseSpectrumMethod::ExactPiecewiseLinear
);
derived_construction!(
    derived_recursive,
    "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
    ShockResponseSpectrumMethod::RecursiveFilter
);

#[runtime_builtin(
    name = "fea.SpectrumEvidence.bounded",
    binding_variant = "default",
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
