use super::*;

pub(super) const SIGNAL_KNOT_CLASS: &str = "fea.SignalKnot";
const SIGNAL_KNOT_FIELDS: &[&str] = &["knot"];
const KNOT_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "abscissa",
        BuiltinParamType::NumericScalar,
        "Finite time in seconds.",
    ),
    required(
        "value",
        BuiltinParamType::NumericArray,
        "Finite scalar or three-component SI ordinate.",
    ),
];
const JUMP_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "abscissa",
        BuiltinParamType::NumericScalar,
        "Finite time in seconds.",
    ),
    required(
        "leftLimit",
        BuiltinParamType::NumericArray,
        "Finite left-hand SI ordinate.",
    ),
    required(
        "rightLimit",
        BuiltinParamType::NumericArray,
        "Finite right-hand SI ordinate.",
    ),
    required(
        "continuity",
        BuiltinParamType::Any,
        "Typed fea.SignalPointContinuity value.",
    ),
];
const TIME_KNOT_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable signal identifier.",
    ),
    required(
        "knots",
        BuiltinParamType::Any,
        "Ordered row or column cell array of typed fea.SignalKnot values.",
    ),
    required(
        "interpolation",
        BuiltinParamType::Any,
        "Typed fea.SignalInterpolation policy.",
    ),
    required(
        "extrapolation",
        BuiltinParamType::Any,
        "Typed fea.SignalExtrapolation policy.",
    ),
];
const TIME_KNOT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The constructor consumes portable typed knot and policy objects; it has no direct numeric input.",
};
const CONTINUOUS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    integer_input("abscissa", "The exact integer is converted once to canonical binary64 seconds."),
    integer_input("value", "The exact scalar or vector components are converted once to canonical binary64 SI storage."),
];
const JUMP_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    integer_input("abscissa", "The exact integer is converted once to canonical binary64 seconds."),
    integer_input("leftLimit", "The exact scalar or vector components are converted once to canonical binary64 SI storage."),
    integer_input("rightLimit", "The exact scalar or vector components are converted once to canonical binary64 SI storage."),
];
const fn integer_input(name: &'static str, notes: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}
const CONTINUOUS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [knot_integer_capability(&CONTINUOUS_INTEGER_INPUTS)];
const JUMP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [knot_integer_capability(&JUMP_INTEGER_INPUTS)];
const fn knot_integer_capability(
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form: "typed signal knot",
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Numeric knot data is validated before explicit conversion to canonical binary64 storage.",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSignalKnot {
    abscissa: f64,
    value: RuntimeSignalKnotValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SignalKnotObject {
    knot: RuntimeSignalKnot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RuntimeSignalKnotValue {
    Continuous(Vec<f64>),
    Jump {
        left_limit: Vec<f64>,
        right_limit: Vec<f64>,
        continuity: runmat_analysis_core::SignalPointContinuity,
    },
}

fn raw_ordinate(builtin: &'static str, value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(_) | Value::Int(_) => Ok(vec![finite_numeric_scalar(builtin, value, label)?]),
        _ => {
            let values = finite_numeric_vector(builtin, value, label)?;
            if values.len() != 3 {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("{label} must be a scalar or three-component row or column vector"),
                ));
            }
            Ok(values)
        }
    }
}

fn knot_object(builtin: &'static str, knot: RuntimeSignalKnot) -> BuiltinResult<Value> {
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        SIGNAL_KNOT_CLASS,
        &SignalKnotObject { knot },
    )
}

const CONTINUOUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "knot = fea.SignalKnot.continuous(abscissa, value)",
    inputs: &KNOT_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const CONTINUOUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTINUOUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.SignalKnot.continuous",
    category = "fea",
    summary = "Create a continuous time-signal knot.",
    keywords = "fea,signal,knot,continuous,time,typed",
    descriptor(CONTINUOUS_DESCRIPTOR),
    integer_capabilities(CONTINUOUS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::signals::knots"
)]
pub async fn continuous(abscissa: Value, value: Value) -> BuiltinResult<Value> {
    let abscissa = finite_numeric_scalar("fea.SignalKnot.continuous", &abscissa, "abscissa")?;
    let value = raw_ordinate("fea.SignalKnot.continuous", &value, "value")?;
    knot_object(
        "fea.SignalKnot.continuous",
        RuntimeSignalKnot {
            abscissa: canonical_zero(abscissa),
            value: RuntimeSignalKnotValue::Continuous(value),
        },
    )
}

const JUMP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "knot = fea.SignalKnot.jump(abscissa, leftLimit, rightLimit, continuity)",
    inputs: &JUMP_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const JUMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &JUMP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.SignalKnot.jump",
    category = "fea",
    summary = "Create a time-signal jump with explicit one-sided limits and point continuity.",
    keywords = "fea,signal,knot,jump,discontinuity,time,typed",
    descriptor(JUMP_DESCRIPTOR),
    integer_capabilities(JUMP_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::signals::knots"
)]
pub async fn jump(
    abscissa: Value,
    left_limit: Value,
    right_limit: Value,
    continuity: Value,
) -> BuiltinResult<Value> {
    let builtin = "fea.SignalKnot.jump";
    let abscissa = finite_numeric_scalar(builtin, &abscissa, "abscissa")?;
    let left_limit = raw_ordinate(builtin, &left_limit, "leftLimit")?;
    let right_limit = raw_ordinate(builtin, &right_limit, "rightLimit")?;
    if left_limit.len() != right_limit.len() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "leftLimit and rightLimit must have the same shape",
        ));
    }
    knot_object(
        builtin,
        RuntimeSignalKnot {
            abscissa: canonical_zero(abscissa),
            value: RuntimeSignalKnotValue::Jump {
                left_limit,
                right_limit,
                continuity: super::super::signal_policies::point_continuity_from_value(
                    builtin,
                    &continuity,
                )?,
            },
        },
    )
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn runtime_knots(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<RuntimeSignalKnot>> {
    let Value::Cell(cell) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "knots must be a row or column cell array",
        ));
    };
    if cell.data.len() < 2 || (cell.rows != 1 && cell.cols != 1) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "knots must contain at least two values in a row or column cell array",
        ));
    }
    let knots = cell
        .data
        .iter()
        .map(|value| {
            object_payload::<SignalKnotObject>(builtin, value, SIGNAL_KNOT_CLASS).and_then(
                |object| {
                    validate_runtime_knot(&object.knot)
                        .map_err(|message| builtin_error(builtin, &ERROR_INPUT, message))?;
                    Ok(object.knot)
                },
            )
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if knots
        .windows(2)
        .any(|pair| pair[0].abscissa >= pair[1].abscissa)
    {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "knot abscissae must be strictly increasing",
        ));
    }
    if matches!(
        knots.first().map(|knot| &knot.value),
        Some(RuntimeSignalKnotValue::Jump { .. })
    ) || matches!(
        knots.last().map(|knot| &knot.value),
        Some(RuntimeSignalKnotValue::Jump { .. })
    ) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "jump knots must be interior to the signal domain",
        ));
    }
    let width = knot_width(&knots[0]);
    if knots.iter().any(|knot| knot_width(knot) != width) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "all knot ordinates must have the same scalar or vector shape",
        ));
    }
    Ok(knots)
}

fn knot_width(knot: &RuntimeSignalKnot) -> usize {
    match &knot.value {
        RuntimeSignalKnotValue::Continuous(value) => value.len(),
        RuntimeSignalKnotValue::Jump { left_limit, .. } => left_limit.len(),
    }
}

fn validate_runtime_knot(knot: &RuntimeSignalKnot) -> Result<(), String> {
    if !knot.abscissa.is_finite() {
        return Err("signal-knot abscissa must be finite".into());
    }
    if knot.abscissa == 0.0 && knot.abscissa.to_bits() != 0.0f64.to_bits() {
        return Err("signal-knot abscissa must encode zero canonically".into());
    }
    let valid_ordinate =
        |value: &[f64]| matches!(value.len(), 1 | 3) && value.iter().all(|entry| entry.is_finite());
    match &knot.value {
        RuntimeSignalKnotValue::Continuous(value) if valid_ordinate(value) => Ok(()),
        RuntimeSignalKnotValue::Jump {
            left_limit,
            right_limit,
            ..
        } if left_limit.len() == right_limit.len()
            && valid_ordinate(left_limit)
            && valid_ordinate(right_limit) =>
        {
            Ok(())
        }
        RuntimeSignalKnotValue::Jump {
            left_limit,
            right_limit,
            ..
        } if left_limit.len() != right_limit.len() => {
            Err("jump-knot limits must have the same shape".into())
        }
        _ => Err("signal-knot ordinates must be finite scalars or three-component vectors".into()),
    }
}

fn validate_signal_knot_object(object: &ObjectInstance) -> Result<(), String> {
    let payload = object_payload::<SignalKnotObject>(
        "fea.SignalKnot portable value",
        &Value::Object(object.clone()),
        SIGNAL_KNOT_CLASS,
    )
    .map_err(|error| error.to_string())?;
    validate_runtime_knot(&payload.knot)
}

fn typed_knots<Q>(
    builtin: &'static str,
    value: &Value,
    width: usize,
    convert: impl Fn(&[f64]) -> SignalOrdinate<Q>,
) -> BuiltinResult<Vec<SignalKnot<Q>>> {
    runtime_knots(builtin, value)?
        .into_iter()
        .map(|knot| {
            if knot_width(&knot) != width {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    if width == 1 {
                        "signal requires scalar knot ordinates"
                    } else {
                        "signal requires three-component knot ordinates"
                    },
                ));
            }
            Ok(SignalKnot {
                abscissa: knot.abscissa,
                value: match knot.value {
                    RuntimeSignalKnotValue::Continuous(value) => {
                        runmat_analysis_core::SignalKnotValue::Continuous(convert(&value))
                    }
                    RuntimeSignalKnotValue::Jump {
                        left_limit,
                        right_limit,
                        continuity,
                    } => runmat_analysis_core::SignalKnotValue::Jump {
                        left_limit: convert(&left_limit),
                        right_limit: convert(&right_limit),
                        continuity,
                    },
                },
            })
        })
        .collect()
}

fn time_knot_signal<Q>(
    id: String,
    shape: SignalShape,
    knots: Vec<SignalKnot<Q>>,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
    wrap: impl FnOnce(Signal<Q>) -> PhysicalSignal,
) -> PhysicalSignal {
    let sample_count = knots.len();
    let frame = matches!(shape, SignalShape::Vector3).then_some(CoordinateFrame::Global);
    wrap(Signal {
        signal_id: SignalId(id),
        shape,
        frame,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots,
            interpolation,
            extrapolation,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count,
        },
        provenance: SignalProvenance::Authored,
    })
}

macro_rules! time_knots_builtin {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $class:expr, $quantity:ty, $variant:path, scalar) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor { label: concat!("signal = ", $builtin, "(id, knots, interpolation, extrapolation)"), inputs: &TIME_KNOT_INPUTS, outputs: &OUT_ANY }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor { signatures: &$signatures, output_mode: BuiltinOutputMode::Fixed, completion_policy: BuiltinCompletionPolicy::Public, errors: &ERRORS };
        #[runtime_builtin(name = $builtin, category = "fea", summary = "Create a typed time history from continuous and jump knots.", keywords = "fea,signal,time,knot,jump,typed,si", descriptor($descriptor), integer_audit(TIME_KNOT_INTEGER_AUDIT), builtin_path = "crate::builtins::fea::signals::knots")]
        pub async fn $function(id: String, knots: Value, interpolation: Value, extrapolation: Value) -> BuiltinResult<Value> {
            let knots = typed_knots($builtin, &knots, 1, |value| SignalOrdinate::Real(SignalValue::Scalar(<$quantity>::new(value[0]))))?;
            let (interpolation, extrapolation) = super::time_samples::policies($builtin, &interpolation, &extrapolation)?;
            signal_object($builtin, $class, time_knot_signal(id, SignalShape::Scalar, knots, interpolation, extrapolation, $variant))
        }
    };
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $class:expr, $quantity:ty, $variant:path, vector) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor { label: concat!("signal = ", $builtin, "(id, knots, interpolation, extrapolation)"), inputs: &TIME_KNOT_INPUTS, outputs: &OUT_ANY }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor { signatures: &$signatures, output_mode: BuiltinOutputMode::Fixed, completion_policy: BuiltinCompletionPolicy::Public, errors: &ERRORS };
        #[runtime_builtin(name = $builtin, category = "fea", summary = "Create a typed time history from continuous and jump knots.", keywords = "fea,signal,time,knot,jump,vector,typed,si", descriptor($descriptor), integer_audit(TIME_KNOT_INTEGER_AUDIT), builtin_path = "crate::builtins::fea::signals::knots")]
        pub async fn $function(id: String, knots: Value, interpolation: Value, extrapolation: Value) -> BuiltinResult<Value> {
            let knots = typed_knots($builtin, &knots, 3, |value| SignalOrdinate::Real(SignalValue::Vector3([<$quantity>::new(value[0]), <$quantity>::new(value[1]), <$quantity>::new(value[2])])))?;
            let (interpolation, extrapolation) = super::time_samples::policies($builtin, &interpolation, &extrapolation)?;
            signal_object($builtin, $class, time_knot_signal(id, SignalShape::Vector3, knots, interpolation, extrapolation, $variant))
        }
    };
}

time_knots_builtin!(
    displacement,
    DISPLACEMENT_DESCRIPTOR,
    DISPLACEMENT_SIGNATURES,
    "fea.DisplacementSignal.fromTimeKnots",
    DISPLACEMENT_SIGNAL_CLASS,
    Displacement,
    PhysicalSignal::Displacement,
    vector
);
time_knots_builtin!(
    rotation,
    ROTATION_DESCRIPTOR,
    ROTATION_SIGNATURES,
    "fea.RotationSignal.fromTimeKnots",
    ROTATION_SIGNAL_CLASS,
    Rotation,
    PhysicalSignal::Rotation,
    vector
);
time_knots_builtin!(
    velocity,
    VELOCITY_DESCRIPTOR,
    VELOCITY_SIGNATURES,
    "fea.VelocitySignal.fromTimeKnots",
    VELOCITY_SIGNAL_CLASS,
    Velocity,
    PhysicalSignal::Velocity,
    vector
);
time_knots_builtin!(
    acceleration,
    ACCELERATION_DESCRIPTOR,
    ACCELERATION_SIGNATURES,
    "fea.AccelerationSignal.fromTimeKnots",
    ACCELERATION_SIGNAL_CLASS,
    Acceleration,
    PhysicalSignal::Acceleration,
    vector
);
time_knots_builtin!(
    angular_velocity,
    ANGULAR_VELOCITY_DESCRIPTOR,
    ANGULAR_VELOCITY_SIGNATURES,
    "fea.AngularVelocitySignal.fromTimeKnots",
    ANGULAR_VELOCITY_SIGNAL_CLASS,
    AngularVelocity,
    PhysicalSignal::AngularVelocity,
    vector
);
time_knots_builtin!(
    angular_acceleration,
    ANGULAR_ACCELERATION_DESCRIPTOR,
    ANGULAR_ACCELERATION_SIGNATURES,
    "fea.AngularAccelerationSignal.fromTimeKnots",
    ANGULAR_ACCELERATION_SIGNAL_CLASS,
    AngularAcceleration,
    PhysicalSignal::AngularAcceleration,
    vector
);
time_knots_builtin!(
    force,
    FORCE_DESCRIPTOR,
    FORCE_SIGNATURES,
    "fea.ForceSignal.fromTimeKnots",
    FORCE_SIGNAL_CLASS,
    Force,
    PhysicalSignal::Force,
    vector
);
time_knots_builtin!(
    moment,
    MOMENT_DESCRIPTOR,
    MOMENT_SIGNATURES,
    "fea.MomentSignal.fromTimeKnots",
    MOMENT_SIGNAL_CLASS,
    Moment,
    PhysicalSignal::Moment,
    vector
);
time_knots_builtin!(
    traction,
    TRACTION_DESCRIPTOR,
    TRACTION_SIGNATURES,
    "fea.TractionSignal.fromTimeKnots",
    TRACTION_SIGNAL_CLASS,
    Traction,
    PhysicalSignal::Traction,
    vector
);
time_knots_builtin!(
    pressure,
    PRESSURE_DESCRIPTOR,
    PRESSURE_SIGNATURES,
    "fea.PressureSignal.fromTimeKnots",
    PRESSURE_SIGNAL_CLASS,
    Pressure,
    PhysicalSignal::Pressure,
    scalar
);
time_knots_builtin!(
    body_force,
    BODY_FORCE_DESCRIPTOR,
    BODY_FORCE_SIGNATURES,
    "fea.BodyForceSignal.fromTimeKnots",
    BODY_FORCE_SIGNAL_CLASS,
    BodyForceDensity,
    PhysicalSignal::BodyForceDensity,
    vector
);
time_knots_builtin!(
    temperature,
    TEMPERATURE_DESCRIPTOR,
    TEMPERATURE_SIGNATURES,
    "fea.TemperatureSignal.fromTimeKnots",
    TEMPERATURE_SIGNAL_CLASS,
    Temperature,
    PhysicalSignal::Temperature,
    scalar
);
time_knots_builtin!(
    temperature_difference,
    TEMPERATURE_DIFFERENCE_DESCRIPTOR,
    TEMPERATURE_DIFFERENCE_SIGNATURES,
    "fea.TemperatureDifferenceSignal.fromTimeKnots",
    TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
    TemperatureDifference,
    PhysicalSignal::TemperatureDifference,
    scalar
);
time_knots_builtin!(
    heat_flux,
    HEAT_FLUX_DESCRIPTOR,
    HEAT_FLUX_SIGNATURES,
    "fea.HeatFluxSignal.fromTimeKnots",
    HEAT_FLUX_SIGNAL_CLASS,
    HeatFlux,
    PhysicalSignal::HeatFlux,
    scalar
);
time_knots_builtin!(
    heat_generation,
    HEAT_GENERATION_DESCRIPTOR,
    HEAT_GENERATION_SIGNATURES,
    "fea.HeatGenerationSignal.fromTimeKnots",
    HEAT_GENERATION_SIGNAL_CLASS,
    HeatGenerationDensity,
    PhysicalSignal::HeatGenerationDensity,
    scalar
);
time_knots_builtin!(
    electric_potential,
    ELECTRIC_POTENTIAL_DESCRIPTOR,
    ELECTRIC_POTENTIAL_SIGNATURES,
    "fea.ElectricPotentialSignal.fromTimeKnots",
    ELECTRIC_POTENTIAL_SIGNAL_CLASS,
    ElectricPotential,
    PhysicalSignal::ElectricPotential,
    scalar
);
time_knots_builtin!(
    electric_current,
    ELECTRIC_CURRENT_DESCRIPTOR,
    ELECTRIC_CURRENT_SIGNATURES,
    "fea.ElectricCurrentSignal.fromTimeKnots",
    ELECTRIC_CURRENT_SIGNAL_CLASS,
    ElectricCurrent,
    PhysicalSignal::ElectricCurrent,
    scalar
);
time_knots_builtin!(
    electric_charge,
    ELECTRIC_CHARGE_DESCRIPTOR,
    ELECTRIC_CHARGE_SIGNATURES,
    "fea.ElectricChargeSignal.fromTimeKnots",
    ELECTRIC_CHARGE_SIGNAL_CLASS,
    ElectricCharge,
    PhysicalSignal::ElectricCharge,
    scalar
);
time_knots_builtin!(
    current_density,
    CURRENT_DENSITY_DESCRIPTOR,
    CURRENT_DENSITY_SIGNATURES,
    "fea.CurrentDensitySignal.fromTimeKnots",
    CURRENT_DENSITY_SIGNAL_CLASS,
    ElectricCurrentDensity,
    PhysicalSignal::ElectricCurrentDensity,
    scalar
);
time_knots_builtin!(
    charge_density,
    CHARGE_DENSITY_DESCRIPTOR,
    CHARGE_DENSITY_SIGNATURES,
    "fea.ChargeDensitySignal.fromTimeKnots",
    CHARGE_DENSITY_SIGNAL_CLASS,
    ElectricChargeDensity,
    PhysicalSignal::ElectricChargeDensity,
    scalar
);
time_knots_builtin!(
    magnetic_vector_potential,
    MAGNETIC_VECTOR_POTENTIAL_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_SIGNATURES,
    "fea.MagneticVectorPotentialSignal.fromTimeKnots",
    MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
    MagneticVectorPotential,
    PhysicalSignal::MagneticVectorPotential,
    vector
);
time_knots_builtin!(
    mass_flow,
    MASS_FLOW_DESCRIPTOR,
    MASS_FLOW_SIGNATURES,
    "fea.MassFlowSignal.fromTimeKnots",
    MASS_FLOW_SIGNAL_CLASS,
    MassFlowRate,
    PhysicalSignal::MassFlowRate,
    scalar
);

pub(super) fn ensure_signal_knot_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(SIGNAL_KNOT_CLASS);
    PORTABLE_SCHEMA.call_once(|| {
        crate::execution::value_codec::register_immutable_value_class(
            crate::execution::value_codec::ImmutableValueClassSchema {
                class_name: SIGNAL_KNOT_CLASS,
                type_identity: "runmat.fea.signal-knot",
                schema_version: 1,
                allowed_fields: SIGNAL_KNOT_FIELDS,
                required_fields: SIGNAL_KNOT_FIELDS,
                validate: validate_signal_knot_object,
            },
        )
        .unwrap_or_else(|error| panic!("failed to register immutable signal knot: {error}"));
    });
    RUNTIME_CLASS.ensure(|| {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: SIGNAL_KNOT_CLASS.into(),
            parent: None,
            properties: immutable_properties(SIGNAL_KNOT_FIELDS),
            methods: HashMap::from([
                (
                    "continuous".into(),
                    static_method("continuous", "fea.SignalKnot.continuous"),
                ),
                ("jump".into(), static_method("jump", "fea.SignalKnot.jump")),
            ]),
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn portable_knots_normalize_authored_zero_and_reject_noncanonical_payloads() {
        ensure_signal_knot_class_registered();
        let value = block_on(continuous(Value::Num(-0.0), Value::Num(1.0))).unwrap();
        let payload: SignalKnotObject = object_payload("test", &value, SIGNAL_KNOT_CLASS).unwrap();
        assert_eq!(payload.knot.abscissa.to_bits(), 0.0f64.to_bits());
        assert!(crate::execution::value_codec::encode_inline_value(&value).is_ok());

        let malformed = knot_object(
            "test",
            RuntimeSignalKnot {
                abscissa: -0.0,
                value: RuntimeSignalKnotValue::Continuous(vec![1.0]),
            },
        )
        .unwrap();
        assert!(crate::execution::value_codec::encode_inline_value(&malformed).is_err());
    }
}
