use std::collections::HashSet;

use super::*;
use runmat_analysis_core::{
    GeneratorArgument, GeneratorCallable, GeneratorDomain, GeneratorEffectPolicy, SignalGenerator,
    SignalSamplingPolicy,
};
use runmat_execution::value::{CallableValue, InlineValue, ValuePayload};

const CALLABLE_INPUTS: [BuiltinParamDescriptor; 5] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable signal identifier.",
    ),
    required(
        "callable",
        BuiltinParamType::Any,
        "Semantically bound RunMat function handle or immutable closure.",
    ),
    required(
        "domain",
        BuiltinParamType::Any,
        "Typed fea.SignalDomain value.",
    ),
    required(
        "sampling",
        BuiltinParamType::Any,
        "Typed fea.SignalSampling value.",
    ),
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Optional Arguments cell array and typed Interpolation/Extrapolation policies.",
    },
];
const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Numeric generator arguments are immutable values inside an explicit cell array and retain exact integer kind; domain and sampling are typed objects.",
};

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

macro_rules! callable_signal_builtin {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $class:expr, $shape:expr, $variant:path
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!(
                "signal = ",
                $builtin,
                "(id, callable, domain, sampling, Name, Value)"
            ),
            inputs: &CALLABLE_INPUTS,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                    name = $builtin,
                    category = "fea",
                    summary = "Create a typed physical signal from a frozen RunMat callable.",
                    keywords = "fea,signal,function,closure,generator,typed",
                    descriptor($descriptor),
                    integer_audit(INTEGER_AUDIT),
                    builtin_path = "crate::builtins::fea::signals::callable"
                )]
        pub async fn $function(args: Vec<Value>) -> BuiltinResult<Value> {
            callable_signal($builtin, $class, $shape, $variant, args)
        }
    };
}

callable_signal_builtin!(
    displacement,
    DISPLACEMENT_DESCRIPTOR,
    DISPLACEMENT_SIGNATURES,
    "fea.DisplacementSignal.fromFunction",
    DISPLACEMENT_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Displacement
);
callable_signal_builtin!(
    rotation,
    ROTATION_DESCRIPTOR,
    ROTATION_SIGNATURES,
    "fea.RotationSignal.fromFunction",
    ROTATION_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Rotation
);
callable_signal_builtin!(
    velocity,
    VELOCITY_DESCRIPTOR,
    VELOCITY_SIGNATURES,
    "fea.VelocitySignal.fromFunction",
    VELOCITY_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Velocity
);
callable_signal_builtin!(
    acceleration,
    ACCELERATION_DESCRIPTOR,
    ACCELERATION_SIGNATURES,
    "fea.AccelerationSignal.fromFunction",
    ACCELERATION_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Acceleration
);
callable_signal_builtin!(
    angular_velocity,
    ANGULAR_VELOCITY_DESCRIPTOR,
    ANGULAR_VELOCITY_SIGNATURES,
    "fea.AngularVelocitySignal.fromFunction",
    ANGULAR_VELOCITY_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::AngularVelocity
);
callable_signal_builtin!(
    angular_acceleration,
    ANGULAR_ACCELERATION_DESCRIPTOR,
    ANGULAR_ACCELERATION_SIGNATURES,
    "fea.AngularAccelerationSignal.fromFunction",
    ANGULAR_ACCELERATION_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::AngularAcceleration
);
callable_signal_builtin!(
    force,
    FORCE_DESCRIPTOR,
    FORCE_SIGNATURES,
    "fea.ForceSignal.fromFunction",
    FORCE_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Force
);
callable_signal_builtin!(
    moment,
    MOMENT_DESCRIPTOR,
    MOMENT_SIGNATURES,
    "fea.MomentSignal.fromFunction",
    MOMENT_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Moment
);
callable_signal_builtin!(
    traction,
    TRACTION_DESCRIPTOR,
    TRACTION_SIGNATURES,
    "fea.TractionSignal.fromFunction",
    TRACTION_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::Traction
);
callable_signal_builtin!(
    pressure,
    PRESSURE_DESCRIPTOR,
    PRESSURE_SIGNATURES,
    "fea.PressureSignal.fromFunction",
    PRESSURE_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::Pressure
);
callable_signal_builtin!(
    body_force,
    BODY_FORCE_DESCRIPTOR,
    BODY_FORCE_SIGNATURES,
    "fea.BodyForceSignal.fromFunction",
    BODY_FORCE_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::BodyForceDensity
);
callable_signal_builtin!(
    temperature,
    TEMPERATURE_DESCRIPTOR,
    TEMPERATURE_SIGNATURES,
    "fea.TemperatureSignal.fromFunction",
    TEMPERATURE_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::Temperature
);
callable_signal_builtin!(
    temperature_difference,
    TEMPERATURE_DIFFERENCE_DESCRIPTOR,
    TEMPERATURE_DIFFERENCE_SIGNATURES,
    "fea.TemperatureDifferenceSignal.fromFunction",
    TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::TemperatureDifference
);
callable_signal_builtin!(
    heat_flux,
    HEAT_FLUX_DESCRIPTOR,
    HEAT_FLUX_SIGNATURES,
    "fea.HeatFluxSignal.fromFunction",
    HEAT_FLUX_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::HeatFlux
);
callable_signal_builtin!(
    heat_generation,
    HEAT_GENERATION_DESCRIPTOR,
    HEAT_GENERATION_SIGNATURES,
    "fea.HeatGenerationSignal.fromFunction",
    HEAT_GENERATION_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::HeatGenerationDensity
);
callable_signal_builtin!(
    electric_potential,
    ELECTRIC_POTENTIAL_DESCRIPTOR,
    ELECTRIC_POTENTIAL_SIGNATURES,
    "fea.ElectricPotentialSignal.fromFunction",
    ELECTRIC_POTENTIAL_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::ElectricPotential
);
callable_signal_builtin!(
    electric_current,
    ELECTRIC_CURRENT_DESCRIPTOR,
    ELECTRIC_CURRENT_SIGNATURES,
    "fea.ElectricCurrentSignal.fromFunction",
    ELECTRIC_CURRENT_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::ElectricCurrent
);
callable_signal_builtin!(
    electric_charge,
    ELECTRIC_CHARGE_DESCRIPTOR,
    ELECTRIC_CHARGE_SIGNATURES,
    "fea.ElectricChargeSignal.fromFunction",
    ELECTRIC_CHARGE_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::ElectricCharge
);
callable_signal_builtin!(
    current_density,
    CURRENT_DENSITY_DESCRIPTOR,
    CURRENT_DENSITY_SIGNATURES,
    "fea.CurrentDensitySignal.fromFunction",
    CURRENT_DENSITY_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::ElectricCurrentDensity
);
callable_signal_builtin!(
    charge_density,
    CHARGE_DENSITY_DESCRIPTOR,
    CHARGE_DENSITY_SIGNATURES,
    "fea.ChargeDensitySignal.fromFunction",
    CHARGE_DENSITY_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::ElectricChargeDensity
);
callable_signal_builtin!(
    magnetic_vector_potential,
    MAGNETIC_VECTOR_POTENTIAL_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_SIGNATURES,
    "fea.MagneticVectorPotentialSignal.fromFunction",
    MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
    SignalShape::Vector3,
    PhysicalSignal::MagneticVectorPotential
);
callable_signal_builtin!(
    mass_flow,
    MASS_FLOW_DESCRIPTOR,
    MASS_FLOW_SIGNATURES,
    "fea.MassFlowSignal.fromFunction",
    MASS_FLOW_SIGNAL_CLASS,
    SignalShape::Scalar,
    PhysicalSignal::MassFlowRate
);

fn callable_signal<Q>(
    builtin: &'static str,
    class_name: &'static str,
    shape: SignalShape,
    wrap: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 4 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "fromFunction requires id, callable, domain, and sampling",
        ));
    }
    let id = scalar_string(&args[0], builtin, &ERROR_INPUT)?;
    let callable = portable_bound_callable(builtin, &args[1])?;
    let domain = super::super::generator_policies::domain_from_value(builtin, &args[2])?;
    let sampling = super::super::generator_policies::sampling_from_value(builtin, &args[3])?;
    validate_domain_sampling(builtin, &domain, &sampling)?;

    let mut arguments = Vec::new();
    let mut interpolation = SignalInterpolation::Linear;
    let mut extrapolation = SignalExtrapolation::Reject;
    let mut seen = HashSet::new();
    for pair in expect_name_value_tail(builtin, &args[4..])? {
        if !seen.insert(pair.key.clone()) {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("duplicate fromFunction option `{}`", pair.key),
            ));
        }
        match pair.key.as_str() {
            "arguments" => arguments = generator_arguments(builtin, pair.value)?,
            "interpolation" => {
                interpolation =
                    super::super::signal_policies::interpolation_from_value(builtin, pair.value)?
            }
            "extrapolation" => {
                extrapolation =
                    super::super::signal_policies::extrapolation_from_value(builtin, pair.value)?
            }
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported fromFunction option `{other}`"),
                ))
            }
        }
    }

    let frame = matches!(shape, SignalShape::Vector3).then_some(CoordinateFrame::Global);
    let signal = wrap(Signal {
        signal_id: SignalId(id),
        shape,
        frame,
        representation: SignalRepresentation::Generator(SignalGenerator {
            invocation: runmat_analysis_core::GeneratorInvocation {
                callable: GeneratorCallable::BoundFunction {
                    function: bound_function_id(builtin, &callable)?,
                    display_name: callable.qualified_name.clone(),
                },
                arguments,
                effects: GeneratorEffectPolicy::Pure,
            },
            domain,
            sampling,
            interpolation,
            extrapolation,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    });
    callable_signal_object(builtin, class_name, signal, callable)
}

fn bound_function_id(
    builtin: &'static str,
    callable: &CallableValue,
) -> BuiltinResult<runmat_types::ProgramFunctionId> {
    let function = callable
        .owner_identity
        .strip_prefix("bound:")
        .ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "signal callable is not semantically bound",
            )
        })?
        .parse::<u32>()
        .map_err(|_| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "signal callable has an invalid function identity",
            )
        })?;
    Ok(runmat_types::ProgramFunctionId(function))
}

fn portable_bound_callable(builtin: &'static str, value: &Value) -> BuiltinResult<CallableValue> {
    let encoded = crate::execution::value_codec::encode_inline_value(value)
        .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
    let ValuePayload::Inline(encoded) = encoded else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "signal generator callable must be inline and portable",
        ));
    };
    let InlineValue::Callable(callable) = *encoded else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "signal generator must be a function handle or closure",
        ));
    };
    callable
        .validate_identity()
        .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
    if !callable.owner_identity.starts_with("bound:") {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "signal generator must be semantically bound to the executing RunMat program",
        ));
    }
    Ok(callable)
}

fn validate_domain_sampling(
    builtin: &'static str,
    domain: &GeneratorDomain,
    sampling: &SignalSamplingPolicy,
) -> BuiltinResult<()> {
    let first = match sampling {
        SignalSamplingPolicy::Uniform { start, .. }
        | SignalSamplingPolicy::Adaptive { start, .. } => *start,
        SignalSamplingPolicy::CanonicalPoints(points) => points[0],
    };
    if matches!(domain, GeneratorDomain::FrequencyHertz) && first < 0.0 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "frequency-domain sampling cannot include negative frequencies",
        ));
    }
    Ok(())
}

fn generator_arguments(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<GeneratorArgument>> {
    let Value::Cell(cell) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "Arguments must be a cell array so each callable argument has an explicit boundary",
        ));
    };
    cell.data
        .iter()
        .enumerate()
        .map(|(index, value)| generator_argument(builtin, index, value))
        .collect()
}

fn generator_argument(
    builtin: &'static str,
    index: usize,
    value: &Value,
) -> BuiltinResult<GeneratorArgument> {
    let argument = match value {
        Value::Bool(value) => GeneratorArgument::Boolean(*value),
        Value::Int(value) => match value {
            IntValue::I8(value) => GeneratorArgument::SignedInteger(i64::from(*value)),
            IntValue::I16(value) => GeneratorArgument::SignedInteger(i64::from(*value)),
            IntValue::I32(value) => GeneratorArgument::SignedInteger(i64::from(*value)),
            IntValue::I64(value) => GeneratorArgument::SignedInteger(*value),
            IntValue::U8(value) => GeneratorArgument::UnsignedInteger(u64::from(*value)),
            IntValue::U16(value) => GeneratorArgument::UnsignedInteger(u64::from(*value)),
            IntValue::U32(value) => GeneratorArgument::UnsignedInteger(u64::from(*value)),
            IntValue::U64(value) => GeneratorArgument::UnsignedInteger(*value),
        },
        Value::Num(value) if value.is_finite() => GeneratorArgument::Real(*value),
        Value::String(value) => GeneratorArgument::Text(value.clone()),
        Value::CharArray(value) if value.rows == 1 => {
            GeneratorArgument::Text(value.data.iter().collect())
        }
        Value::Tensor(_) => GeneratorArgument::RealVector(finite_numeric_vector(
            builtin,
            value,
            "Arguments entry",
        )?),
        _ => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!(
                    "Arguments entry {} must be a finite real scalar/vector, exact integer, logical scalar, or text scalar",
                    index + 1
                ),
            ))
        }
    };
    Ok(argument)
}

pub(super) fn callable_builtin(class_name: &str) -> &'static str {
    match class_name {
        DISPLACEMENT_SIGNAL_CLASS => "fea.DisplacementSignal.fromFunction",
        ROTATION_SIGNAL_CLASS => "fea.RotationSignal.fromFunction",
        VELOCITY_SIGNAL_CLASS => "fea.VelocitySignal.fromFunction",
        ACCELERATION_SIGNAL_CLASS => "fea.AccelerationSignal.fromFunction",
        ANGULAR_VELOCITY_SIGNAL_CLASS => "fea.AngularVelocitySignal.fromFunction",
        ANGULAR_ACCELERATION_SIGNAL_CLASS => "fea.AngularAccelerationSignal.fromFunction",
        FORCE_SIGNAL_CLASS => "fea.ForceSignal.fromFunction",
        MOMENT_SIGNAL_CLASS => "fea.MomentSignal.fromFunction",
        TRACTION_SIGNAL_CLASS => "fea.TractionSignal.fromFunction",
        PRESSURE_SIGNAL_CLASS => "fea.PressureSignal.fromFunction",
        BODY_FORCE_SIGNAL_CLASS => "fea.BodyForceSignal.fromFunction",
        TEMPERATURE_SIGNAL_CLASS => "fea.TemperatureSignal.fromFunction",
        TEMPERATURE_DIFFERENCE_SIGNAL_CLASS => "fea.TemperatureDifferenceSignal.fromFunction",
        HEAT_FLUX_SIGNAL_CLASS => "fea.HeatFluxSignal.fromFunction",
        HEAT_GENERATION_SIGNAL_CLASS => "fea.HeatGenerationSignal.fromFunction",
        ELECTRIC_POTENTIAL_SIGNAL_CLASS => "fea.ElectricPotentialSignal.fromFunction",
        ELECTRIC_CURRENT_SIGNAL_CLASS => "fea.ElectricCurrentSignal.fromFunction",
        ELECTRIC_CHARGE_SIGNAL_CLASS => "fea.ElectricChargeSignal.fromFunction",
        CURRENT_DENSITY_SIGNAL_CLASS => "fea.CurrentDensitySignal.fromFunction",
        CHARGE_DENSITY_SIGNAL_CLASS => "fea.ChargeDensitySignal.fromFunction",
        MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS => "fea.MagneticVectorPotentialSignal.fromFunction",
        MASS_FLOW_SIGNAL_CLASS => "fea.MassFlowSignal.fromFunction",
        _ => unreachable!("signal schemas contain only known classes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::generator_policies::{time_domain, uniform_sampling};
    use futures::executor::block_on;
    use runmat_value::{CellArray, Closure};

    #[test]
    fn callable_constructor_freezes_captures_and_exact_argument_kinds() {
        let domain = block_on(time_domain()).unwrap();
        let sampling = block_on(uniform_sampling(
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Int(IntValue::U64(3)),
        ))
        .unwrap();
        let callable = Value::Closure(Closure {
            function_name: "anonymous#7".into(),
            bound_function: Some(7),
            captures: vec![Value::Tensor(
                Tensor::new_2d(vec![1.0, 2.0, 3.0], 3, 1).unwrap(),
            )],
        });
        let arguments = Value::Cell(
            CellArray::new(
                vec![
                    Value::Int(IntValue::U64(u64::MAX)),
                    Value::String("pulse".into()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let signal = block_on(acceleration(vec![
            Value::String("base".into()),
            callable,
            domain,
            sampling,
            Value::String("Arguments".into()),
            arguments,
        ]))
        .unwrap();
        let runtime = runtime_signal_from_value("test", &signal).unwrap();
        let generator = runtime.signal.generator().unwrap();
        assert_eq!(
            generator.invocation.callable,
            GeneratorCallable::BoundFunction {
                function: runmat_types::ProgramFunctionId(7),
                display_name: "anonymous#7".into(),
            }
        );
        assert_eq!(
            generator.invocation.arguments,
            vec![
                GeneratorArgument::UnsignedInteger(u64::MAX),
                GeneratorArgument::Text("pulse".into())
            ]
        );
        let binding = runtime.frozen_generator.as_ref().unwrap();
        assert_eq!(binding.callable.captures.len(), 1);
        assert_eq!(binding.callable.owner_identity, "bound:7");

        let encoded = crate::execution::value_codec::encode_inline_value(&signal).unwrap();
        let decoded = crate::execution::value_codec::decode_inline_value(&encoded).unwrap();
        assert_eq!(
            runtime_signal_from_value("test", &decoded).unwrap(),
            runtime
        );
    }

    #[test]
    fn callable_constructor_rejects_workspace_and_nonportable_captures() {
        let domain = block_on(time_domain()).unwrap();
        let sampling = block_on(uniform_sampling(
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Int(IntValue::U64(3)),
        ))
        .unwrap();
        assert!(block_on(acceleration(vec![
            Value::String("base".into()),
            Value::FunctionHandle("loads.pulse".into()),
            domain.clone(),
            sampling.clone(),
        ]))
        .is_err());
        assert!(block_on(acceleration(vec![
            Value::String("base".into()),
            Value::Closure(Closure {
                function_name: "loads.pulse".into(),
                bound_function: Some(8),
                captures: vec![Value::ClassRef("mutable.SessionObject".into())],
            }),
            domain,
            sampling,
        ]))
        .is_err());
    }
}
