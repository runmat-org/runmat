use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::builtins::fea::values::{
    finite_numeric_scalar, finite_numeric_vector, finite_numeric_vector3,
    finite_numeric_vector3_samples,
};
use runmat_analysis_core::{
    Acceleration, AngularAcceleration, AngularVelocity, BodyForceDensity, ElectricCharge,
    ElectricChargeDensity, ElectricCurrentDensity, Force, HeatGenerationDensity, MassFlowRate,
    Moment, Signal, SignalApproximationEvidence, SignalExtrapolation, SignalId,
    SignalInterpolation, SignalKnot, SignalOrdinate, SignalPrecision, SignalProvenance,
    SignalRepresentation, SignalShape, SignalValue, TabulatedSignal, TabulatedSignalDomain,
    Traction,
};
use runmat_execution::value::CallableValue;

pub(super) const DISPLACEMENT_SIGNAL_CLASS: &str = "fea.DisplacementSignal";
pub(super) const ROTATION_SIGNAL_CLASS: &str = "fea.RotationSignal";
pub(super) const VELOCITY_SIGNAL_CLASS: &str = "fea.VelocitySignal";
pub(super) const ACCELERATION_SIGNAL_CLASS: &str = "fea.AccelerationSignal";
pub(super) const ANGULAR_VELOCITY_SIGNAL_CLASS: &str = "fea.AngularVelocitySignal";
pub(super) const ANGULAR_ACCELERATION_SIGNAL_CLASS: &str = "fea.AngularAccelerationSignal";
pub(super) const FORCE_SIGNAL_CLASS: &str = "fea.ForceSignal";
pub(super) const MOMENT_SIGNAL_CLASS: &str = "fea.MomentSignal";
pub(super) const TRACTION_SIGNAL_CLASS: &str = "fea.TractionSignal";
pub(super) const PRESSURE_SIGNAL_CLASS: &str = "fea.PressureSignal";
pub(super) const BODY_FORCE_SIGNAL_CLASS: &str = "fea.BodyForceSignal";
pub(super) const TEMPERATURE_SIGNAL_CLASS: &str = "fea.TemperatureSignal";
pub(super) const TEMPERATURE_DIFFERENCE_SIGNAL_CLASS: &str = "fea.TemperatureDifferenceSignal";
pub(super) const HEAT_FLUX_SIGNAL_CLASS: &str = "fea.HeatFluxSignal";
pub(super) const HEAT_GENERATION_SIGNAL_CLASS: &str = "fea.HeatGenerationSignal";
pub(super) const ELECTRIC_POTENTIAL_SIGNAL_CLASS: &str = "fea.ElectricPotentialSignal";
pub(super) const ELECTRIC_CURRENT_SIGNAL_CLASS: &str = "fea.ElectricCurrentSignal";
pub(super) const ELECTRIC_CHARGE_SIGNAL_CLASS: &str = "fea.ElectricChargeSignal";
pub(super) const CURRENT_DENSITY_SIGNAL_CLASS: &str = "fea.CurrentDensitySignal";
pub(super) const CHARGE_DENSITY_SIGNAL_CLASS: &str = "fea.ChargeDensitySignal";
pub(super) const MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS: &str = "fea.MagneticVectorPotentialSignal";
pub(super) const MASS_FLOW_SIGNAL_CLASS: &str = "fea.MassFlowSignal";

const SIGNAL_FIELDS: &[&str] = &["signal", FROZEN_GENERATOR_PROPERTY];
const SIGNAL_REQUIRED_FIELDS: &[&str] = &["signal"];
pub(super) const FROZEN_GENERATOR_PROPERTY: &str = "frozen_generator";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SignalObject {
    signal: PhysicalSignal,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct FrozenSignalGenerator {
    pub(super) signal_id: SignalId,
    pub(super) callable: CallableValue,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RuntimeSignal {
    pub(super) signal: PhysicalSignal,
    pub(super) frozen_generator: Option<FrozenSignalGenerator>,
}

const SCALAR_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable signal identifier.",
    ),
    required(
        "value",
        BuiltinParamType::NumericScalar,
        "Finite scalar in the canonical SI unit named by the constructor.",
    ),
];
const VECTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable signal identifier.",
    ),
    required(
        "components",
        BuiltinParamType::NumericArray,
        "Finite 1-by-3 or 3-by-1 vector in the canonical SI unit named by the constructor.",
    ),
];
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

fn signal_object(
    builtin: &'static str,
    class_name: &'static str,
    signal: PhysicalSignal,
) -> BuiltinResult<Value> {
    if signal.signal_id().0.trim().is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "signal id must not be empty",
        ));
    }
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        class_name,
        &SignalObject { signal },
    )
}

fn callable_signal_object(
    builtin: &'static str,
    class_name: &'static str,
    signal: PhysicalSignal,
    callable: CallableValue,
) -> BuiltinResult<Value> {
    let signal_id = signal.signal_id().clone();
    let mut value = signal_object(builtin, class_name, signal)?;
    let Value::Object(object) = &mut value else {
        unreachable!("signal_object always returns an object")
    };
    object.properties.insert(
        FROZEN_GENERATOR_PROPERTY.into(),
        callable_to_value(builtin, callable)?,
    );
    runtime_signal_from_value(builtin, &value).and_then(|runtime| {
        if runtime
            .frozen_generator
            .as_ref()
            .is_some_and(|binding| binding.signal_id == signal_id)
        {
            Ok(value)
        } else {
            Err(builtin_error(
                builtin,
                &ERROR_INTERNAL,
                "frozen signal generator was not retained",
            ))
        }
    })
}

fn callable_to_value(builtin: &'static str, callable: CallableValue) -> BuiltinResult<Value> {
    crate::execution::value_codec::decode_inline_value(
        &runmat_execution::value::ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::Callable(callable),
        )),
    )
    .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))
}

fn constant_signal<Q>(
    id: String,
    shape: SignalShape,
    value: SignalValue<Q>,
    wrap: impl FnOnce(Signal<Q>) -> PhysicalSignal,
) -> PhysicalSignal {
    let frame = matches!(shape, SignalShape::Vector3).then_some(CoordinateFrame::Global);
    wrap(Signal {
        signal_id: SignalId(id),
        shape,
        frame,
        representation: SignalRepresentation::Constant(value),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })
}

macro_rules! scalar_signal {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $label:literal, $summary:literal, $class:expr,
        $quantity:ty, $variant:path
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &SCALAR_INPUTS,
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
            summary = $summary,
            keywords = "fea,signal,constant,typed,si",
            descriptor($descriptor),
            integer_capabilities(crate::builtins::fea::values::QUANTITY_INTEGER_CAPABILITIES),
            builtin_path = "crate::builtins::fea::signals"
        )]
        pub async fn $function(id: String, value: Value) -> BuiltinResult<Value> {
            let value = finite_numeric_scalar($builtin, &value, "value")?;
            signal_object(
                $builtin,
                $class,
                constant_signal(
                    id,
                    SignalShape::Scalar,
                    SignalValue::Scalar(<$quantity>::new(value)),
                    $variant,
                ),
            )
        }
    };
}

macro_rules! vector_signal {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $label:literal, $summary:literal, $class:expr,
        $quantity:ty, $variant:path
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &VECTOR_INPUTS,
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
            summary = $summary,
            keywords = "fea,signal,constant,vector,typed,si",
            descriptor($descriptor),
            integer_capabilities(crate::builtins::fea::values::VECTOR_QUANTITY_INTEGER_CAPABILITIES),
            builtin_path = "crate::builtins::fea::signals"
        )]
        pub async fn $function(id: String, components: Value) -> BuiltinResult<Value> {
            let components = finite_numeric_vector3($builtin, &components, "components")?;
            signal_object(
                $builtin,
                $class,
                constant_signal(
                    id,
                    SignalShape::Vector3,
                    SignalValue::Vector3(components.map(|value| <$quantity>::new(value))),
                    $variant,
                ),
            )
        }
    };
}

vector_signal!(
    displacement_constant,
    DISPLACEMENT_DESCRIPTOR,
    DISPLACEMENT_SIGNATURES,
    "fea.DisplacementSignal.constantMeters",
    "signal = fea.DisplacementSignal.constantMeters(id, components)",
    "Create a constant displacement signal in meters.",
    DISPLACEMENT_SIGNAL_CLASS,
    Displacement,
    PhysicalSignal::Displacement
);
vector_signal!(
    rotation_constant,
    ROTATION_DESCRIPTOR,
    ROTATION_SIGNATURES,
    "fea.RotationSignal.constantRadians",
    "signal = fea.RotationSignal.constantRadians(id, components)",
    "Create a constant rotation signal in radians.",
    ROTATION_SIGNAL_CLASS,
    Rotation,
    PhysicalSignal::Rotation
);
vector_signal!(
    velocity_constant,
    VELOCITY_DESCRIPTOR,
    VELOCITY_SIGNATURES,
    "fea.VelocitySignal.constantMetersPerSecond",
    "signal = fea.VelocitySignal.constantMetersPerSecond(id, components)",
    "Create a constant velocity signal in meters per second.",
    VELOCITY_SIGNAL_CLASS,
    Velocity,
    PhysicalSignal::Velocity
);
vector_signal!(
    acceleration_constant,
    ACCELERATION_DESCRIPTOR,
    ACCELERATION_SIGNATURES,
    "fea.AccelerationSignal.constantMetersPerSecondSquared",
    "signal = fea.AccelerationSignal.constantMetersPerSecondSquared(id, components)",
    "Create a constant acceleration signal in meters per second squared.",
    ACCELERATION_SIGNAL_CLASS,
    Acceleration,
    PhysicalSignal::Acceleration
);
vector_signal!(
    angular_velocity_constant,
    ANGULAR_VELOCITY_DESCRIPTOR,
    ANGULAR_VELOCITY_SIGNATURES,
    "fea.AngularVelocitySignal.constantRadiansPerSecond",
    "signal = fea.AngularVelocitySignal.constantRadiansPerSecond(id, components)",
    "Create a constant angular-velocity signal in radians per second.",
    ANGULAR_VELOCITY_SIGNAL_CLASS,
    AngularVelocity,
    PhysicalSignal::AngularVelocity
);
vector_signal!(
    angular_acceleration_constant,
    ANGULAR_ACCELERATION_DESCRIPTOR,
    ANGULAR_ACCELERATION_SIGNATURES,
    "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
    "signal = fea.AngularAccelerationSignal.constantRadiansPerSecondSquared(id, components)",
    "Create a constant angular-acceleration signal in radians per second squared.",
    ANGULAR_ACCELERATION_SIGNAL_CLASS,
    AngularAcceleration,
    PhysicalSignal::AngularAcceleration
);
vector_signal!(
    force_constant,
    FORCE_DESCRIPTOR,
    FORCE_SIGNATURES,
    "fea.ForceSignal.constantNewtons",
    "signal = fea.ForceSignal.constantNewtons(id, components)",
    "Create a constant force signal in newtons.",
    FORCE_SIGNAL_CLASS,
    Force,
    PhysicalSignal::Force
);
vector_signal!(
    moment_constant,
    MOMENT_DESCRIPTOR,
    MOMENT_SIGNATURES,
    "fea.MomentSignal.constantNewtonMeters",
    "signal = fea.MomentSignal.constantNewtonMeters(id, components)",
    "Create a constant moment signal in newton-meters.",
    MOMENT_SIGNAL_CLASS,
    Moment,
    PhysicalSignal::Moment
);
vector_signal!(
    traction_constant,
    TRACTION_DESCRIPTOR,
    TRACTION_SIGNATURES,
    "fea.TractionSignal.constantPascals",
    "signal = fea.TractionSignal.constantPascals(id, components)",
    "Create a constant traction signal in pascals.",
    TRACTION_SIGNAL_CLASS,
    Traction,
    PhysicalSignal::Traction
);
scalar_signal!(
    pressure_constant,
    PRESSURE_DESCRIPTOR,
    PRESSURE_SIGNATURES,
    "fea.PressureSignal.constantPascals",
    "signal = fea.PressureSignal.constantPascals(id, value)",
    "Create a constant pressure signal in pascals.",
    PRESSURE_SIGNAL_CLASS,
    Pressure,
    PhysicalSignal::Pressure
);
vector_signal!(
    body_force_constant,
    BODY_FORCE_DESCRIPTOR,
    BODY_FORCE_SIGNATURES,
    "fea.BodyForceSignal.constantNewtonsPerCubicMeter",
    "signal = fea.BodyForceSignal.constantNewtonsPerCubicMeter(id, components)",
    "Create a constant body-force-density signal in newtons per cubic meter.",
    BODY_FORCE_SIGNAL_CLASS,
    BodyForceDensity,
    PhysicalSignal::BodyForceDensity
);
scalar_signal!(
    temperature_constant,
    TEMPERATURE_DESCRIPTOR,
    TEMPERATURE_SIGNATURES,
    "fea.TemperatureSignal.constantKelvin",
    "signal = fea.TemperatureSignal.constantKelvin(id, value)",
    "Create a constant absolute-temperature signal in kelvin.",
    TEMPERATURE_SIGNAL_CLASS,
    Temperature,
    PhysicalSignal::Temperature
);
scalar_signal!(
    temperature_difference_constant,
    TEMPERATURE_DIFFERENCE_DESCRIPTOR,
    TEMPERATURE_DIFFERENCE_SIGNATURES,
    "fea.TemperatureDifferenceSignal.constantKelvin",
    "signal = fea.TemperatureDifferenceSignal.constantKelvin(id, value)",
    "Create a constant temperature-difference signal in kelvin.",
    TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
    TemperatureDifference,
    PhysicalSignal::TemperatureDifference
);
scalar_signal!(
    heat_flux_constant,
    HEAT_FLUX_DESCRIPTOR,
    HEAT_FLUX_SIGNATURES,
    "fea.HeatFluxSignal.constantWattsPerSquareMeter",
    "signal = fea.HeatFluxSignal.constantWattsPerSquareMeter(id, value)",
    "Create a constant heat-flux signal in watts per square meter.",
    HEAT_FLUX_SIGNAL_CLASS,
    HeatFlux,
    PhysicalSignal::HeatFlux
);
scalar_signal!(
    heat_generation_constant,
    HEAT_GENERATION_DESCRIPTOR,
    HEAT_GENERATION_SIGNATURES,
    "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
    "signal = fea.HeatGenerationSignal.constantWattsPerCubicMeter(id, value)",
    "Create a constant volumetric heat-generation signal in watts per cubic meter.",
    HEAT_GENERATION_SIGNAL_CLASS,
    HeatGenerationDensity,
    PhysicalSignal::HeatGenerationDensity
);
scalar_signal!(
    electric_potential_constant,
    ELECTRIC_POTENTIAL_DESCRIPTOR,
    ELECTRIC_POTENTIAL_SIGNATURES,
    "fea.ElectricPotentialSignal.constantVolts",
    "signal = fea.ElectricPotentialSignal.constantVolts(id, value)",
    "Create a constant electric-potential signal in volts.",
    ELECTRIC_POTENTIAL_SIGNAL_CLASS,
    ElectricPotential,
    PhysicalSignal::ElectricPotential
);
scalar_signal!(
    electric_current_constant,
    ELECTRIC_CURRENT_DESCRIPTOR,
    ELECTRIC_CURRENT_SIGNATURES,
    "fea.ElectricCurrentSignal.constantAmperes",
    "signal = fea.ElectricCurrentSignal.constantAmperes(id, value)",
    "Create a constant electric-current signal in amperes.",
    ELECTRIC_CURRENT_SIGNAL_CLASS,
    ElectricCurrent,
    PhysicalSignal::ElectricCurrent
);
scalar_signal!(
    electric_charge_constant,
    ELECTRIC_CHARGE_DESCRIPTOR,
    ELECTRIC_CHARGE_SIGNATURES,
    "fea.ElectricChargeSignal.constantCoulombs",
    "signal = fea.ElectricChargeSignal.constantCoulombs(id, value)",
    "Create a constant electric-charge signal in coulombs.",
    ELECTRIC_CHARGE_SIGNAL_CLASS,
    ElectricCharge,
    PhysicalSignal::ElectricCharge
);
vector_signal!(
    current_density_constant,
    CURRENT_DENSITY_DESCRIPTOR,
    CURRENT_DENSITY_SIGNATURES,
    "fea.CurrentDensitySignal.constantAmperesPerSquareMeter",
    "signal = fea.CurrentDensitySignal.constantAmperesPerSquareMeter(id, components)",
    "Create a constant current-density signal in amperes per square meter.",
    CURRENT_DENSITY_SIGNAL_CLASS,
    ElectricCurrentDensity,
    PhysicalSignal::ElectricCurrentDensity
);
scalar_signal!(
    charge_density_constant,
    CHARGE_DENSITY_DESCRIPTOR,
    CHARGE_DENSITY_SIGNATURES,
    "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
    "signal = fea.ChargeDensitySignal.constantCoulombsPerCubicMeter(id, value)",
    "Create a constant charge-density signal in coulombs per cubic meter.",
    CHARGE_DENSITY_SIGNAL_CLASS,
    ElectricChargeDensity,
    PhysicalSignal::ElectricChargeDensity
);
vector_signal!(
    magnetic_vector_potential_constant,
    MAGNETIC_VECTOR_POTENTIAL_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_SIGNATURES,
    "fea.MagneticVectorPotentialSignal.constantWebersPerMeter",
    "signal = fea.MagneticVectorPotentialSignal.constantWebersPerMeter(id, components)",
    "Create a constant magnetic-vector-potential signal in webers per meter.",
    MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
    MagneticVectorPotential,
    PhysicalSignal::MagneticVectorPotential
);
scalar_signal!(
    mass_flow_constant,
    MASS_FLOW_DESCRIPTOR,
    MASS_FLOW_SIGNATURES,
    "fea.MassFlowSignal.constantKilogramsPerSecond",
    "signal = fea.MassFlowSignal.constantKilogramsPerSecond(id, value)",
    "Create a constant mass-flow signal in kilograms per second.",
    MASS_FLOW_SIGNAL_CLASS,
    MassFlowRate,
    PhysicalSignal::MassFlowRate
);

pub(super) mod bindings;
pub(crate) mod callable;
pub(crate) mod knots;
pub(super) mod materialize;
pub(crate) mod time_samples;

pub(super) fn physical_signal_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<PhysicalSignal> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "expected a typed FEA signal object",
        ));
    };
    let expected_quantity = signal_class_quantity(&object.class_name).ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("unsupported FEA signal class `{}`", object.class_name),
        )
    })?;
    let mut public_object = object.clone();
    public_object.properties.remove(FROZEN_GENERATOR_PROPERTY);
    let payload: SignalObject = object_payload(
        builtin,
        &Value::Object(public_object),
        signal_class_name(expected_quantity),
    )?;
    if payload.signal.quantity() != expected_quantity {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{} contains the wrong physical quantity", object.class_name),
        ));
    }
    Ok(payload.signal)
}

pub(super) fn runtime_signal_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<RuntimeSignal> {
    let signal = physical_signal_from_value(builtin, value)?;
    let Value::Object(object) = value else {
        unreachable!("physical_signal_from_value accepted only an object")
    };
    let frozen_generator = object
        .properties
        .get(FROZEN_GENERATOR_PROPERTY)
        .map(|callable| frozen_generator_from_value(builtin, &signal, callable))
        .transpose()?;
    if signal.generator().is_some() != frozen_generator.is_some() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "a runtime callable signal must retain exactly one frozen generator binding",
        ));
    }
    Ok(RuntimeSignal {
        signal,
        frozen_generator,
    })
}

fn frozen_generator_from_value(
    builtin: &'static str,
    signal: &PhysicalSignal,
    value: &Value,
) -> BuiltinResult<FrozenSignalGenerator> {
    let payload = crate::execution::value_codec::encode_inline_value(value)
        .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
    let runmat_execution::value::ValuePayload::Inline(value) = payload else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "frozen signal callable must be an inline portable value",
        ));
    };
    let runmat_execution::value::InlineValue::Callable(callable) = *value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "frozen signal generator must be a function handle or closure",
        ));
    };
    callable
        .validate_identity()
        .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
    if !callable.owner_identity.starts_with("bound:") {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "signal generators must be semantically bound functions or closures",
        ));
    }
    let generator = signal.generator().ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            "only callable-backed signals may retain a frozen generator",
        )
    })?;
    if generator.invocation.callable.display_name() != callable.qualified_name {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "frozen callable identity does not match the signal generator",
        ));
    }
    Ok(FrozenSignalGenerator {
        signal_id: signal.signal_id().clone(),
        callable,
    })
}

fn signal_class_name(quantity: runmat_analysis_core::PhysicalQuantityKind) -> &'static str {
    use runmat_analysis_core::PhysicalQuantityKind::*;
    match quantity {
        Displacement => DISPLACEMENT_SIGNAL_CLASS,
        Rotation => ROTATION_SIGNAL_CLASS,
        Velocity => VELOCITY_SIGNAL_CLASS,
        Acceleration => ACCELERATION_SIGNAL_CLASS,
        AngularVelocity => ANGULAR_VELOCITY_SIGNAL_CLASS,
        AngularAcceleration => ANGULAR_ACCELERATION_SIGNAL_CLASS,
        Force => FORCE_SIGNAL_CLASS,
        Moment => MOMENT_SIGNAL_CLASS,
        Traction => TRACTION_SIGNAL_CLASS,
        Pressure => PRESSURE_SIGNAL_CLASS,
        BodyForceDensity => BODY_FORCE_SIGNAL_CLASS,
        Temperature => TEMPERATURE_SIGNAL_CLASS,
        TemperatureDifference => TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
        HeatFlux => HEAT_FLUX_SIGNAL_CLASS,
        HeatGenerationDensity => HEAT_GENERATION_SIGNAL_CLASS,
        ElectricPotential => ELECTRIC_POTENTIAL_SIGNAL_CLASS,
        ElectricCurrent => ELECTRIC_CURRENT_SIGNAL_CLASS,
        ElectricCharge => ELECTRIC_CHARGE_SIGNAL_CLASS,
        ElectricCurrentDensity => CURRENT_DENSITY_SIGNAL_CLASS,
        ElectricChargeDensity => CHARGE_DENSITY_SIGNAL_CLASS,
        MagneticVectorPotential => MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
        MassFlowRate => MASS_FLOW_SIGNAL_CLASS,
    }
}

fn signal_class_quantity(class_name: &str) -> Option<runmat_analysis_core::PhysicalQuantityKind> {
    use runmat_analysis_core::PhysicalQuantityKind::*;
    Some(match class_name {
        DISPLACEMENT_SIGNAL_CLASS => Displacement,
        ROTATION_SIGNAL_CLASS => Rotation,
        VELOCITY_SIGNAL_CLASS => Velocity,
        ACCELERATION_SIGNAL_CLASS => Acceleration,
        ANGULAR_VELOCITY_SIGNAL_CLASS => AngularVelocity,
        ANGULAR_ACCELERATION_SIGNAL_CLASS => AngularAcceleration,
        FORCE_SIGNAL_CLASS => Force,
        MOMENT_SIGNAL_CLASS => Moment,
        TRACTION_SIGNAL_CLASS => Traction,
        PRESSURE_SIGNAL_CLASS => Pressure,
        BODY_FORCE_SIGNAL_CLASS => BodyForceDensity,
        TEMPERATURE_SIGNAL_CLASS => Temperature,
        TEMPERATURE_DIFFERENCE_SIGNAL_CLASS => TemperatureDifference,
        HEAT_FLUX_SIGNAL_CLASS => HeatFlux,
        HEAT_GENERATION_SIGNAL_CLASS => HeatGenerationDensity,
        ELECTRIC_POTENTIAL_SIGNAL_CLASS => ElectricPotential,
        ELECTRIC_CURRENT_SIGNAL_CLASS => ElectricCurrent,
        ELECTRIC_CHARGE_SIGNAL_CLASS => ElectricCharge,
        CURRENT_DENSITY_SIGNAL_CLASS => ElectricCurrentDensity,
        CHARGE_DENSITY_SIGNAL_CLASS => ElectricChargeDensity,
        MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS => MagneticVectorPotential,
        MASS_FLOW_SIGNAL_CLASS => MassFlowRate,
        _ => return None,
    })
}

pub(super) fn ensure_signal_classes_registered() {
    knots::ensure_signal_knot_class_registered();
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(DISPLACEMENT_SIGNAL_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
        for (class_name, type_identity, _, _) in signal_schemas() {
            crate::execution::value_codec::register_immutable_value_class(
                crate::execution::value_codec::ImmutableValueClassSchema {
                    class_name,
                    type_identity,
                    schema_version: 1,
                    allowed_fields: SIGNAL_FIELDS,
                    required_fields: SIGNAL_REQUIRED_FIELDS,
                    validate: validate_signal_object,
                },
            )
            .unwrap_or_else(|error| {
                panic!("failed to register immutable FEA signal class {class_name}: {error}")
            });
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        for (class_name, _, method_name, builtin_name) in signal_schemas() {
            let mut methods =
                HashMap::from([(method_name.into(), static_method(method_name, builtin_name))]);
            methods.insert(
                "fromTimeSamples".into(),
                static_method("fromTimeSamples", time_sample_builtin(class_name)),
            );
            methods.insert(
                "fromTimeKnots".into(),
                static_method("fromTimeKnots", time_knot_builtin(class_name)),
            );
            methods.insert(
                "fromFunction".into(),
                static_method("fromFunction", callable::callable_builtin(class_name)),
            );
            crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                name: class_name.into(),
                parent: None,
                properties: signal_properties(),
                methods,
            });
        }
    });
}

fn time_knot_builtin(class_name: &str) -> &'static str {
    match class_name {
        DISPLACEMENT_SIGNAL_CLASS => "fea.DisplacementSignal.fromTimeKnots",
        ROTATION_SIGNAL_CLASS => "fea.RotationSignal.fromTimeKnots",
        VELOCITY_SIGNAL_CLASS => "fea.VelocitySignal.fromTimeKnots",
        ACCELERATION_SIGNAL_CLASS => "fea.AccelerationSignal.fromTimeKnots",
        ANGULAR_VELOCITY_SIGNAL_CLASS => "fea.AngularVelocitySignal.fromTimeKnots",
        ANGULAR_ACCELERATION_SIGNAL_CLASS => "fea.AngularAccelerationSignal.fromTimeKnots",
        FORCE_SIGNAL_CLASS => "fea.ForceSignal.fromTimeKnots",
        MOMENT_SIGNAL_CLASS => "fea.MomentSignal.fromTimeKnots",
        TRACTION_SIGNAL_CLASS => "fea.TractionSignal.fromTimeKnots",
        PRESSURE_SIGNAL_CLASS => "fea.PressureSignal.fromTimeKnots",
        BODY_FORCE_SIGNAL_CLASS => "fea.BodyForceSignal.fromTimeKnots",
        TEMPERATURE_SIGNAL_CLASS => "fea.TemperatureSignal.fromTimeKnots",
        TEMPERATURE_DIFFERENCE_SIGNAL_CLASS => "fea.TemperatureDifferenceSignal.fromTimeKnots",
        HEAT_FLUX_SIGNAL_CLASS => "fea.HeatFluxSignal.fromTimeKnots",
        HEAT_GENERATION_SIGNAL_CLASS => "fea.HeatGenerationSignal.fromTimeKnots",
        ELECTRIC_POTENTIAL_SIGNAL_CLASS => "fea.ElectricPotentialSignal.fromTimeKnots",
        ELECTRIC_CURRENT_SIGNAL_CLASS => "fea.ElectricCurrentSignal.fromTimeKnots",
        ELECTRIC_CHARGE_SIGNAL_CLASS => "fea.ElectricChargeSignal.fromTimeKnots",
        CURRENT_DENSITY_SIGNAL_CLASS => "fea.CurrentDensitySignal.fromTimeKnots",
        CHARGE_DENSITY_SIGNAL_CLASS => "fea.ChargeDensitySignal.fromTimeKnots",
        MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS => "fea.MagneticVectorPotentialSignal.fromTimeKnots",
        MASS_FLOW_SIGNAL_CLASS => "fea.MassFlowSignal.fromTimeKnots",
        _ => unreachable!("signal schemas contain only known classes"),
    }
}

fn time_sample_builtin(class_name: &str) -> &'static str {
    match class_name {
        DISPLACEMENT_SIGNAL_CLASS => "fea.DisplacementSignal.fromTimeSamples",
        ROTATION_SIGNAL_CLASS => "fea.RotationSignal.fromTimeSamples",
        VELOCITY_SIGNAL_CLASS => "fea.VelocitySignal.fromTimeSamples",
        ACCELERATION_SIGNAL_CLASS => "fea.AccelerationSignal.fromTimeSamples",
        ANGULAR_VELOCITY_SIGNAL_CLASS => "fea.AngularVelocitySignal.fromTimeSamples",
        ANGULAR_ACCELERATION_SIGNAL_CLASS => "fea.AngularAccelerationSignal.fromTimeSamples",
        FORCE_SIGNAL_CLASS => "fea.ForceSignal.fromTimeSamples",
        MOMENT_SIGNAL_CLASS => "fea.MomentSignal.fromTimeSamples",
        TRACTION_SIGNAL_CLASS => "fea.TractionSignal.fromTimeSamples",
        PRESSURE_SIGNAL_CLASS => "fea.PressureSignal.fromTimeSamples",
        BODY_FORCE_SIGNAL_CLASS => "fea.BodyForceSignal.fromTimeSamples",
        TEMPERATURE_SIGNAL_CLASS => "fea.TemperatureSignal.fromTimeSamples",
        TEMPERATURE_DIFFERENCE_SIGNAL_CLASS => "fea.TemperatureDifferenceSignal.fromTimeSamples",
        HEAT_FLUX_SIGNAL_CLASS => "fea.HeatFluxSignal.fromTimeSamples",
        HEAT_GENERATION_SIGNAL_CLASS => "fea.HeatGenerationSignal.fromTimeSamples",
        ELECTRIC_POTENTIAL_SIGNAL_CLASS => "fea.ElectricPotentialSignal.fromTimeSamples",
        ELECTRIC_CURRENT_SIGNAL_CLASS => "fea.ElectricCurrentSignal.fromTimeSamples",
        ELECTRIC_CHARGE_SIGNAL_CLASS => "fea.ElectricChargeSignal.fromTimeSamples",
        CURRENT_DENSITY_SIGNAL_CLASS => "fea.CurrentDensitySignal.fromTimeSamples",
        CHARGE_DENSITY_SIGNAL_CLASS => "fea.ChargeDensitySignal.fromTimeSamples",
        MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS => {
            "fea.MagneticVectorPotentialSignal.fromTimeSamples"
        }
        MASS_FLOW_SIGNAL_CLASS => "fea.MassFlowSignal.fromTimeSamples",
        _ => unreachable!("signal schemas contain only known classes"),
    }
}

fn validate_signal_object(object: &ObjectInstance) -> Result<(), String> {
    if !matches!(object.properties.get("signal"), Some(Value::Struct(_))) {
        return Err(format!(
            "{}.signal must be a typed structure",
            object.class_name
        ));
    }
    if let Some(callable) = object.properties.get(FROZEN_GENERATOR_PROPERTY) {
        let payload = crate::execution::value_codec::encode_inline_value(callable)
            .map_err(|error| error.to_string())?;
        if !matches!(
            payload,
            runmat_execution::value::ValuePayload::Inline(value)
                if matches!(*value, runmat_execution::value::InlineValue::Callable(_))
        ) {
            return Err(format!(
                "{}.{} must be a portable callable",
                object.class_name, FROZEN_GENERATOR_PROPERTY
            ));
        }
    }
    Ok(())
}

fn signal_properties() -> HashMap<String, crate::class_registry::RuntimeProperty> {
    let mut properties = immutable_properties(SIGNAL_FIELDS);
    properties
        .get_mut(FROZEN_GENERATOR_PROPERTY)
        .expect("frozen generator property is registered")
        .get_access = MemberAccess::Private;
    properties
}

fn immutable_properties(
    fields: &[&str],
) -> HashMap<String, crate::class_registry::RuntimeProperty> {
    fields
        .iter()
        .map(|field| {
            (
                (*field).into(),
                crate::class_registry::RuntimeProperty {
                    name: (*field).into(),
                    is_static: false,
                    is_constant: true,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    default_value: None,
                },
            )
        })
        .collect()
}

fn static_method(method: &str, builtin: &str) -> crate::class_registry::RuntimeMethod {
    crate::class_registry::RuntimeMethod {
        name: method.into(),
        is_static: true,
        is_abstract: false,
        is_sealed: false,
        access: MemberAccess::Public,
        function_name: builtin.into(),
        implicit_class_argument: None,
    }
}

type SignalSchema = (&'static str, &'static str, &'static str, &'static str);

fn signal_schemas() -> [SignalSchema; 22] {
    [
        (
            DISPLACEMENT_SIGNAL_CLASS,
            "runmat.fea.signal.displacement",
            "constantMeters",
            "fea.DisplacementSignal.constantMeters",
        ),
        (
            ROTATION_SIGNAL_CLASS,
            "runmat.fea.signal.rotation",
            "constantRadians",
            "fea.RotationSignal.constantRadians",
        ),
        (
            VELOCITY_SIGNAL_CLASS,
            "runmat.fea.signal.velocity",
            "constantMetersPerSecond",
            "fea.VelocitySignal.constantMetersPerSecond",
        ),
        (
            ACCELERATION_SIGNAL_CLASS,
            "runmat.fea.signal.acceleration",
            "constantMetersPerSecondSquared",
            "fea.AccelerationSignal.constantMetersPerSecondSquared",
        ),
        (
            ANGULAR_VELOCITY_SIGNAL_CLASS,
            "runmat.fea.signal.angular-velocity",
            "constantRadiansPerSecond",
            "fea.AngularVelocitySignal.constantRadiansPerSecond",
        ),
        (
            ANGULAR_ACCELERATION_SIGNAL_CLASS,
            "runmat.fea.signal.angular-acceleration",
            "constantRadiansPerSecondSquared",
            "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
        ),
        (
            FORCE_SIGNAL_CLASS,
            "runmat.fea.signal.force",
            "constantNewtons",
            "fea.ForceSignal.constantNewtons",
        ),
        (
            MOMENT_SIGNAL_CLASS,
            "runmat.fea.signal.moment",
            "constantNewtonMeters",
            "fea.MomentSignal.constantNewtonMeters",
        ),
        (
            TRACTION_SIGNAL_CLASS,
            "runmat.fea.signal.traction",
            "constantPascals",
            "fea.TractionSignal.constantPascals",
        ),
        (
            PRESSURE_SIGNAL_CLASS,
            "runmat.fea.signal.pressure",
            "constantPascals",
            "fea.PressureSignal.constantPascals",
        ),
        (
            BODY_FORCE_SIGNAL_CLASS,
            "runmat.fea.signal.body-force",
            "constantNewtonsPerCubicMeter",
            "fea.BodyForceSignal.constantNewtonsPerCubicMeter",
        ),
        (
            TEMPERATURE_SIGNAL_CLASS,
            "runmat.fea.signal.temperature",
            "constantKelvin",
            "fea.TemperatureSignal.constantKelvin",
        ),
        (
            TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
            "runmat.fea.signal.temperature-difference",
            "constantKelvin",
            "fea.TemperatureDifferenceSignal.constantKelvin",
        ),
        (
            HEAT_FLUX_SIGNAL_CLASS,
            "runmat.fea.signal.heat-flux",
            "constantWattsPerSquareMeter",
            "fea.HeatFluxSignal.constantWattsPerSquareMeter",
        ),
        (
            HEAT_GENERATION_SIGNAL_CLASS,
            "runmat.fea.signal.heat-generation",
            "constantWattsPerCubicMeter",
            "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
        ),
        (
            ELECTRIC_POTENTIAL_SIGNAL_CLASS,
            "runmat.fea.signal.electric-potential",
            "constantVolts",
            "fea.ElectricPotentialSignal.constantVolts",
        ),
        (
            ELECTRIC_CURRENT_SIGNAL_CLASS,
            "runmat.fea.signal.electric-current",
            "constantAmperes",
            "fea.ElectricCurrentSignal.constantAmperes",
        ),
        (
            ELECTRIC_CHARGE_SIGNAL_CLASS,
            "runmat.fea.signal.electric-charge",
            "constantCoulombs",
            "fea.ElectricChargeSignal.constantCoulombs",
        ),
        (
            CURRENT_DENSITY_SIGNAL_CLASS,
            "runmat.fea.signal.current-density",
            "constantAmperesPerSquareMeter",
            "fea.CurrentDensitySignal.constantAmperesPerSquareMeter",
        ),
        (
            CHARGE_DENSITY_SIGNAL_CLASS,
            "runmat.fea.signal.charge-density",
            "constantCoulombsPerCubicMeter",
            "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
        ),
        (
            MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
            "runmat.fea.signal.magnetic-vector-potential",
            "constantWebersPerMeter",
            "fea.MagneticVectorPotentialSignal.constantWebersPerMeter",
        ),
        (
            MASS_FLOW_SIGNAL_CLASS,
            "runmat.fea.signal.mass-flow",
            "constantKilogramsPerSecond",
            "fea.MassFlowSignal.constantKilogramsPerSecond",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::signal_policies::{
        extrapolation_reject, interpolation_linear, point_continuity_right,
    };
    use futures::executor::block_on;
    use runmat_value::CellArray;

    #[test]
    fn force_signal_is_typed_portable_and_rejects_bad_shape() {
        let components = Value::Tensor(Tensor::new_2d(vec![0.0, -1250.0, 0.0], 1, 3).unwrap());
        let value = block_on(force_constant("tip.force".into(), components)).unwrap();
        let signal = physical_signal_from_value("test", &value).unwrap();
        assert_eq!(
            signal.quantity(),
            runmat_analysis_core::PhysicalQuantityKind::Force
        );
        assert!(matches!(signal.shape(), SignalShape::Vector3));
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(
                &crate::execution::value_codec::encode_inline_value(&value).unwrap()
            )
            .unwrap(),
            value
        );
        assert!(block_on(force_constant(
            "bad".into(),
            Value::Tensor(Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap())
        ))
        .is_err());
    }

    #[test]
    fn signal_class_cannot_lie_about_its_quantity() {
        let value = block_on(pressure_constant("pressure".into(), Value::Num(3.0))).unwrap();
        let Value::Object(mut object) = value else {
            unreachable!()
        };
        object.class_name = FORCE_SIGNAL_CLASS.into();
        assert!(physical_signal_from_value("test", &Value::Object(object)).is_err());
    }

    #[test]
    fn time_samples_preserve_vector_samples_and_closed_policies() {
        let times = Value::Tensor(Tensor::new_2d(vec![0.0, 0.25], 2, 1).unwrap());
        let values =
            Value::Tensor(Tensor::new_2d(vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0], 2, 3).unwrap());
        let signal = block_on(time_samples::force_time_samples(
            "pulse".into(),
            times,
            values,
            block_on(interpolation_linear()).unwrap(),
            block_on(extrapolation_reject()).unwrap(),
        ))
        .unwrap();
        let PhysicalSignal::Force(signal) = physical_signal_from_value("test", &signal).unwrap()
        else {
            panic!("typed class must contain a force signal")
        };
        assert_eq!(signal.evidence.sample_count, 2);
        let SignalRepresentation::Tabulated(table) = signal.representation else {
            panic!("time samples must create a tabulated signal")
        };
        assert_eq!(table.interpolation, SignalInterpolation::Linear);
        assert_eq!(table.extrapolation, SignalExtrapolation::Reject);
        assert_eq!(
            table
                .knots
                .into_iter()
                .map(|knot| knot.value)
                .collect::<Vec<_>>(),
            vec![
                runmat_analysis_core::SignalKnotValue::Continuous(SignalOrdinate::Real(
                    SignalValue::Vector3([Force::new(0.0), Force::new(1.0), Force::new(2.0),])
                )),
                runmat_analysis_core::SignalKnotValue::Continuous(SignalOrdinate::Real(
                    SignalValue::Vector3([Force::new(10.0), Force::new(11.0), Force::new(12.0),])
                )),
            ]
        );
    }

    #[test]
    fn time_samples_reject_unsorted_axes_and_mismatched_values() {
        let interpolation = block_on(interpolation_linear()).unwrap();
        let extrapolation = block_on(extrapolation_reject()).unwrap();
        let unsorted = Value::Tensor(Tensor::new_2d(vec![0.0, 0.5, 0.25], 3, 1).unwrap());
        let values = Value::Tensor(Tensor::new_2d(vec![1.0, 2.0, 3.0], 3, 1).unwrap());
        assert!(block_on(time_samples::pressure_time_samples(
            "bad-axis".into(),
            unsorted,
            values,
            interpolation.clone(),
            extrapolation.clone(),
        ))
        .is_err());

        let times = Value::Tensor(Tensor::new_2d(vec![0.0, 0.5], 2, 1).unwrap());
        let values = Value::Tensor(Tensor::new_2d(vec![1.0], 1, 1).unwrap());
        assert!(block_on(time_samples::pressure_time_samples(
            "bad-values".into(),
            times,
            values,
            interpolation,
            extrapolation,
        ))
        .is_err());
    }

    #[test]
    fn time_knots_author_a_typed_jump_without_losing_its_sides() {
        let scalar = |value| Value::Num(value);
        let first = block_on(knots::continuous(scalar(0.0), scalar(273.0))).unwrap();
        let jump = block_on(knots::jump(
            scalar(0.5),
            scalar(273.0),
            scalar(373.0),
            block_on(point_continuity_right()).unwrap(),
        ))
        .unwrap();
        let last = block_on(knots::continuous(scalar(1.0), scalar(373.0))).unwrap();
        let knots = Value::Cell(CellArray::new(vec![first, jump, last], 3, 1).unwrap());
        let value = block_on(knots::temperature(
            "thermal.step".into(),
            knots,
            block_on(interpolation_linear()).unwrap(),
            block_on(extrapolation_reject()).unwrap(),
        ))
        .unwrap();
        let PhysicalSignal::Temperature(signal) =
            physical_signal_from_value("test", &value).unwrap()
        else {
            panic!("typed class must contain a temperature signal")
        };
        let SignalRepresentation::Tabulated(table) = signal.representation else {
            panic!("time knots must create a tabulated signal")
        };
        assert!(matches!(
            &table.knots[1].value,
            runmat_analysis_core::SignalKnotValue::Jump {
                left_limit: SignalOrdinate::Real(SignalValue::Scalar(left)),
                right_limit: SignalOrdinate::Real(SignalValue::Scalar(right)),
                continuity: runmat_analysis_core::SignalPointContinuity::Right,
            } if left.kelvin == 273.0 && right.kelvin == 373.0
        ));
    }

    #[test]
    fn time_knots_reject_endpoint_jumps_and_mixed_shapes() {
        let scalar = |value| Value::Num(value);
        let jump = block_on(knots::jump(
            scalar(0.0),
            scalar(0.0),
            scalar(1.0),
            block_on(point_continuity_right()).unwrap(),
        ))
        .unwrap();
        let last = block_on(knots::continuous(scalar(1.0), scalar(1.0))).unwrap();
        let endpoint_jump = Value::Cell(CellArray::new(vec![jump, last], 1, 2).unwrap());
        assert!(block_on(knots::temperature(
            "bad.endpoint".into(),
            endpoint_jump,
            block_on(interpolation_linear()).unwrap(),
            block_on(extrapolation_reject()).unwrap(),
        ))
        .is_err());

        let first = block_on(knots::continuous(scalar(0.0), scalar(0.0))).unwrap();
        let vector = block_on(knots::continuous(
            scalar(1.0),
            Value::Tensor(Tensor::new_2d(vec![1.0, 2.0, 3.0], 1, 3).unwrap()),
        ))
        .unwrap();
        let mixed = Value::Cell(CellArray::new(vec![first, vector], 1, 2).unwrap());
        assert!(block_on(knots::temperature(
            "bad.shape".into(),
            mixed,
            block_on(interpolation_linear()).unwrap(),
            block_on(extrapolation_reject()).unwrap(),
        ))
        .is_err());
    }
}
