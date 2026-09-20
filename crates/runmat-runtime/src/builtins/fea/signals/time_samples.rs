use super::*;

const TIME_SAMPLE_INPUTS: [BuiltinParamDescriptor; 5] = [
    required("id", BuiltinParamType::StringScalar, "Stable signal identifier."),
    required(
        "times",
        BuiltinParamType::NumericArray,
        "Finite time samples in seconds as a nonempty row or column vector.",
    ),
    required(
        "values",
        BuiltinParamType::NumericArray,
        "Finite scalar or three-component samples in the canonical SI unit named by the signal class.",
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
const TIME_SAMPLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "times",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Time samples are validated as an ordered host vector before conversion to canonical binary64 seconds.",
    },
    BuiltinIntegerInputCapability {
        name: "values",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Physical samples are shape-checked on the host before one conversion to canonical binary64 SI storage.",
    },
];
const TIME_SAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed time-sampled physical signal",
        inputs: &TIME_SAMPLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Time and physical ordinate arrays are portable host inputs; resident arrays are rejected without provider execution.",
    }];

fn time_signal<Q>(
    id: String,
    shape: SignalShape,
    times: Vec<f64>,
    values: Vec<SignalOrdinate<Q>>,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
    wrap: impl FnOnce(Signal<Q>) -> PhysicalSignal,
) -> BuiltinResult<PhysicalSignal> {
    if times.len() < 2 || times.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(builtin_error(
            "FEA time signal",
            &ERROR_INPUT,
            "time samples must contain at least two strictly increasing values",
        ));
    }
    let sample_count = values.len();
    let frame = matches!(shape, SignalShape::Vector3).then_some(CoordinateFrame::Global);
    Ok(wrap(Signal {
        signal_id: SignalId(id),
        shape,
        frame,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: times
                .into_iter()
                .zip(values)
                .map(|(abscissa, value)| SignalKnot::continuous(abscissa, value))
                .collect(),
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
    }))
}

pub(super) fn policies(
    builtin: &'static str,
    interpolation: &Value,
    extrapolation: &Value,
) -> BuiltinResult<(SignalInterpolation, SignalExtrapolation)> {
    Ok((
        super::signal_policies::interpolation_from_value(builtin, interpolation)?,
        super::signal_policies::extrapolation_from_value(builtin, extrapolation)?,
    ))
}

macro_rules! scalar_time_signal {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $label:literal, $summary:literal, $class:expr,
        $quantity:ty, $variant:path
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &TIME_SAMPLE_INPUTS,
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
                                            keywords = "fea,signal,time,history,samples,typed,si",
                                            descriptor($descriptor),
                                            integer_capabilities(TIME_SAMPLE_INTEGER_CAPABILITIES),
                                            builtin_path = "crate::builtins::fea::signals::time_samples"
                                        )]
        pub async fn $function(
            id: String,
            times: Value,
            values: Value,
            interpolation: Value,
            extrapolation: Value,
        ) -> BuiltinResult<Value> {
            let times = finite_numeric_vector($builtin, &times, "times")?;
            let values = finite_numeric_vector($builtin, &values, "values")?;
            if values.len() != times.len() {
                return Err(builtin_error(
                    $builtin,
                    &ERROR_INPUT,
                    "values must contain one scalar sample for each time",
                ));
            }
            let (interpolation, extrapolation) =
                policies($builtin, &interpolation, &extrapolation)?;
            let signal = time_signal(
                id,
                SignalShape::Scalar,
                times,
                values
                    .into_iter()
                    .map(|value| SignalOrdinate::Real(SignalValue::Scalar(<$quantity>::new(value))))
                    .collect(),
                interpolation,
                extrapolation,
                $variant,
            )?;
            signal_object($builtin, $class, signal)
        }
    };
}

macro_rules! vector_time_signal {
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $label:literal, $summary:literal, $class:expr,
        $quantity:ty, $variant:path
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &TIME_SAMPLE_INPUTS,
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
                                    keywords = "fea,signal,time,history,samples,vector,typed,si",
                                    descriptor($descriptor),
                                    integer_capabilities(TIME_SAMPLE_INTEGER_CAPABILITIES),
                                    builtin_path = "crate::builtins::fea::signals::time_samples"
                                )]
        pub async fn $function(
            id: String,
            times: Value,
            values: Value,
            interpolation: Value,
            extrapolation: Value,
        ) -> BuiltinResult<Value> {
            let times = finite_numeric_vector($builtin, &times, "times")?;
            let values = finite_numeric_vector3_samples($builtin, &values, times.len(), "values")?;
            let (interpolation, extrapolation) =
                policies($builtin, &interpolation, &extrapolation)?;
            let signal = time_signal(
                id,
                SignalShape::Vector3,
                times,
                values
                    .into_iter()
                    .map(|sample| {
                        SignalOrdinate::Real(SignalValue::Vector3(
                            sample.map(|value| <$quantity>::new(value)),
                        ))
                    })
                    .collect(),
                interpolation,
                extrapolation,
                $variant,
            )?;
            signal_object($builtin, $class, signal)
        }
    };
}

vector_time_signal!(
    displacement_time_samples,
    DISPLACEMENT_TIME_DESCRIPTOR,
    DISPLACEMENT_TIME_SIGNATURES,
    "fea.DisplacementSignal.fromTimeSamples",
    "signal = fea.DisplacementSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a displacement history from time samples in meters.",
    DISPLACEMENT_SIGNAL_CLASS,
    Displacement,
    PhysicalSignal::Displacement
);
vector_time_signal!(
    rotation_time_samples,
    ROTATION_TIME_DESCRIPTOR,
    ROTATION_TIME_SIGNATURES,
    "fea.RotationSignal.fromTimeSamples",
    "signal = fea.RotationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a rotation history from time samples in radians.",
    ROTATION_SIGNAL_CLASS,
    Rotation,
    PhysicalSignal::Rotation
);
vector_time_signal!(
    velocity_time_samples,
    VELOCITY_TIME_DESCRIPTOR,
    VELOCITY_TIME_SIGNATURES,
    "fea.VelocitySignal.fromTimeSamples",
    "signal = fea.VelocitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a velocity history from time samples in meters per second.",
    VELOCITY_SIGNAL_CLASS,
    Velocity,
    PhysicalSignal::Velocity
);
vector_time_signal!(
    acceleration_time_samples,
    ACCELERATION_TIME_DESCRIPTOR,
    ACCELERATION_TIME_SIGNATURES,
    "fea.AccelerationSignal.fromTimeSamples",
    "signal = fea.AccelerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an acceleration history from time samples in meters per second squared.",
    ACCELERATION_SIGNAL_CLASS,
    Acceleration,
    PhysicalSignal::Acceleration
);
vector_time_signal!(
    angular_velocity_time_samples,
    ANGULAR_VELOCITY_TIME_DESCRIPTOR,
    ANGULAR_VELOCITY_TIME_SIGNATURES,
    "fea.AngularVelocitySignal.fromTimeSamples",
    "signal = fea.AngularVelocitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an angular-velocity history from time samples in radians per second.",
    ANGULAR_VELOCITY_SIGNAL_CLASS,
    AngularVelocity,
    PhysicalSignal::AngularVelocity
);
vector_time_signal!(
    angular_acceleration_time_samples,
    ANGULAR_ACCELERATION_TIME_DESCRIPTOR,
    ANGULAR_ACCELERATION_TIME_SIGNATURES,
    "fea.AngularAccelerationSignal.fromTimeSamples",
    "signal = fea.AngularAccelerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an angular-acceleration history from time samples in radians per second squared.",
    ANGULAR_ACCELERATION_SIGNAL_CLASS,
    AngularAcceleration,
    PhysicalSignal::AngularAcceleration
);
vector_time_signal!(
    force_time_samples,
    FORCE_TIME_DESCRIPTOR,
    FORCE_TIME_SIGNATURES,
    "fea.ForceSignal.fromTimeSamples",
    "signal = fea.ForceSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a force history from time samples in newtons.",
    FORCE_SIGNAL_CLASS,
    Force,
    PhysicalSignal::Force
);
vector_time_signal!(
    moment_time_samples,
    MOMENT_TIME_DESCRIPTOR,
    MOMENT_TIME_SIGNATURES,
    "fea.MomentSignal.fromTimeSamples",
    "signal = fea.MomentSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a moment history from time samples in newton-meters.",
    MOMENT_SIGNAL_CLASS,
    Moment,
    PhysicalSignal::Moment
);
vector_time_signal!(
    traction_time_samples,
    TRACTION_TIME_DESCRIPTOR,
    TRACTION_TIME_SIGNATURES,
    "fea.TractionSignal.fromTimeSamples",
    "signal = fea.TractionSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a traction history from time samples in pascals.",
    TRACTION_SIGNAL_CLASS,
    Traction,
    PhysicalSignal::Traction
);
scalar_time_signal!(
    pressure_time_samples,
    PRESSURE_TIME_DESCRIPTOR,
    PRESSURE_TIME_SIGNATURES,
    "fea.PressureSignal.fromTimeSamples",
    "signal = fea.PressureSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a pressure history from time samples in pascals.",
    PRESSURE_SIGNAL_CLASS,
    Pressure,
    PhysicalSignal::Pressure
);
vector_time_signal!(
    body_force_time_samples,
    BODY_FORCE_TIME_DESCRIPTOR,
    BODY_FORCE_TIME_SIGNATURES,
    "fea.BodyForceSignal.fromTimeSamples",
    "signal = fea.BodyForceSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a body-force-density history from time samples.",
    BODY_FORCE_SIGNAL_CLASS,
    BodyForceDensity,
    PhysicalSignal::BodyForceDensity
);
scalar_time_signal!(
    temperature_time_samples,
    TEMPERATURE_TIME_DESCRIPTOR,
    TEMPERATURE_TIME_SIGNATURES,
    "fea.TemperatureSignal.fromTimeSamples",
    "signal = fea.TemperatureSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an absolute-temperature history from time samples in kelvin.",
    TEMPERATURE_SIGNAL_CLASS,
    Temperature,
    PhysicalSignal::Temperature
);
scalar_time_signal!(
    temperature_difference_time_samples,
    TEMPERATURE_DIFFERENCE_TIME_DESCRIPTOR,
    TEMPERATURE_DIFFERENCE_TIME_SIGNATURES,
    "fea.TemperatureDifferenceSignal.fromTimeSamples",
    "signal = fea.TemperatureDifferenceSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a temperature-difference history from time samples in kelvin.",
    TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
    TemperatureDifference,
    PhysicalSignal::TemperatureDifference
);
scalar_time_signal!(
    heat_flux_time_samples,
    HEAT_FLUX_TIME_DESCRIPTOR,
    HEAT_FLUX_TIME_SIGNATURES,
    "fea.HeatFluxSignal.fromTimeSamples",
    "signal = fea.HeatFluxSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a heat-flux history from time samples.",
    HEAT_FLUX_SIGNAL_CLASS,
    HeatFlux,
    PhysicalSignal::HeatFlux
);
scalar_time_signal!(
    heat_generation_time_samples,
    HEAT_GENERATION_TIME_DESCRIPTOR,
    HEAT_GENERATION_TIME_SIGNATURES,
    "fea.HeatGenerationSignal.fromTimeSamples",
    "signal = fea.HeatGenerationSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a volumetric heat-generation history from time samples.",
    HEAT_GENERATION_SIGNAL_CLASS,
    HeatGenerationDensity,
    PhysicalSignal::HeatGenerationDensity
);
scalar_time_signal!(
    electric_potential_time_samples,
    ELECTRIC_POTENTIAL_TIME_DESCRIPTOR,
    ELECTRIC_POTENTIAL_TIME_SIGNATURES,
    "fea.ElectricPotentialSignal.fromTimeSamples",
    "signal = fea.ElectricPotentialSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an electric-potential history from time samples in volts.",
    ELECTRIC_POTENTIAL_SIGNAL_CLASS,
    ElectricPotential,
    PhysicalSignal::ElectricPotential
);
scalar_time_signal!(
    electric_current_time_samples,
    ELECTRIC_CURRENT_TIME_DESCRIPTOR,
    ELECTRIC_CURRENT_TIME_SIGNATURES,
    "fea.ElectricCurrentSignal.fromTimeSamples",
    "signal = fea.ElectricCurrentSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an electric-current history from time samples in amperes.",
    ELECTRIC_CURRENT_SIGNAL_CLASS,
    ElectricCurrent,
    PhysicalSignal::ElectricCurrent
);
scalar_time_signal!(
    electric_charge_time_samples,
    ELECTRIC_CHARGE_TIME_DESCRIPTOR,
    ELECTRIC_CHARGE_TIME_SIGNATURES,
    "fea.ElectricChargeSignal.fromTimeSamples",
    "signal = fea.ElectricChargeSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an electric-charge history from time samples in coulombs.",
    ELECTRIC_CHARGE_SIGNAL_CLASS,
    ElectricCharge,
    PhysicalSignal::ElectricCharge
);
scalar_time_signal!(
    current_density_time_samples,
    CURRENT_DENSITY_TIME_DESCRIPTOR,
    CURRENT_DENSITY_TIME_SIGNATURES,
    "fea.CurrentDensitySignal.fromTimeSamples",
    "signal = fea.CurrentDensitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an electric-current-density history from time samples.",
    CURRENT_DENSITY_SIGNAL_CLASS,
    ElectricCurrentDensity,
    PhysicalSignal::ElectricCurrentDensity
);
scalar_time_signal!(
    charge_density_time_samples,
    CHARGE_DENSITY_TIME_DESCRIPTOR,
    CHARGE_DENSITY_TIME_SIGNATURES,
    "fea.ChargeDensitySignal.fromTimeSamples",
    "signal = fea.ChargeDensitySignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create an electric-charge-density history from time samples.",
    CHARGE_DENSITY_SIGNAL_CLASS,
    ElectricChargeDensity,
    PhysicalSignal::ElectricChargeDensity
);
vector_time_signal!(
    magnetic_vector_potential_time_samples,
    MAGNETIC_VECTOR_POTENTIAL_TIME_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_TIME_SIGNATURES,
    "fea.MagneticVectorPotentialSignal.fromTimeSamples",
    "signal = fea.MagneticVectorPotentialSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a magnetic-vector-potential history from time samples.",
    MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
    MagneticVectorPotential,
    PhysicalSignal::MagneticVectorPotential
);
scalar_time_signal!(
    mass_flow_time_samples,
    MASS_FLOW_TIME_DESCRIPTOR,
    MASS_FLOW_TIME_SIGNATURES,
    "fea.MassFlowSignal.fromTimeSamples",
    "signal = fea.MassFlowSignal.fromTimeSamples(id, times, values, interpolation, extrapolation)",
    "Create a mass-flow history from time samples in kilograms per second.",
    MASS_FLOW_SIGNAL_CLASS,
    MassFlowRate,
    PhysicalSignal::MassFlowRate
);
