use super::*;

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
        $function:ident, $builtin:literal, $class:expr, $quantity:ty, $variant:path $(,)?
    ) => {
        #[runtime_builtin(
                                    name = $builtin,
                                    binding_variant = "default",
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
        $function:ident, $builtin:literal, $class:expr, $quantity:ty, $variant:path $(,)?
    ) => {
        #[runtime_builtin(
                                    name = $builtin,
                                    binding_variant = "default",
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
    "fea.DisplacementSignal.fromTimeSamples",
    DISPLACEMENT_SIGNAL_CLASS,
    Displacement,
    PhysicalSignal::Displacement,
);
vector_time_signal!(
    rotation_time_samples,
    "fea.RotationSignal.fromTimeSamples",
    ROTATION_SIGNAL_CLASS,
    Rotation,
    PhysicalSignal::Rotation,
);
vector_time_signal!(
    velocity_time_samples,
    "fea.VelocitySignal.fromTimeSamples",
    VELOCITY_SIGNAL_CLASS,
    Velocity,
    PhysicalSignal::Velocity,
);
vector_time_signal!(
    acceleration_time_samples,
    "fea.AccelerationSignal.fromTimeSamples",
    ACCELERATION_SIGNAL_CLASS,
    Acceleration,
    PhysicalSignal::Acceleration,
);
vector_time_signal!(
    angular_velocity_time_samples,
    "fea.AngularVelocitySignal.fromTimeSamples",
    ANGULAR_VELOCITY_SIGNAL_CLASS,
    AngularVelocity,
    PhysicalSignal::AngularVelocity,
);
vector_time_signal!(
    angular_acceleration_time_samples,
    "fea.AngularAccelerationSignal.fromTimeSamples",
    ANGULAR_ACCELERATION_SIGNAL_CLASS,
    AngularAcceleration,
    PhysicalSignal::AngularAcceleration,
);
vector_time_signal!(
    force_time_samples,
    "fea.ForceSignal.fromTimeSamples",
    FORCE_SIGNAL_CLASS,
    Force,
    PhysicalSignal::Force,
);
vector_time_signal!(
    moment_time_samples,
    "fea.MomentSignal.fromTimeSamples",
    MOMENT_SIGNAL_CLASS,
    Moment,
    PhysicalSignal::Moment,
);
vector_time_signal!(
    traction_time_samples,
    "fea.TractionSignal.fromTimeSamples",
    TRACTION_SIGNAL_CLASS,
    Traction,
    PhysicalSignal::Traction,
);
scalar_time_signal!(
    pressure_time_samples,
    "fea.PressureSignal.fromTimeSamples",
    PRESSURE_SIGNAL_CLASS,
    Pressure,
    PhysicalSignal::Pressure,
);
vector_time_signal!(
    body_force_time_samples,
    "fea.BodyForceSignal.fromTimeSamples",
    BODY_FORCE_SIGNAL_CLASS,
    BodyForceDensity,
    PhysicalSignal::BodyForceDensity,
);
scalar_time_signal!(
    temperature_time_samples,
    "fea.TemperatureSignal.fromTimeSamples",
    TEMPERATURE_SIGNAL_CLASS,
    Temperature,
    PhysicalSignal::Temperature,
);
scalar_time_signal!(
    temperature_difference_time_samples,
    "fea.TemperatureDifferenceSignal.fromTimeSamples",
    TEMPERATURE_DIFFERENCE_SIGNAL_CLASS,
    TemperatureDifference,
    PhysicalSignal::TemperatureDifference,
);
scalar_time_signal!(
    heat_flux_time_samples,
    "fea.HeatFluxSignal.fromTimeSamples",
    HEAT_FLUX_SIGNAL_CLASS,
    HeatFlux,
    PhysicalSignal::HeatFlux,
);
scalar_time_signal!(
    heat_generation_time_samples,
    "fea.HeatGenerationSignal.fromTimeSamples",
    HEAT_GENERATION_SIGNAL_CLASS,
    HeatGenerationDensity,
    PhysicalSignal::HeatGenerationDensity,
);
scalar_time_signal!(
    electric_potential_time_samples,
    "fea.ElectricPotentialSignal.fromTimeSamples",
    ELECTRIC_POTENTIAL_SIGNAL_CLASS,
    ElectricPotential,
    PhysicalSignal::ElectricPotential,
);
scalar_time_signal!(
    electric_current_time_samples,
    "fea.ElectricCurrentSignal.fromTimeSamples",
    ELECTRIC_CURRENT_SIGNAL_CLASS,
    ElectricCurrent,
    PhysicalSignal::ElectricCurrent,
);
scalar_time_signal!(
    electric_charge_time_samples,
    "fea.ElectricChargeSignal.fromTimeSamples",
    ELECTRIC_CHARGE_SIGNAL_CLASS,
    ElectricCharge,
    PhysicalSignal::ElectricCharge,
);
vector_time_signal!(
    current_density_time_samples,
    "fea.CurrentDensitySignal.fromTimeSamples",
    CURRENT_DENSITY_SIGNAL_CLASS,
    ElectricCurrentDensity,
    PhysicalSignal::ElectricCurrentDensity,
);
scalar_time_signal!(
    charge_density_time_samples,
    "fea.ChargeDensitySignal.fromTimeSamples",
    CHARGE_DENSITY_SIGNAL_CLASS,
    ElectricChargeDensity,
    PhysicalSignal::ElectricChargeDensity,
);
vector_time_signal!(
    magnetic_vector_potential_time_samples,
    "fea.MagneticVectorPotentialSignal.fromTimeSamples",
    MAGNETIC_VECTOR_POTENTIAL_SIGNAL_CLASS,
    MagneticVectorPotential,
    PhysicalSignal::MagneticVectorPotential,
);
scalar_time_signal!(
    mass_flow_time_samples,
    "fea.MassFlowSignal.fromTimeSamples",
    MASS_FLOW_SIGNAL_CLASS,
    MassFlowRate,
    PhysicalSignal::MassFlowRate,
);
scalar_time_signal!(
    volume_flow_time_samples,
    "fea.VolumeFlowSignal.fromTimeSamples",
    VOLUME_FLOW_SIGNAL_CLASS,
    VolumeFlowRate,
    PhysicalSignal::VolumeFlowRate,
);
