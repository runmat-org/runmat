use runmat_analysis_core::{ExcitationAction, PhysicalSignal, SignalShape};

use super::{
    model::RealizedExcitationKind,
    realize::RealizationError,
    sample::{
        expect_scalar, expect_vector3, expect_vector3_or_direction, sample_signal,
        ExcitationSamplePoint,
    },
};

macro_rules! scalar_action {
    ($signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => sample_signal(signal, $sample, |value| value.canonical_value())
                .and_then(expect_scalar),
            _ => Err(quantity_mismatch()),
        }
    };
}

macro_rules! vector_action {
    ($signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => sample_signal(signal, $sample, |value| value.canonical_value())
                .and_then(expect_vector3),
            _ => Err(quantity_mismatch()),
        }
    };
}

macro_rules! directional_vector_action {
    ($action:expr, $signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => sample_signal(signal, $sample, |value| value.canonical_value())
                .and_then(|value| {
                    expect_vector3_or_direction(value, $action.excitation().direction)
                }),
            _ => Err(quantity_mismatch()),
        }
    };
}

pub(super) fn realize_kind(
    action: &ExcitationAction,
    signal: &PhysicalSignal,
    sample: ExcitationSamplePoint<'_>,
) -> Result<RealizedExcitationKind, RealizationError> {
    Ok(match action {
        ExcitationAction::Force(_) => RealizedExcitationKind::Force(directional_vector_action!(
            action,
            signal,
            PhysicalSignal::Force,
            newtons,
            sample
        )?),
        ExcitationAction::Moment(_) => RealizedExcitationKind::Moment(directional_vector_action!(
            action,
            signal,
            PhysicalSignal::Moment,
            newton_meters,
            sample
        )?),
        ExcitationAction::Traction(_) => RealizedExcitationKind::Traction(vector_action!(
            signal,
            PhysicalSignal::Traction,
            pascals,
            sample
        )?),
        ExcitationAction::Pressure(_) => RealizedExcitationKind::Pressure(scalar_action!(
            signal,
            PhysicalSignal::Pressure,
            pascals,
            sample
        )?),
        ExcitationAction::BodyForce(_)
            if matches!(signal.shape(), SignalShape::TopologyField { .. }) =>
        {
            let PhysicalSignal::BodyForceDensity(signal) = signal else {
                return Err(quantity_mismatch());
            };
            RealizedExcitationKind::BodyForceField(super::field::NodalVectorField::sample(
                signal,
                sample,
                |value| value.canonical_value(),
            )?)
        }
        ExcitationAction::BodyForce(_) => RealizedExcitationKind::BodyForce(vector_action!(
            signal,
            PhysicalSignal::BodyForceDensity,
            newtons_per_cubic_meter,
            sample
        )?),
        ExcitationAction::HeatGeneration(action)
            if matches!(signal.shape(), SignalShape::TopologyField { .. }) =>
        {
            use runmat_analysis_core::{
                CoordinateFrame, ExcitationDistribution, ExcitationKinematics,
            };
            if action.frame != CoordinateFrame::Global
                || action.distribution != ExcitationDistribution::TopologyField
                || action.kinematics != ExcitationKinematics::FixedFrame
            {
                return Err(RealizationError::new(
                    "element scalar density requires global fixed-frame topology-field distribution",
                ));
            }
            let PhysicalSignal::HeatGenerationDensity(signal) = signal else {
                return Err(quantity_mismatch());
            };
            RealizedExcitationKind::HeatGenerationField(super::field::ElementScalarField::sample(
                signal,
                sample,
                |value| value.canonical_value(),
            )?)
        }
        ExcitationAction::HeatGeneration(_) => {
            RealizedExcitationKind::HeatGeneration(scalar_action!(
                signal,
                PhysicalSignal::HeatGenerationDensity,
                watts_per_cubic_meter,
                sample
            )?)
        }
        ExcitationAction::HeatFlux(_) => RealizedExcitationKind::HeatFlux(scalar_action!(
            signal,
            PhysicalSignal::HeatFlux,
            watts_per_square_meter,
            sample
        )?),
        ExcitationAction::TemperatureField(_) => RealizedExcitationKind::TemperatureField(
            scalar_action!(signal, PhysicalSignal::Temperature, kelvin, sample)?,
        ),
        ExcitationAction::AmbientTemperature(_) => RealizedExcitationKind::AmbientTemperature(
            scalar_action!(signal, PhysicalSignal::Temperature, kelvin, sample)?,
        ),
        ExcitationAction::TerminalVoltage(_) => RealizedExcitationKind::TerminalVoltage(
            scalar_action!(signal, PhysicalSignal::ElectricPotential, volts, sample)?,
        ),
        ExcitationAction::ElectricCurrent(_) => RealizedExcitationKind::ElectricCurrent(
            scalar_action!(signal, PhysicalSignal::ElectricCurrent, amperes, sample)?,
        ),
        ExcitationAction::ElectricCharge(_) => RealizedExcitationKind::ElectricCharge(
            scalar_action!(signal, PhysicalSignal::ElectricCharge, coulombs, sample)?,
        ),
        ExcitationAction::ElectricChargeDensity(_) => {
            RealizedExcitationKind::ElectricChargeDensity(scalar_action!(
                signal,
                PhysicalSignal::ElectricChargeDensity,
                coulombs_per_cubic_meter,
                sample
            )?)
        }
        ExcitationAction::CurrentDensity(_) => {
            RealizedExcitationKind::CurrentDensity(vector_action!(
                signal,
                PhysicalSignal::ElectricCurrentDensity,
                amperes_per_square_meter,
                sample
            )?)
        }
        ExcitationAction::MagneticVolumeCurrentDensity(_) => {
            let value = vector_action!(
                signal,
                PhysicalSignal::ElectricCurrentDensity,
                amperes_per_square_meter,
                sample
            )?;
            if value
                .real
                .iter()
                .chain(&value.imaginary)
                .any(|value| !value.is_finite())
            {
                return Err(RealizationError::new(
                    "magnetic volume current density must be finite",
                ));
            }
            RealizedExcitationKind::MagneticVolumeCurrentDensity(value)
        }
        ExcitationAction::MagneticVectorPotential(_) => {
            RealizedExcitationKind::MagneticVectorPotential(vector_action!(
                signal,
                PhysicalSignal::MagneticVectorPotential,
                webers_per_meter,
                sample
            )?)
        }
        ExcitationAction::AcousticPressure(_) => RealizedExcitationKind::AcousticPressure(
            scalar_action!(signal, PhysicalSignal::Pressure, pascals, sample)?,
        ),
        ExcitationAction::AcousticIncidentPressure(excitation) => {
            if signal.shape() != &SignalShape::Scalar
                || excitation.direction.is_some()
                || excitation.distribution
                    != runmat_analysis_core::ExcitationDistribution::UniformKinematic
                || excitation.kinematics != runmat_analysis_core::ExcitationKinematics::FixedFrame
                || excitation.projection.preserve_moment
            {
                return Err(RealizationError::new("incident pressure requires a fixed-frame uniform scalar pressure without direction or moment projection"));
            }
            RealizedExcitationKind::AcousticIncidentPressure(scalar_action!(
                signal,
                PhysicalSignal::Pressure,
                pascals,
                sample
            )?)
        }
        ExcitationAction::NormalVelocity(_) => RealizedExcitationKind::NormalVelocity(
            scalar_action!(signal, PhysicalSignal::Velocity, meters_per_second, sample)?,
        ),
        ExcitationAction::VolumeVelocity(excitation) => {
            if signal.shape() != &SignalShape::Scalar
                || excitation.direction.is_some()
                || excitation.distribution
                    != runmat_analysis_core::ExcitationDistribution::PreserveResultant
                || excitation.kinematics != runmat_analysis_core::ExcitationKinematics::FixedFrame
                || !excitation.projection.preserve_resultant
                || excitation.projection.preserve_moment
            {
                return Err(RealizationError::new("volume velocity requires a fixed-frame scalar total flow with resultant preservation and no direction or moment projection"));
            }
            RealizedExcitationKind::VolumeVelocity(scalar_action!(
                signal,
                PhysicalSignal::VolumeFlowRate,
                cubic_meters_per_second,
                sample
            )?)
        }
        ExcitationAction::InletVelocity(_) => {
            RealizedExcitationKind::InletVelocity(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Velocity,
                meters_per_second,
                sample
            )?)
        }
        ExcitationAction::MassFlow(_) => RealizedExcitationKind::MassFlow(scalar_action!(
            signal,
            PhysicalSignal::MassFlowRate,
            kilograms_per_second,
            sample
        )?),
        ExcitationAction::WallMotion(_) => {
            RealizedExcitationKind::WallMotion(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Velocity,
                meters_per_second,
                sample
            )?)
        }
    })
}

fn quantity_mismatch() -> RealizationError {
    RealizationError::new("excitation action and physical signal quantity do not match")
}
