use runmat_analysis_core::{ExcitationAction, PhysicalSignal};

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
            $variant(signal) => {
                sample_signal(signal, $sample, |value| value.$field).and_then(expect_scalar)
            }
            _ => Err(quantity_mismatch()),
        }
    };
}

macro_rules! vector_action {
    ($signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => {
                sample_signal(signal, $sample, |value| value.$field).and_then(expect_vector3)
            }
            _ => Err(quantity_mismatch()),
        }
    };
}

macro_rules! directional_vector_action {
    ($action:expr, $signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => {
                sample_signal(signal, $sample, |value| value.$field).and_then(|value| {
                    expect_vector3_or_direction(value, $action.excitation().direction)
                })
            }
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
        ExcitationAction::BodyForce(_) => RealizedExcitationKind::BodyForce(vector_action!(
            signal,
            PhysicalSignal::BodyForceDensity,
            newtons_per_cubic_meter,
            sample
        )?),
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
        ExcitationAction::CurrentDensity(_) => {
            RealizedExcitationKind::CurrentDensity(vector_action!(
                signal,
                PhysicalSignal::ElectricCurrentDensity,
                amperes_per_square_meter,
                sample
            )?)
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
        ExcitationAction::NormalVelocity(_) => RealizedExcitationKind::NormalVelocity(
            scalar_action!(signal, PhysicalSignal::Velocity, meters_per_second, sample)?,
        ),
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
