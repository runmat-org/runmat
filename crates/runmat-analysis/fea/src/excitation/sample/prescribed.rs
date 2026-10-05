//! Shared quantity/shape-matched prescribed sampling; duplicate IDs fail closed.
use super::{
    expect_scalar, expect_vector3, sample_signal, ComplexScalar, ComplexVector3,
    ExcitationSamplePoint, RealizationError, SampledValue,
};
use runmat_analysis_core::{
    ElectricPotential, HeatFlux, PhysicalSignal, PrescribedValue, Pressure, Signal, Temperature,
    Velocity,
};

pub(crate) fn sample_prescribed_velocity(
    value: &PrescribedValue<[Velocity; 3]>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexVector3, RealizationError> {
    match value {
        PrescribedValue::Constant(value) => expect_vector3(SampledValue::Vector3(
            ComplexVector3::real(value.map(|v| v.meters_per_second)),
        )),
        PrescribedValue::Signal(_) => {
            let Some(PhysicalSignal::Velocity(signal)) = value.signal(signals) else {
                return Err(RealizationError::new(
                    "missing, duplicate or incompatible velocity signal",
                ));
            };
            if signal.frame != Some(runmat_analysis_core::CoordinateFrame::Global) {
                return Err(RealizationError::new(
                    "prescribed velocity requires global frame",
                ));
            }
            expect_vector3(sample_signal(signal, sample, |v| v.meters_per_second)?)
        }
    }
}

pub(crate) fn sample_prescribed_potential(
    value: &PrescribedValue<ElectricPotential>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexScalar, RealizationError> {
    sample_prescribed_scalar(
        value,
        signals,
        sample,
        |signal| match signal {
            PhysicalSignal::ElectricPotential(signal) => Some(signal),
            _ => None,
        },
        |value| value.volts,
    )
}

pub(crate) fn sample_prescribed_temperature(
    value: &PrescribedValue<Temperature>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexScalar, RealizationError> {
    sample_prescribed_scalar(
        value,
        signals,
        sample,
        |signal| match signal {
            PhysicalSignal::Temperature(signal) => Some(signal),
            _ => None,
        },
        |temperature| temperature.kelvin,
    )
}

pub(crate) fn sample_prescribed_heat_flux(
    value: &PrescribedValue<HeatFlux>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexScalar, RealizationError> {
    sample_prescribed_scalar(
        value,
        signals,
        sample,
        |signal| match signal {
            PhysicalSignal::HeatFlux(signal) => Some(signal),
            _ => None,
        },
        |heat_flux| heat_flux.watts_per_square_meter,
    )
}

pub(crate) fn sample_prescribed_pressure(
    value: &PrescribedValue<Pressure>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexScalar, RealizationError> {
    sample_prescribed_scalar(
        value,
        signals,
        sample,
        |signal| match signal {
            PhysicalSignal::Pressure(signal) => Some(signal),
            _ => None,
        },
        |pressure| pressure.pascals,
    )
}

fn sample_prescribed_scalar<'a, Q>(
    value: &'a PrescribedValue<Q>,
    signals: &'a [PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
    typed_signal: impl Fn(&'a PhysicalSignal) -> Option<&'a Signal<Q>>,
    component: impl Fn(&Q) -> f64,
) -> Result<ComplexScalar, RealizationError> {
    match value {
        PrescribedValue::Constant(value) => {
            expect_scalar(SampledValue::Scalar(ComplexScalar::real(component(value))))
        }
        PrescribedValue::Signal(signal_id) => {
            let mut matching = signals
                .iter()
                .filter(|candidate| candidate.signal_id() == signal_id);
            let candidate = matching
                .next()
                .ok_or_else(|| RealizationError::new("missing prescribed signal"))?;
            if matching.next().is_some() {
                return Err(RealizationError::new(
                    "duplicate prescribed signal identity",
                ));
            }
            let signal = typed_signal(candidate)
                .ok_or_else(|| RealizationError::new("incompatible prescribed signal quantity"))?;
            expect_scalar(sample_signal(signal, sample, component)?)
        }
    }
}
