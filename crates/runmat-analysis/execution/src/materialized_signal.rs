use runmat_analysis_core::{
    CanonicalQuantity, GeneratorDomain, PhysicalSignal, Signal, SignalAxis, SignalOrdinate,
    SignalProvenance, SignalRepresentation, SignalShape, SignalValue, TabulatedSignal,
};
use runmat_analysis_fea::execution::{CanonicalSignalRealization, CanonicalSignalValues};

use crate::{AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult};

/// Convert a verified canonical realization into the typed table consumed by solver admission.
///
/// The canonical artifact remains the reusable execution result. This projection contains no
/// callable or runtime handle and is restricted to mesh-independent shapes; topology fields stay
/// artifact-backed until the selected solver has an explicit field projection contract.
pub fn materialize_signal_for_solver(
    authored: &PhysicalSignal,
    admitted: &AdmittedSignalGenerator,
    realization: &CanonicalSignalRealization,
) -> AnalysisBridgeResult<PhysicalSignal> {
    realization.validate()?;
    if authored.signal_id() != &realization.signal_id
        || authored.quantity() != realization.quantity
        || authored.shape() != &realization.shape
        || authored.precision() != realization.precision
        || realization.materialization != admitted.materialization_identity
        || authored
            .generator()
            .map(|value| value.qualified_function.as_str())
            != Some(admitted.callable.qualified_name.as_str())
        || realization.sample_points().map(f64::to_bits).ne(admitted
            .sample_points
            .iter()
            .copied()
            .map(f64::to_bits))
    {
        return Err(AnalysisBridgeError::Identity(
            "canonical signal realization differs from its authored signal",
        ));
    }
    macro_rules! materialize {
        ($variant:ident, $signal:expr) => {
            PhysicalSignal::$variant(materialize_typed_signal($signal, realization)?)
        };
    }
    Ok(match authored {
        PhysicalSignal::Displacement(signal) => materialize!(Displacement, signal),
        PhysicalSignal::Rotation(signal) => materialize!(Rotation, signal),
        PhysicalSignal::Velocity(signal) => materialize!(Velocity, signal),
        PhysicalSignal::Acceleration(signal) => materialize!(Acceleration, signal),
        PhysicalSignal::AngularVelocity(signal) => materialize!(AngularVelocity, signal),
        PhysicalSignal::AngularAcceleration(signal) => {
            materialize!(AngularAcceleration, signal)
        }
        PhysicalSignal::Force(signal) => materialize!(Force, signal),
        PhysicalSignal::Moment(signal) => materialize!(Moment, signal),
        PhysicalSignal::Traction(signal) => materialize!(Traction, signal),
        PhysicalSignal::Pressure(signal) => materialize!(Pressure, signal),
        PhysicalSignal::BodyForceDensity(signal) => materialize!(BodyForceDensity, signal),
        PhysicalSignal::Temperature(signal) => materialize!(Temperature, signal),
        PhysicalSignal::TemperatureDifference(signal) => {
            materialize!(TemperatureDifference, signal)
        }
        PhysicalSignal::HeatFlux(signal) => materialize!(HeatFlux, signal),
        PhysicalSignal::HeatGenerationDensity(signal) => {
            materialize!(HeatGenerationDensity, signal)
        }
        PhysicalSignal::ElectricPotential(signal) => materialize!(ElectricPotential, signal),
        PhysicalSignal::ElectricCurrent(signal) => materialize!(ElectricCurrent, signal),
        PhysicalSignal::ElectricCharge(signal) => materialize!(ElectricCharge, signal),
        PhysicalSignal::ElectricCurrentDensity(signal) => {
            materialize!(ElectricCurrentDensity, signal)
        }
        PhysicalSignal::ElectricChargeDensity(signal) => {
            materialize!(ElectricChargeDensity, signal)
        }
        PhysicalSignal::MagneticVectorPotential(signal) => {
            materialize!(MagneticVectorPotential, signal)
        }
        PhysicalSignal::MassFlowRate(signal) => materialize!(MassFlowRate, signal),
    })
}

fn materialize_typed_signal<Q: CanonicalQuantity>(
    authored: &Signal<Q>,
    realization: &CanonicalSignalRealization,
) -> AnalysisBridgeResult<Signal<Q>> {
    let SignalRepresentation::Generator(generator) = &authored.representation else {
        return Err(AnalysisBridgeError::Invalid(
            "only callable-backed signals can consume a generator realization".into(),
        ));
    };
    if generator.domain != realization.domain
        || generator.interpolation != realization.interpolation
        || generator.extrapolation != realization.extrapolation
    {
        return Err(AnalysisBridgeError::Identity(
            "canonical signal realization changes its generator sampling contract",
        ));
    }
    let component_count = match &authored.shape {
        SignalShape::Scalar => 1,
        SignalShape::Vector3 => 3,
        SignalShape::Tensor(extents) => extents
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| AnalysisBridgeError::Invalid("signal tensor shape overflows".into()))?,
        SignalShape::TopologyField { .. } => {
            return Err(AnalysisBridgeError::Invalid(
                "topology-field generators require mesh-aware artifact projection".into(),
            ))
        }
    };
    let values = ordinates::<Q>(&authored.shape, component_count, &realization.values)?;
    Ok(Signal {
        signal_id: authored.signal_id.clone(),
        shape: authored.shape.clone(),
        frame: authored.frame.clone(),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            axis: signal_axis(&realization.domain, realization.sample_points().collect()),
            values,
            interpolation: realization.interpolation,
            extrapolation: realization.extrapolation,
            breakpoints: Vec::new(),
        }),
        precision: realization.precision,
        evidence: realization.evidence.clone(),
        provenance: SignalProvenance::Generated(realization.provenance.clone()),
    })
}

fn signal_axis(domain: &GeneratorDomain, points: Vec<f64>) -> SignalAxis {
    match domain {
        GeneratorDomain::TimeSeconds => SignalAxis::TimeSeconds(points),
        GeneratorDomain::FrequencyHertz => SignalAxis::FrequencyHertz(points),
        GeneratorDomain::StudyParameter { parameter_id } => SignalAxis::StudyParameter {
            parameter_id: parameter_id.clone(),
            canonical_values: points,
        },
    }
}

fn ordinates<Q: CanonicalQuantity>(
    shape: &SignalShape,
    component_count: usize,
    values: &CanonicalSignalValues,
) -> AnalysisBridgeResult<Vec<SignalOrdinate<Q>>> {
    let make = |bits: &[u64]| -> AnalysisBridgeResult<Vec<SignalValue<Q>>> {
        bits.chunks_exact(component_count)
            .map(|sample| signal_value(shape, sample))
            .collect()
    };
    match values {
        CanonicalSignalValues::Real { values } => Ok(make(values)?
            .into_iter()
            .map(SignalOrdinate::Real)
            .collect()),
        CanonicalSignalValues::Complex { real, imaginary } => {
            let real = make(real)?;
            let imaginary = make(imaginary)?;
            if real.len() != imaginary.len() {
                return Err(AnalysisBridgeError::Identity(
                    "complex signal realization components differ in length",
                ));
            }
            Ok(real
                .into_iter()
                .zip(imaginary)
                .map(|(real, imaginary)| SignalOrdinate::Complex { real, imaginary })
                .collect())
        }
    }
}

fn signal_value<Q: CanonicalQuantity>(
    shape: &SignalShape,
    bits: &[u64],
) -> AnalysisBridgeResult<SignalValue<Q>> {
    let values = bits
        .iter()
        .map(|bits| Q::from_canonical_value(f64::from_bits(*bits)))
        .collect::<Vec<_>>();
    match shape {
        SignalShape::Scalar if values.len() == 1 => Ok(SignalValue::Scalar(values[0])),
        SignalShape::Vector3 if values.len() == 3 => {
            Ok(SignalValue::Vector3([values[0], values[1], values[2]]))
        }
        SignalShape::Tensor(extents) if !values.is_empty() => Ok(SignalValue::Tensor {
            shape: extents.clone(),
            values,
        }),
        _ => Err(AnalysisBridgeError::Identity(
            "canonical signal sample differs from its declared shape",
        )),
    }
}
