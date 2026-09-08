use runmat_analysis_core::{
    CanonicalQuantity, GeneratorDomain, PhysicalSignal, Signal, SignalAxis, SignalOrdinate,
    SignalProvenance, SignalRepresentation, SignalShape, SignalValue, TabulatedSignal,
};
use runmat_analysis_fea::execution::{
    CanonicalSignalContent, CanonicalSignalRealization, CanonicalSignalValues,
};
use runmat_execution::Digest;

use crate::{AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult};

/// Derive the same provider-neutral numerical content identity used by generated realizations.
///
/// Authorship and callable provenance deliberately do not enter this digest. Breakpoints are not
/// silently discarded: generated tables currently cannot encode them, so a table with authored
/// breakpoint semantics is outside this equivalence contract.
pub fn canonical_tabulated_signal_content_digest(
    signal: &PhysicalSignal,
) -> AnalysisBridgeResult<Digest> {
    macro_rules! canonicalize {
        ($signal:expr) => {
            canonicalize_typed_table(signal.quantity(), $signal)
        };
    }
    match signal {
        PhysicalSignal::Displacement(value) => canonicalize!(value),
        PhysicalSignal::Rotation(value) => canonicalize!(value),
        PhysicalSignal::Velocity(value) => canonicalize!(value),
        PhysicalSignal::Acceleration(value) => canonicalize!(value),
        PhysicalSignal::AngularVelocity(value) => canonicalize!(value),
        PhysicalSignal::AngularAcceleration(value) => canonicalize!(value),
        PhysicalSignal::Force(value) => canonicalize!(value),
        PhysicalSignal::Moment(value) => canonicalize!(value),
        PhysicalSignal::Traction(value) => canonicalize!(value),
        PhysicalSignal::Pressure(value) => canonicalize!(value),
        PhysicalSignal::BodyForceDensity(value) => canonicalize!(value),
        PhysicalSignal::Temperature(value) => canonicalize!(value),
        PhysicalSignal::TemperatureDifference(value) => canonicalize!(value),
        PhysicalSignal::HeatFlux(value) => canonicalize!(value),
        PhysicalSignal::HeatGenerationDensity(value) => canonicalize!(value),
        PhysicalSignal::ElectricPotential(value) => canonicalize!(value),
        PhysicalSignal::ElectricCurrent(value) => canonicalize!(value),
        PhysicalSignal::ElectricCharge(value) => canonicalize!(value),
        PhysicalSignal::ElectricCurrentDensity(value) => canonicalize!(value),
        PhysicalSignal::ElectricChargeDensity(value) => canonicalize!(value),
        PhysicalSignal::MagneticVectorPotential(value) => canonicalize!(value),
        PhysicalSignal::MassFlowRate(value) => canonicalize!(value),
    }
}

fn canonicalize_typed_table<Q: CanonicalQuantity>(
    quantity: runmat_analysis_core::PhysicalQuantityKind,
    signal: &Signal<Q>,
) -> AnalysisBridgeResult<Digest> {
    let SignalRepresentation::Tabulated(table) = &signal.representation else {
        return Err(AnalysisBridgeError::Invalid(
            "canonical table content requires a tabulated signal".into(),
        ));
    };
    if !table.breakpoints.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "a table with authored breakpoints is not equivalent to a generated realization".into(),
        ));
    }
    let (domain, sample_points) = match &table.axis {
        SignalAxis::TimeSeconds(points) => (GeneratorDomain::TimeSeconds, points),
        SignalAxis::FrequencyHertz(points) => (GeneratorDomain::FrequencyHertz, points),
        SignalAxis::StudyParameter {
            parameter_id,
            canonical_values,
        } => (
            GeneratorDomain::StudyParameter {
                parameter_id: parameter_id.clone(),
            },
            canonical_values,
        ),
    };
    let sample_point_bits = sample_points
        .iter()
        .copied()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let values = canonical_table_values(&signal.shape, &table.values, signal.precision)?;
    CanonicalSignalContent {
        quantity,
        shape: &signal.shape,
        topology_entity_count: None,
        domain: &domain,
        sample_points: &sample_point_bits,
        precision: signal.precision,
        values: &values,
        interpolation: table.interpolation,
        extrapolation: table.extrapolation,
        evidence: &signal.evidence,
    }
    .digest()
    .map_err(Into::into)
}

fn canonical_table_values<Q: CanonicalQuantity>(
    shape: &SignalShape,
    ordinates: &[SignalOrdinate<Q>],
    precision: runmat_analysis_core::SignalPrecision,
) -> AnalysisBridgeResult<CanonicalSignalValues> {
    let mut real = Vec::new();
    let mut imaginary = Vec::new();
    for ordinate in ordinates {
        match ordinate {
            SignalOrdinate::Real(value) if imaginary.is_empty() => {
                append_signal_value(shape, value, precision, &mut real)?;
            }
            SignalOrdinate::Complex {
                real: real_value,
                imaginary: imaginary_value,
            } if real.is_empty() || !imaginary.is_empty() => {
                append_signal_value(shape, real_value, precision, &mut real)?;
                append_signal_value(shape, imaginary_value, precision, &mut imaginary)?;
            }
            _ => {
                return Err(AnalysisBridgeError::Invalid(
                    "tabulated signal mixes real and complex ordinates".into(),
                ))
            }
        }
    }
    match precision {
        runmat_analysis_core::SignalPrecision::Real32
        | runmat_analysis_core::SignalPrecision::Real64
            if imaginary.is_empty() =>
        {
            Ok(CanonicalSignalValues::Real { values: real })
        }
        runmat_analysis_core::SignalPrecision::Complex64
        | runmat_analysis_core::SignalPrecision::Complex128
            if !imaginary.is_empty() =>
        {
            Ok(CanonicalSignalValues::Complex { real, imaginary })
        }
        _ => Err(AnalysisBridgeError::Invalid(
            "tabulated ordinates differ from the declared signal precision".into(),
        )),
    }
}

fn append_signal_value<Q: CanonicalQuantity>(
    declared_shape: &SignalShape,
    value: &SignalValue<Q>,
    precision: runmat_analysis_core::SignalPrecision,
    output: &mut Vec<u64>,
) -> AnalysisBridgeResult<()> {
    match (declared_shape, value) {
        (SignalShape::Scalar, SignalValue::Scalar(value)) => {
            output.push(canonical_component_bits(*value, precision)?);
        }
        (SignalShape::Vector3, SignalValue::Vector3(values)) => {
            for value in values {
                output.push(canonical_component_bits(*value, precision)?);
            }
        }
        (SignalShape::Tensor(declared), SignalValue::Tensor { shape, values })
            if declared == shape =>
        {
            for value in values {
                output.push(canonical_component_bits(*value, precision)?);
            }
        }
        (SignalShape::TopologyField { .. }, _) => {
            return Err(AnalysisBridgeError::Invalid(
                "topology-field tables require mesh-aware canonicalization".into(),
            ))
        }
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "tabulated ordinate differs from the declared signal shape".into(),
            ))
        }
    }
    Ok(())
}

fn canonical_component_bits<Q: CanonicalQuantity>(
    value: Q,
    precision: runmat_analysis_core::SignalPrecision,
) -> AnalysisBridgeResult<u64> {
    let value = value.canonical_value();
    if !value.is_finite() {
        return Err(AnalysisBridgeError::Invalid(
            "tabulated signal contains a non-finite ordinate".into(),
        ));
    }
    let canonical = if value == 0.0 { 0.0 } else { value };
    Ok(match precision {
        runmat_analysis_core::SignalPrecision::Real32
        | runmat_analysis_core::SignalPrecision::Complex64 => f64::from(canonical as f32).to_bits(),
        runmat_analysis_core::SignalPrecision::Real64
        | runmat_analysis_core::SignalPrecision::Complex128 => canonical.to_bits(),
    })
}

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
            .map(|value| value.callable.display_name())
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
