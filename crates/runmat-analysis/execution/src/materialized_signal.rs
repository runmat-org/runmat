use runmat_analysis_core::{
    CanonicalQuantity, GeneratorDomain, PhysicalSignal, Signal, SignalKnot, SignalKnotValue,
    SignalOrdinate, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
    TabulatedSignal, TabulatedSignalDomain,
};
use runmat_analysis_fea::execution::{
    CanonicalSignalContent, CanonicalSignalKnot, CanonicalSignalRealization, CanonicalSignalValues,
};
use runmat_execution::Digest;

use crate::{AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult};

#[cfg(test)]
#[path = "materialized_signal/topology_tests.rs"]
mod topology_tests;

#[cfg(test)]
#[path = "materialized_signal/authority_tests.rs"]
mod authority_tests;

/// Derive the same provider-neutral numerical content identity used by generated realizations.
///
/// Authorship and callable provenance deliberately do not enter this digest. Knot discontinuities
/// and point-continuity choices are numerical content and therefore participate in the identity.
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
        PhysicalSignal::HeatRate(value) => canonicalize!(value),
        PhysicalSignal::ElectricPotential(value) => canonicalize!(value),
        PhysicalSignal::ElectricCurrent(value) => canonicalize!(value),
        PhysicalSignal::ElectricCharge(value) => canonicalize!(value),
        PhysicalSignal::ElectricCurrentDensity(value) => canonicalize!(value),
        PhysicalSignal::ElectricChargeDensity(value) => canonicalize!(value),
        PhysicalSignal::MagneticVectorPotential(value) => canonicalize!(value),
        PhysicalSignal::MassFlowRate(value) => canonicalize!(value),
        PhysicalSignal::VolumeFlowRate(value) => canonicalize!(value),
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
    let domain = match &table.domain {
        TabulatedSignalDomain::TimeSeconds => GeneratorDomain::TimeSeconds,
        TabulatedSignalDomain::FrequencyHertz => GeneratorDomain::FrequencyHertz,
        TabulatedSignalDomain::StudyParameter {
            parameter_id,
            quantity,
        } => GeneratorDomain::StudyParameter {
            parameter_id: parameter_id.clone(),
            quantity: *quantity,
        },
    };
    let knots = table
        .knots
        .iter()
        .map(|knot| match &knot.value {
            SignalKnotValue::Continuous(_) => CanonicalSignalKnot::continuous(knot.abscissa),
            SignalKnotValue::Jump { continuity, .. } => {
                CanonicalSignalKnot::jump(knot.abscissa, *continuity)
            }
        })
        .collect::<Vec<_>>();
    let ordinates = table
        .knots
        .iter()
        .flat_map(|knot| match &knot.value {
            SignalKnotValue::Continuous(value) => vec![value],
            SignalKnotValue::Jump {
                left_limit,
                right_limit,
                ..
            } => vec![left_limit, right_limit],
        })
        .collect::<Vec<_>>();
    let topology_entity_count = table_entity_count(&signal.shape, &ordinates)?;
    let values = canonical_table_values(
        &signal.shape,
        &ordinates,
        signal.precision,
        topology_entity_count,
    )?;
    CanonicalSignalContent {
        quantity,
        shape: &signal.shape,
        topology_entity_count: topology_entity_count.map(|count| count as u64),
        domain: &domain,
        knots: &knots,
        precision: signal.precision,
        values: &values,
        interpolation: table.interpolation,
        extrapolation: table.extrapolation,
        evidence: &signal.evidence,
    }
    .digest()
    .map_err(Into::into)
}

fn table_entity_count<Q>(
    shape: &SignalShape,
    ordinates: &[&SignalOrdinate<Q>],
) -> AnalysisBridgeResult<Option<usize>> {
    let SignalShape::TopologyField {
        component_count, ..
    } = shape
    else {
        return Ok(None);
    };
    let value = match ordinates.first().copied() {
        Some(SignalOrdinate::Real(value)) => value,
        Some(SignalOrdinate::Complex { real, .. }) => real,
        None => {
            return Err(AnalysisBridgeError::Invalid(
                "topology-field table is empty".into(),
            ))
        }
    };
    match value {
        SignalValue::Tensor { shape, .. }
            if shape.len() == 2
                && shape[0] > 0
                && shape[1] == *component_count
                && *component_count > 0 =>
        {
            Ok(Some(shape[0]))
        }
        _ => Err(AnalysisBridgeError::Invalid(
            "topology-field ordinate requires entity-by-component values".into(),
        )),
    }
}

fn canonical_table_values<Q: CanonicalQuantity>(
    shape: &SignalShape,
    ordinates: &[&SignalOrdinate<Q>],
    precision: runmat_analysis_core::SignalPrecision,
    topology_entity_count: Option<usize>,
) -> AnalysisBridgeResult<CanonicalSignalValues> {
    let mut real = Vec::new();
    let mut imaginary = Vec::new();
    for &ordinate in ordinates {
        match ordinate {
            SignalOrdinate::Real(value) if imaginary.is_empty() => {
                append_signal_value(shape, value, precision, topology_entity_count, &mut real)?;
            }
            SignalOrdinate::Complex {
                real: real_value,
                imaginary: imaginary_value,
            } if real.is_empty() || !imaginary.is_empty() => {
                append_signal_value(
                    shape,
                    real_value,
                    precision,
                    topology_entity_count,
                    &mut real,
                )?;
                append_signal_value(
                    shape,
                    imaginary_value,
                    precision,
                    topology_entity_count,
                    &mut imaginary,
                )?;
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
    topology_entity_count: Option<usize>,
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
        (
            SignalShape::TopologyField {
                component_count, ..
            },
            SignalValue::Tensor { shape, values },
        ) if topology_entity_count.is_some_and(|count| shape == &[count, *component_count])
            && declared_shape.realized_component_count(topology_entity_count)
                == Some(values.len()) =>
        {
            for value in values {
                output.push(canonical_component_bits(*value, precision)?);
            }
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
/// callable or runtime handle. Topology fields retain their association and exact entity rows;
/// the physics owner must validate their topology identity against the selected solver mesh.
pub fn materialize_signal_for_solver(
    authored: &PhysicalSignal,
    admitted: &AdmittedSignalGenerator,
    realization: &CanonicalSignalRealization,
) -> AnalysisBridgeResult<PhysicalSignal> {
    realization.validate()?;
    validate_realization_artifact_inventory(authored, realization)?;
    let axis = authored
        .generator()
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("signal materialization requires a generator axis".into())
        })?
        .domain
        .axis_descriptor(None)
        .map_err(AnalysisBridgeError::Invalid)?;
    if admitted.schema_version != 2
        || admitted.sample_points.descriptor() != axis
        || realization
            .domain
            .axis_descriptor(None)
            .map_err(AnalysisBridgeError::Invalid)?
            != axis
        || authored.signal_id() != &realization.signal_id
        || authored.quantity() != realization.quantity
        || authored.shape() != &realization.shape
        || authored.precision() != realization.precision
        || realization.materialization != admitted.materialization_identity
        || realization.provenance != admitted.provenance
        || authored
            .generator()
            .map(|value| value.invocation.callable.display_name())
            != Some(admitted.callable.qualified_name.as_str())
        || realization.sample_points().map(f64::to_bits).ne(admitted
            .sample_points
            .canonical_values()
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
        PhysicalSignal::HeatRate(signal) => materialize!(HeatRate, signal),
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
        PhysicalSignal::VolumeFlowRate(signal) => materialize!(VolumeFlowRate, signal),
    })
}

fn validate_realization_artifact_inventory(
    authored: &PhysicalSignal,
    realization: &CanonicalSignalRealization,
) -> AnalysisBridgeResult<()> {
    let generator = authored.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid(
            "only callable-backed signals can consume a generator realization".into(),
        )
    })?;
    let mut expected = std::collections::BTreeMap::new();
    for argument in &generator.invocation.arguments {
        let runmat_analysis_core::GeneratorArgument::Artifact(artifact) = argument else {
            continue;
        };
        let digest = artifact
            .content_digest
            .parse::<Digest>()
            .map_err(AnalysisBridgeError::Platform)?;
        let key = (artifact.artifact_id.as_str(), digest);
        if expected
            .insert(key, artifact.schema.as_str())
            .is_some_and(|schema| schema != artifact.schema)
        {
            return Err(AnalysisBridgeError::Identity(
                "authored immutable signal input declarations conflict",
            ));
        }
    }
    let actual = realization
        .input_admissions
        .iter()
        .map(|admission| {
            (
                (admission.artifact_id.as_str(), admission.logical_digest),
                admission.value_schema.as_str(),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    if actual != expected {
        return Err(AnalysisBridgeError::Identity(
            "canonical signal input admissions differ from authored immutable arguments",
        ));
    }
    Ok(())
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
    let entity_count = realization
        .topology_entity_count
        .map(usize::try_from)
        .transpose()
        .map_err(|_| AnalysisBridgeError::Invalid("signal entity count overflows".into()))?;
    let component_count = authored
        .shape
        .realized_component_count(entity_count)
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("signal shape or entity count is invalid".into())
        })?;
    let mut ordinates =
        ordinates::<Q>(&authored.shape, component_count, &realization.values)?.into_iter();
    let knots = realization
        .knots()
        .map(|knot| {
            let first = ordinates.next().ok_or(AnalysisBridgeError::Identity(
                "canonical signal realization has fewer ordinates than knots",
            ))?;
            Ok(match knot.continuity {
                None => SignalKnot::continuous(knot.abscissa(), first),
                Some(continuity) => {
                    let right_limit = ordinates.next().ok_or(AnalysisBridgeError::Identity(
                        "canonical signal jump is missing its right limit",
                    ))?;
                    SignalKnot::jump(knot.abscissa(), first, right_limit, continuity)
                }
            })
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    if ordinates.next().is_some() {
        return Err(AnalysisBridgeError::Identity(
            "canonical signal realization has excess ordinates",
        ));
    }
    Ok(Signal {
        signal_id: authored.signal_id.clone(),
        shape: authored.shape.clone(),
        frame: authored.frame.clone(),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: signal_domain(&realization.domain),
            knots,
            interpolation: realization.interpolation,
            extrapolation: realization.extrapolation,
        }),
        precision: realization.precision,
        evidence: realization.evidence.clone(),
        provenance: SignalProvenance::Generated(realization.provenance.clone()),
    })
}

fn signal_domain(domain: &GeneratorDomain) -> TabulatedSignalDomain {
    match domain {
        GeneratorDomain::TimeSeconds => TabulatedSignalDomain::TimeSeconds,
        GeneratorDomain::FrequencyHertz => TabulatedSignalDomain::FrequencyHertz,
        GeneratorDomain::StudyParameter {
            parameter_id,
            quantity,
        } => TabulatedSignalDomain::StudyParameter {
            parameter_id: parameter_id.clone(),
            quantity: *quantity,
        },
    }
}

fn ordinates<Q: CanonicalQuantity>(
    shape: &SignalShape,
    component_count: usize,
    values: &CanonicalSignalValues,
) -> AnalysisBridgeResult<Vec<SignalOrdinate<Q>>> {
    let make = |bits: &[u64]| -> AnalysisBridgeResult<Vec<SignalValue<Q>>> {
        if component_count == 0 || !bits.len().is_multiple_of(component_count) {
            return Err(AnalysisBridgeError::Identity(
                "canonical signal has an incomplete ordinate",
            ));
        }
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
        SignalShape::TopologyField {
            component_count, ..
        } if *component_count > 0
            && !values.is_empty()
            && values.len().is_multiple_of(*component_count) =>
        {
            Ok(SignalValue::Tensor {
                shape: vec![values.len() / component_count, *component_count],
                values,
            })
        }
        _ => Err(AnalysisBridgeError::Identity(
            "canonical signal sample differs from its declared shape",
        )),
    }
}
