use std::collections::HashSet;

use crate::{
    problem::{quantity::FiniteQuantity, signal::*},
    AnalysisValidationError,
};

#[cfg(test)]
mod deferred_topology_tests;
mod generator;
#[cfg(test)]
mod topology_field_tests;
#[cfg(test)]
mod volume_flow_tests;

use generator::validate_generator;
pub(crate) use generator::validate_generator_invocation;

pub(crate) fn validate_signals(signals: &[PhysicalSignal]) -> Result<(), AnalysisValidationError> {
    validate_signals_with_inputs(signals, None)
}
pub(crate) fn validate_signals_with_inputs(
    signals: &[PhysicalSignal],
    inputs: Option<&[crate::AuthoredInputDefinition]>,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for signal in signals {
        let id = &signal.signal_id().0;
        if id.trim().is_empty() || !ids.insert(id.as_str()) {
            return invalid(id, "signal ids must be non-empty and unique");
        }
        let result = match signal {
            PhysicalSignal::Displacement(value) => validate_signal(value, inputs),
            PhysicalSignal::Rotation(value) => validate_signal(value, inputs),
            PhysicalSignal::Velocity(value) => validate_signal(value, inputs),
            PhysicalSignal::Acceleration(value) => validate_signal(value, inputs),
            PhysicalSignal::AngularVelocity(value) => validate_signal(value, inputs),
            PhysicalSignal::AngularAcceleration(value) => validate_signal(value, inputs),
            PhysicalSignal::Force(value) => validate_signal(value, inputs),
            PhysicalSignal::Moment(value) => validate_signal(value, inputs),
            PhysicalSignal::Traction(value) => validate_signal(value, inputs),
            PhysicalSignal::Pressure(value) => validate_signal(value, inputs),
            PhysicalSignal::BodyForceDensity(value) => validate_signal(value, inputs),
            PhysicalSignal::Temperature(value) => validate_signal(value, inputs),
            PhysicalSignal::TemperatureDifference(value) => validate_signal(value, inputs),
            PhysicalSignal::HeatFlux(value) => validate_signal(value, inputs),
            PhysicalSignal::HeatGenerationDensity(value) => validate_signal(value, inputs),
            PhysicalSignal::HeatRate(value) => validate_signal(value, inputs),
            PhysicalSignal::ElectricPotential(value) => validate_signal(value, inputs),
            PhysicalSignal::ElectricCurrent(value) => validate_signal(value, inputs),
            PhysicalSignal::ElectricCharge(value) => validate_signal(value, inputs),
            PhysicalSignal::ElectricCurrentDensity(value) => validate_signal(value, inputs),
            PhysicalSignal::ElectricChargeDensity(value) => validate_signal(value, inputs),
            PhysicalSignal::MagneticVectorPotential(value) => validate_signal(value, inputs),
            PhysicalSignal::MassFlowRate(value) => validate_signal(value, inputs),
            PhysicalSignal::VolumeFlowRate(value) => {
                if value.shape != SignalShape::Scalar {
                    Err("volume-flow-rate signals require scalar shape".into())
                } else {
                    validate_signal(value, inputs)
                }
            }
        };
        if let Err(detail) = result {
            return invalid(id, detail);
        }
    }
    Ok(())
}

pub fn validate_physical_signals(
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    validate_signals(signals)
}

fn invalid<T>(signal_id: &str, detail: impl Into<String>) -> Result<T, AnalysisValidationError> {
    Err(AnalysisValidationError::InvalidSignal {
        signal_id: signal_id.to_string(),
        detail: detail.into(),
    })
}

fn validate_signal<Q: FiniteQuantity>(
    signal: &Signal<Q>,
    inputs: Option<&[crate::AuthoredInputDefinition]>,
) -> Result<(), String> {
    if matches!(signal.shape, SignalShape::DeferredTopologyField { .. })
        && (!matches!(signal.representation, SignalRepresentation::Generator(_))
            || !matches!(signal.provenance, SignalProvenance::Authored))
    {
        return Err("deferred topology fields require an authored generator declaration".into());
    }
    let mut expected_shape = signal
        .shape
        .component_extents()
        .ok_or_else(|| "signal shape must be complete and non-empty where required".to_string())?;
    if let SignalShape::TopologyField {
        component_count, ..
    } = &signal.shape
    {
        if let Some(shape) = concrete_topology_shape(&signal.representation)? {
            if shape.len() != 2 || shape[0] == 0 || shape[1] != *component_count {
                return Err("topology-field values require exact [entity_count, component_count] tensors with positive entity count".to_string());
            }
            expected_shape = shape;
        }
    }
    // A topology axis indexes entities, not physical vector components. A scalar
    // field may retain a declared spatial frame but needs no component frame.
    let scalar_field = signal.shape.topology_component_count() == Some(1);
    if !expected_shape.is_empty() && !scalar_field && signal.frame.is_none() {
        return Err("vector, tensor, and topology-field signals require a frame".to_string());
    }
    if expected_shape.is_empty() && signal.frame.is_some() {
        return Err("scalar signals cannot declare a coordinate frame".to_string());
    }
    if let Some(frame) = &signal.frame {
        crate::validate_authored::validate_coordinate_frame(frame)
            .map_err(|error| error.to_string())?;
    }
    validate_evidence(&signal.evidence)?;
    validate_provenance(&signal.provenance)?;
    match &signal.representation {
        SignalRepresentation::Constant(value) => {
            validate_real_precision(signal.precision)?;
            validate_sample_count(&signal.evidence, 1)?;
            validate_declared_shape(validate_value(value)?, &expected_shape)
        }
        SignalRepresentation::Tabulated(table) => {
            validate_sample_count(&signal.evidence, table.knots.len())?;
            validate_table(table, signal.precision, &expected_shape, inputs)
        }
        SignalRepresentation::Waveform(waveform) => {
            validate_real_precision(signal.precision)?;
            validate_sample_count(&signal.evidence, 0)?;
            validate_waveform(waveform, &expected_shape)
        }
        SignalRepresentation::Artifact(artifact) => {
            validate_sample_count(&signal.evidence, artifact.sample_count)?;
            validate_artifact_with_inputs(artifact, inputs)?;
            match (&signal.shape, &artifact.topology_digest) {
                (SignalShape::TopologyField { topology_id, .. }, Some(digest))
                    if topology_id == digest =>
                {
                    Ok(())
                }
                (SignalShape::TopologyField { .. }, _) => Err(
                    "topology-field artifact must bind the declared topology digest".to_string(),
                ),
                (_, None) => Ok(()),
                (_, Some(_)) => {
                    Err("only topology-field signals may bind a topology digest".to_string())
                }
            }
        }
        SignalRepresentation::Generator(generator) => {
            validate_sample_count(&signal.evidence, 0)?;
            if !matches!(signal.provenance, SignalProvenance::Authored) {
                return Err(
                    "callable declarations are authored; generated provenance belongs to their materialized artifact"
                        .to_string(),
                );
            }
            validate_generator(generator, inputs)
        }
    }
}

// The mesh-bound entity count is inferred once from concrete data. Existing table and
// waveform validation then requires every ordinate/offset to have this exact shape;
// topology realization is responsible for matching the rows to the bound mesh.
fn concrete_topology_shape<Q: FiniteQuantity>(
    representation: &SignalRepresentation<Q>,
) -> Result<Option<Vec<usize>>, String> {
    let value = match representation {
        SignalRepresentation::Constant(value) => value,
        SignalRepresentation::Waveform(waveform) => &waveform.amplitude,
        SignalRepresentation::Tabulated(table) => {
            let knot = table
                .knots
                .first()
                .ok_or_else(|| "topology-field table requires ordinates".to_string())?;
            let ordinate = match &knot.value {
                SignalKnotValue::Continuous(value) => value,
                SignalKnotValue::Jump { left_limit, .. } => left_limit,
            };
            match ordinate {
                SignalOrdinate::Real(value) | SignalOrdinate::Complex { real: value, .. } => value,
            }
        }
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => return Ok(None),
    };
    if !matches!(value, SignalValue::Tensor { .. }) {
        return Err("concrete topology-field values must be tensors".to_string());
    }
    validate_value(value).map(Some)
}

fn validate_real_precision(precision: SignalPrecision) -> Result<(), String> {
    if matches!(precision, SignalPrecision::Real32 | SignalPrecision::Real64) {
        Ok(())
    } else {
        Err("real-valued representations require real precision".to_string())
    }
}

fn validate_sample_count(
    evidence: &SignalApproximationEvidence,
    expected: usize,
) -> Result<(), String> {
    if evidence.sample_count == expected {
        Ok(())
    } else {
        Err("evidence sample count must match the signal representation".to_string())
    }
}

fn validate_declared_shape(actual: Vec<usize>, expected: &[usize]) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err("signal values do not match the declared shape".to_string())
    }
}

fn validate_evidence(evidence: &SignalApproximationEvidence) -> Result<(), String> {
    for (name, value) in [
        ("max_absolute_error", evidence.max_absolute_error),
        ("max_relative_error", evidence.max_relative_error),
        (
            "derivative_max_absolute_error",
            evidence.derivative_max_absolute_error,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    Ok(())
}

fn validate_provenance(provenance: &SignalProvenance) -> Result<(), String> {
    match provenance {
        SignalProvenance::Authored => Ok(()),
        SignalProvenance::Imported { source_digest } if !source_digest.trim().is_empty() => Ok(()),
        SignalProvenance::Retained { source_digest }
            if source_digest.strip_prefix("sha256:").is_some_and(|encoded| {
                encoded.len() == 64
                    && encoded.bytes().all(|byte| {
                        byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                    })
            }) =>
        {
            Ok(())
        }
        SignalProvenance::Generated(generator)
            if !generator.project_revision.trim().is_empty()
                && !generator.callable_id.trim().is_empty()
                && !generator.frozen_input_digest.trim().is_empty() =>
        {
            Ok(())
        }
        SignalProvenance::Imported { .. } => {
            Err("imported signal provenance requires a source digest".to_string())
        }
        SignalProvenance::Retained { .. } => {
            Err("retained signal provenance requires a canonical source digest".to_string())
        }
        SignalProvenance::Generated(_) => Err(
            "generator provenance requires project revision, callable identity, and frozen-input digest"
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod retained_provenance_tests;

fn validate_value<Q: FiniteQuantity>(value: &SignalValue<Q>) -> Result<Vec<usize>, String> {
    match value {
        SignalValue::Scalar(value) if value.is_finite() => Ok(Vec::new()),
        SignalValue::Vector3(values) if values.iter().all(FiniteQuantity::is_finite) => Ok(vec![3]),
        SignalValue::Tensor { shape, values }
            if !shape.is_empty()
                && shape.iter().all(|extent| *extent > 0)
                && shape
                    .iter()
                    .try_fold(1usize, |size, extent| size.checked_mul(*extent))
                    == Some(values.len())
                && values.iter().all(FiniteQuantity::is_finite) =>
        {
            Ok(shape.clone())
        }
        _ => Err("values must be finite and have a non-empty, exact tensor shape".to_string()),
    }
}

fn validate_ordinate<Q: FiniteQuantity>(
    ordinate: &SignalOrdinate<Q>,
) -> Result<(Vec<usize>, bool), String> {
    match ordinate {
        SignalOrdinate::Real(value) => Ok((validate_value(value)?, false)),
        SignalOrdinate::Complex { real, imaginary } => {
            let real_shape = validate_value(real)?;
            let imaginary_shape = validate_value(imaginary)?;
            if real_shape != imaginary_shape {
                return Err(
                    "complex real and imaginary values must have identical shape".to_string(),
                );
            }
            Ok((real_shape, true))
        }
    }
}

fn validate_table<Q: FiniteQuantity>(
    table: &TabulatedSignal<Q>,
    precision: SignalPrecision,
    declared_shape: &[usize],
    inputs: Option<&[crate::AuthoredInputDefinition]>,
) -> Result<(), String> {
    table.coordinate_field(inputs)?;
    if let TabulatedSignalDomain::StudyParameter { parameter_id, .. } = &table.domain {
        if parameter_id.trim().is_empty() {
            return Err("study-parameter axis requires a parameter id".to_string());
        }
    }
    if table.knots.is_empty()
        || table.knots.iter().any(|knot| {
            !knot.abscissa.is_finite() || (knot.abscissa == 0.0 && knot.abscissa.is_sign_negative())
        })
        || table
            .knots
            .windows(2)
            .any(|pair| pair[0].abscissa >= pair[1].abscissa)
    {
        return Err("tabulated knots must have finite, strictly increasing abscissae".to_string());
    }
    if matches!(table.domain, TabulatedSignalDomain::FrequencyHertz)
        && table.knots.iter().any(|knot| knot.abscissa < 0.0)
    {
        return Err("frequency values cannot be negative".to_string());
    }
    let mut expected_shape = None;
    let mut expected_complex = None;
    for (index, knot) in table.knots.iter().enumerate() {
        if matches!(knot.value, SignalKnotValue::Jump { .. })
            && (index == 0 || index + 1 == table.knots.len())
        {
            return Err("jump knots must be interior so both one-sided limits exist".to_string());
        }
        let mut validate_one = |ordinate: &SignalOrdinate<Q>| -> Result<(), String> {
            let (shape, complex) = validate_ordinate(ordinate)?;
            validate_declared_shape(shape.clone(), declared_shape)?;
            if expected_shape
                .as_ref()
                .is_some_and(|expected| expected != &shape)
            {
                return Err("all signal ordinates must have the same shape".to_string());
            }
            expected_shape.get_or_insert(shape);
            if expected_complex.is_some_and(|expected| expected != complex) {
                return Err(
                    "all signal ordinates must use the same real or complex representation"
                        .to_string(),
                );
            }
            expected_complex.get_or_insert(complex);
            Ok(())
        };
        match &knot.value {
            SignalKnotValue::Continuous(value) => validate_one(value)?,
            SignalKnotValue::Jump {
                left_limit,
                right_limit,
                ..
            } => {
                validate_one(left_limit)?;
                validate_one(right_limit)?;
            }
        }
    }
    let has_complex = expected_complex.unwrap_or(false);
    if has_complex
        && !matches!(
            precision,
            SignalPrecision::Complex64 | SignalPrecision::Complex128
        )
    {
        return Err("complex ordinates require complex precision".to_string());
    }
    if !has_complex
        && matches!(
            precision,
            SignalPrecision::Complex64 | SignalPrecision::Complex128
        )
    {
        return Err("complex precision requires complex ordinates".to_string());
    }
    Ok(())
}

fn validate_waveform<Q: FiniteQuantity>(
    waveform: &StandardWaveform<Q>,
    declared_shape: &[usize],
) -> Result<(), String> {
    let shape = validate_value(&waveform.amplitude)?;
    validate_declared_shape(shape.clone(), declared_shape)?;
    if let Some(offset) = &waveform.offset {
        if validate_value(offset)? != shape {
            return Err("waveform amplitude and offset must have identical shape".to_string());
        }
    }
    if !waveform.start_seconds.canonical_value().is_finite()
        || !waveform.phase_radians.canonical_value().is_finite()
        || waveform
            .duration_seconds
            .map(|quantity| quantity.canonical_value())
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || waveform
            .frequency_hz
            .map(|quantity| quantity.canonical_value())
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || waveform
            .duty_cycle
            .is_some_and(|value| !(value.is_finite() && 0.0 < value && value <= 1.0))
    {
        return Err("waveform parameters must be finite and physically valid".to_string());
    }
    let valid_parameters = match waveform.kind {
        StandardWaveformKind::Step => {
            waveform
                .duration_seconds
                .map(|quantity| quantity.canonical_value())
                .is_none()
                && waveform
                    .frequency_hz
                    .map(|quantity| quantity.canonical_value())
                    .is_none()
                && waveform.phase_radians.canonical_value() == 0.0
                && waveform.duty_cycle.is_none()
        }
        StandardWaveformKind::Ramp
        | StandardWaveformKind::HalfSinePulse
        | StandardWaveformKind::TriangularPulse => {
            waveform
                .duration_seconds
                .map(|quantity| quantity.canonical_value())
                .is_some()
                && waveform
                    .frequency_hz
                    .map(|quantity| quantity.canonical_value())
                    .is_none()
                && waveform.phase_radians.canonical_value() == 0.0
                && waveform.duty_cycle.is_none()
        }
        StandardWaveformKind::Sine => {
            waveform
                .frequency_hz
                .map(|quantity| quantity.canonical_value())
                .is_some()
                && waveform.duty_cycle.is_none()
        }
        StandardWaveformKind::RectangularPulse => {
            waveform
                .duration_seconds
                .map(|quantity| quantity.canonical_value())
                .is_some()
                && waveform
                    .frequency_hz
                    .map(|quantity| quantity.canonical_value())
                    .is_none()
                && waveform.phase_radians.canonical_value() == 0.0
                && waveform.duty_cycle.is_some()
        }
    };
    if !valid_parameters {
        return Err("waveform kind has missing or inapplicable parameters".to_string());
    }
    Ok(())
}

fn validate_artifact(artifact: &SignalArtifactRef) -> Result<(), String> {
    validate_artifact_with_inputs(artifact, None)
}
fn validate_artifact_with_inputs(
    artifact: &SignalArtifactRef,
    inputs: Option<&[crate::AuthoredInputDefinition]>,
) -> Result<(), String> {
    artifact.domain.axis_descriptor(inputs)?;
    if artifact.artifact_id.trim().is_empty()
        || artifact.content_digest.trim().is_empty()
        || artifact.schema.trim().is_empty()
        || artifact.sample_count == 0
    {
        return Err(
            "signal artifact identity, digest, schema, and sample count are required".to_string(),
        );
    }
    let valid_domain = match &artifact.domain {
        SignalArtifactDomain::Constant => artifact.sample_count == 1,
        SignalArtifactDomain::TimeSeconds { start, end } => {
            start.is_finite() && end.is_finite() && start < end
        }
        SignalArtifactDomain::FrequencyHertz { start, end } => {
            start.is_finite() && end.is_finite() && *start >= 0.0 && start < end
        }
        SignalArtifactDomain::StudyParameter {
            parameter_id,
            start,
            end,
            ..
        } => !parameter_id.trim().is_empty() && start.is_finite() && end.is_finite() && start < end,
    };
    if valid_domain {
        Ok(())
    } else {
        Err("signal artifact domain must be finite, ordered, and complete".to_string())
    }
}

#[cfg(test)]
mod tests;
