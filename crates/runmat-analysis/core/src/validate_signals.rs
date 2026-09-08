use std::collections::HashSet;

use crate::{
    problem::{quantity::FiniteQuantity, signal::*},
    AnalysisValidationError,
};

pub(crate) fn validate_signals(signals: &[PhysicalSignal]) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for signal in signals {
        let id = &signal.signal_id().0;
        if id.trim().is_empty() || !ids.insert(id.as_str()) {
            return invalid(id, "signal ids must be non-empty and unique");
        }
        let result = match signal {
            PhysicalSignal::Displacement(value) => validate_signal(value),
            PhysicalSignal::Rotation(value) => validate_signal(value),
            PhysicalSignal::Velocity(value) => validate_signal(value),
            PhysicalSignal::Acceleration(value) => validate_signal(value),
            PhysicalSignal::AngularVelocity(value) => validate_signal(value),
            PhysicalSignal::AngularAcceleration(value) => validate_signal(value),
            PhysicalSignal::Force(value) => validate_signal(value),
            PhysicalSignal::Moment(value) => validate_signal(value),
            PhysicalSignal::Traction(value) => validate_signal(value),
            PhysicalSignal::Pressure(value) => validate_signal(value),
            PhysicalSignal::BodyForceDensity(value) => validate_signal(value),
            PhysicalSignal::Temperature(value) => validate_signal(value),
            PhysicalSignal::TemperatureDifference(value) => validate_signal(value),
            PhysicalSignal::HeatFlux(value) => validate_signal(value),
            PhysicalSignal::HeatGenerationDensity(value) => validate_signal(value),
            PhysicalSignal::ElectricPotential(value) => validate_signal(value),
            PhysicalSignal::ElectricCurrent(value) => validate_signal(value),
            PhysicalSignal::ElectricCharge(value) => validate_signal(value),
            PhysicalSignal::ElectricCurrentDensity(value) => validate_signal(value),
            PhysicalSignal::ElectricChargeDensity(value) => validate_signal(value),
            PhysicalSignal::MagneticVectorPotential(value) => validate_signal(value),
            PhysicalSignal::MassFlowRate(value) => validate_signal(value),
        };
        if let Err(detail) = result {
            return invalid(id, detail);
        }
    }
    Ok(())
}

fn invalid<T>(signal_id: &str, detail: impl Into<String>) -> Result<T, AnalysisValidationError> {
    Err(AnalysisValidationError::InvalidSignal {
        signal_id: signal_id.to_string(),
        detail: detail.into(),
    })
}

fn validate_signal<Q: FiniteQuantity>(signal: &Signal<Q>) -> Result<(), String> {
    let expected_shape = signal
        .shape
        .component_extents()
        .ok_or_else(|| "signal shape must be complete and non-empty where required".to_string())?;
    if matches!(&signal.shape, SignalShape::TopologyField { .. })
        && !matches!(
            &signal.representation,
            SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_)
        )
    {
        return Err("topology-field signals must be artifact-backed or generated".to_string());
    }
    if !expected_shape.is_empty() && signal.frame.is_none() {
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
            validate_sample_count(&signal.evidence, table.values.len())?;
            validate_table(table, signal.precision, &expected_shape)
        }
        SignalRepresentation::Waveform(waveform) => {
            validate_real_precision(signal.precision)?;
            validate_sample_count(&signal.evidence, 0)?;
            validate_waveform(waveform, &expected_shape)
        }
        SignalRepresentation::Artifact(artifact) => {
            validate_sample_count(&signal.evidence, artifact.sample_count)?;
            validate_artifact(artifact)?;
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
            validate_generator(generator)
        }
    }
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
        SignalProvenance::Generated(_) => Err(
            "generator provenance requires project revision, callable identity, and frozen-input digest"
                .to_string(),
        ),
    }
}

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
) -> Result<(), String> {
    let axis = match &table.axis {
        SignalAxis::TimeSeconds(values) | SignalAxis::FrequencyHertz(values) => values,
        SignalAxis::StudyParameter {
            parameter_id,
            canonical_values,
        } => {
            if parameter_id.trim().is_empty() {
                return Err("study-parameter axis requires a parameter id".to_string());
            }
            canonical_values
        }
    };
    if axis.is_empty()
        || axis.len() != table.values.len()
        || axis.iter().any(|value| !value.is_finite())
        || axis.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(
            "tabulated axis must be finite, strictly increasing, and match values".to_string(),
        );
    }
    if matches!(table.axis, SignalAxis::FrequencyHertz(_)) && axis.iter().any(|value| *value < 0.0)
    {
        return Err("frequency values cannot be negative".to_string());
    }
    let mut expected_shape = None;
    let mut has_complex = false;
    for ordinate in &table.values {
        let (shape, complex) = validate_ordinate(ordinate)?;
        validate_declared_shape(shape.clone(), declared_shape)?;
        if expected_shape
            .as_ref()
            .is_some_and(|expected| expected != &shape)
        {
            return Err("all signal ordinates must have the same shape".to_string());
        }
        expected_shape.get_or_insert(shape);
        has_complex |= complex;
    }
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
    let mut previous_breakpoint = None;
    for breakpoint in &table.breakpoints {
        if !breakpoint.abscissa.is_finite()
            || breakpoint.abscissa < axis[0]
            || breakpoint.abscissa > axis[axis.len() - 1]
        {
            return Err("breakpoints must lie on the tabulated domain".to_string());
        }
        if previous_breakpoint.is_some_and(|previous| breakpoint.abscissa <= previous) {
            return Err("breakpoints must be strictly increasing and unique".to_string());
        }
        if axis
            .binary_search_by(|sample| sample.total_cmp(&breakpoint.abscissa))
            .is_err()
        {
            return Err("breakpoints must identify tabulated axis samples".to_string());
        }
        previous_breakpoint = Some(breakpoint.abscissa);
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
    if !waveform.start_seconds.is_finite()
        || !waveform.phase_radians.is_finite()
        || waveform
            .duration_seconds
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || waveform
            .frequency_hz
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || waveform
            .duty_cycle
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err("waveform parameters must be finite and physically valid".to_string());
    }
    if matches!(waveform.kind, StandardWaveformKind::Sine) && waveform.frequency_hz.is_none() {
        return Err("sine waveforms require frequency_hz".to_string());
    }
    Ok(())
}

fn validate_artifact(artifact: &SignalArtifactRef) -> Result<(), String> {
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
        } => !parameter_id.trim().is_empty() && start.is_finite() && end.is_finite() && start < end,
    };
    if valid_domain {
        Ok(())
    } else {
        Err("signal artifact domain must be finite, ordered, and complete".to_string())
    }
}

fn validate_generator(generator: &SignalGenerator) -> Result<(), String> {
    match &generator.callable {
        crate::GeneratorCallable::ProjectFunction { qualified_name } => {
            let parts = qualified_name.split('.').collect::<Vec<_>>();
            if parts.len() < 2 || !parts.iter().all(|part| is_identifier(part)) {
                return Err("project generator requires a qualified function name".to_string());
            }
        }
        crate::GeneratorCallable::BoundFunction { display_name, .. }
            if display_name.trim().is_empty() =>
        {
            return Err("bound generator requires a display name".to_string())
        }
        crate::GeneratorCallable::BoundFunction { .. } => {}
    }
    if matches!(
        &generator.domain,
        GeneratorDomain::StudyParameter { parameter_id } if parameter_id.trim().is_empty()
    ) {
        return Err("study-parameter generators require a parameter id".to_string());
    }
    for argument in &generator.arguments {
        match argument {
            GeneratorArgument::Real(value) if !value.is_finite() => {
                return Err("generator real arguments must be finite".to_string())
            }
            GeneratorArgument::RealVector(values)
                if values.is_empty() || values.iter().any(|value| !value.is_finite()) =>
            {
                return Err("generator vector arguments must be non-empty and finite".to_string())
            }
            GeneratorArgument::Artifact(artifact) => validate_artifact(artifact)?,
            _ => {}
        }
    }
    let result = match &generator.sampling {
        SignalSamplingPolicy::Uniform {
            start,
            end,
            sample_count,
        } if start.is_finite() && end.is_finite() && start < end && *sample_count >= 2 => Ok(()),
        SignalSamplingPolicy::Adaptive {
            start,
            end,
            max_absolute_error,
            max_samples,
        } if start.is_finite()
            && end.is_finite()
            && start < end
            && max_absolute_error.is_finite()
            && *max_absolute_error > 0.0
            && *max_samples >= 2 =>
        {
            Ok(())
        }
        SignalSamplingPolicy::CanonicalPoints(points)
            if points.len() >= 2
                && points.iter().all(|point| point.is_finite())
                && points.windows(2).all(|pair| pair[0] < pair[1]) =>
        {
            Ok(())
        }
        _ => Err("generator sampling policy is invalid".to_string()),
    };
    result?;
    let points_are_nonnegative = match &generator.sampling {
        SignalSamplingPolicy::Uniform { start, .. }
        | SignalSamplingPolicy::Adaptive { start, .. } => *start >= 0.0,
        SignalSamplingPolicy::CanonicalPoints(points) => {
            points.first().is_some_and(|first| *first >= 0.0)
        }
    };
    if generator.domain == GeneratorDomain::FrequencyHertz && !points_are_nonnegative {
        return Err("frequency generators cannot sample negative frequencies".to_string());
    }
    Ok(())
}

fn is_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use crate::{
        BreakpointSide, Force, GeneratorDomain, GeneratorEffectPolicy, GeneratorProvenance,
        PhysicalSignal, Signal, SignalApproximationEvidence, SignalAxis, SignalBreakpoint,
        SignalExtrapolation, SignalGenerator, SignalId, SignalInterpolation, SignalOrdinate,
        SignalPrecision, SignalProvenance, SignalRepresentation, SignalSamplingPolicy, SignalShape,
        SignalValue, TabulatedSignal,
    };

    use super::validate_signals;

    fn generated_force(provenance: SignalProvenance) -> PhysicalSignal {
        PhysicalSignal::Force(Signal::<Force> {
            signal_id: SignalId("loads.launch_force".to_string()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Generator(SignalGenerator {
                callable: crate::GeneratorCallable::ProjectFunction {
                    qualified_name: "loads.launch_force".to_string(),
                },
                arguments: Vec::new(),
                domain: GeneratorDomain::TimeSeconds,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.0,
                    end: 0.1,
                    sample_count: 1001,
                },
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
                effects: GeneratorEffectPolicy::Pure,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 0,
            },
            provenance,
        })
    }

    #[test]
    fn authored_generator_declaration_is_valid() {
        let signal = generated_force(SignalProvenance::Authored);

        validate_signals(&[signal]).expect("authored generator should validate");
    }

    #[test]
    fn bound_function_identity_is_typed_and_requires_a_display_name() {
        let mut signal = generated_force(SignalProvenance::Authored);
        set_bound_generator(&mut signal, 17, "pressureHistory");
        validate_signals(&[signal.clone()]).expect("semantic callable identity should validate");

        set_bound_generator(&mut signal, 17, "");
        assert!(validate_signals(&[signal.clone()]).is_err());
    }

    fn set_bound_generator(signal: &mut PhysicalSignal, function: u32, display_name: &str) {
        let PhysicalSignal::Force(signal) = signal else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut signal.representation else {
            unreachable!()
        };
        generator.callable = crate::GeneratorCallable::BoundFunction {
            function: runmat_types::ProgramFunctionId(function),
            display_name: display_name.into(),
        };
    }

    #[test]
    fn generator_cannot_claim_materialized_provenance() {
        let signal = generated_force(SignalProvenance::Generated(GeneratorProvenance {
            project_revision: "sha256:project".to_string(),
            callable_id: "loads.launch_force".to_string(),
            frozen_input_digest: "sha256:inputs".to_string(),
        }));

        assert!(validate_signals(&[signal]).is_err());
    }

    #[test]
    fn tabulated_breakpoints_are_ordered_axis_samples() {
        let mut signal = generated_force(SignalProvenance::Authored);
        let PhysicalSignal::Force(signal) = &mut signal else {
            unreachable!()
        };
        signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
            axis: SignalAxis::TimeSeconds(vec![0.0, 0.5, 1.0]),
            values: vec![0.0, 1.0, 2.0]
                .into_iter()
                .map(|value| SignalOrdinate::Real(SignalValue::Scalar(Force::new(value))))
                .collect(),
            interpolation: SignalInterpolation::StepRight,
            extrapolation: SignalExtrapolation::Reject,
            breakpoints: vec![SignalBreakpoint {
                abscissa: 0.5,
                side: BreakpointSide::Both,
            }],
        });
        signal.evidence.sample_count = 3;
        validate_signals(&[PhysicalSignal::Force(signal.clone())])
            .expect("axis-aligned breakpoint should validate");

        let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
            unreachable!()
        };
        table.breakpoints[0].abscissa = 0.25;
        let error = validate_signals(&[PhysicalSignal::Force(signal.clone())]).unwrap_err();
        assert!(error
            .to_string()
            .contains("breakpoints must identify tabulated axis samples"));

        let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
            unreachable!()
        };
        table.breakpoints = vec![
            SignalBreakpoint {
                abscissa: 0.5,
                side: BreakpointSide::Left,
            },
            SignalBreakpoint {
                abscissa: 0.5,
                side: BreakpointSide::Right,
            },
        ];
        let error = validate_signals(&[PhysicalSignal::Force(signal.clone())]).unwrap_err();
        assert!(error
            .to_string()
            .contains("breakpoints must be strictly increasing and unique"));
    }
}
