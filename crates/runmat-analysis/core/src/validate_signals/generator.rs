use crate::problem::signal::{
    GeneratorArgument, GeneratorDomain, SignalGenerator, SignalSamplingPolicy,
};

use super::validate_artifact;

pub(super) fn validate_generator(
    generator: &SignalGenerator,
    inputs: Option<&[crate::AuthoredInputDefinition]>,
) -> Result<(), String> {
    generator.domain.axis_descriptor(inputs)?;
    validate_generator_invocation(&generator.invocation)?;
    if matches!(
        &generator.domain,
        GeneratorDomain::StudyParameter { parameter_id, .. } if parameter_id.trim().is_empty()
    ) {
        return Err("study-parameter generators require a parameter id".to_string());
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

pub(crate) fn validate_generator_invocation(
    invocation: &crate::GeneratorInvocation,
) -> Result<(), String> {
    match &invocation.callable {
        crate::GeneratorCallable::ImportedFunction(identity) => {
            if identity.module_path.trim().is_empty()
                || !is_identifier(&identity.symbol)
                || identity.source_digest.len() != 64
                || !identity
                    .source_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(
                    "imported generator requires a module path, function symbol, and SHA-256 source digest"
                        .to_string(),
                );
            }
        }
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
    for argument in &invocation.arguments {
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
