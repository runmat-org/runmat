use runmat_types::SequenceUse;
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;

pub use runmat_value::ValueSequence;

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub fn sequence_error_to_runtime(error: runmat_value::ValueSequenceError) -> RuntimeError {
    let identifier = match error {
        runmat_value::ValueSequenceError::TooManyOutputs { .. } => "OutputSequenceLimitExceeded",
        runmat_value::ValueSequenceError::TransientValue(_) => "NestedLegacyOutputList",
        runmat_value::ValueSequenceError::SharedOutputStorage => "SharedOutputStorage",
    };
    semantic_error(identifier, error.to_string())
}

/// Take the first output without detaching an exact physical backing claim.
/// Callers retain responsibility for their own required-cardinality check.
pub fn into_first_output(
    values: runmat_value::OutputListStorage,
) -> Result<Option<Value>, RuntimeError> {
    values.try_into_first().map_err(|_| {
        sequence_error_to_runtime(runmat_value::ValueSequenceError::SharedOutputStorage)
    })
}

/// Bridge only to legacy consumers whose ABI requires a bare Vec. A claimed
/// or shared output backing is refused rather than detached from its owner.
pub fn into_unbounded_values(sequence: ValueSequence) -> Result<Vec<Value>, RuntimeError> {
    into_unbounded_output_values(sequence.into_output_storage())
}

/// Compatibility bridge for an output owner whose consumer still requires an
/// ordinary Vec and cannot retain an exact physical backing claim.
pub fn into_unbounded_output_values(
    values: runmat_value::OutputListStorage,
) -> Result<Vec<Value>, RuntimeError> {
    values
        .try_into_owned()
        .map_err(|_| {
            sequence_error_to_runtime(runmat_value::ValueSequenceError::SharedOutputStorage)
        })?
        .try_into_unbounded_vec()
        .map_err(|_| {
            sequence_error_to_runtime(runmat_value::ValueSequenceError::SharedOutputStorage)
        })
}

/// Constructs the checked one-value result used by runtime call boundaries.
///
/// Keeping this fallible prevents fixtures and adapters from bypassing the
/// same transient-value validation applied to multi-output sequences.
pub fn single_value_sequence(value: Value) -> Result<ValueSequence, RuntimeError> {
    ValueSequence::single(value).map_err(sequence_error_to_runtime)
}

/// Adapts a scalar-result future at an explicitly single-valued call boundary.
pub fn single_value_future<F>(
    future: F,
) -> Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>>>>
where
    F: Future<Output = Result<Value, RuntimeError>> + 'static,
{
    Box::pin(async move { single_value_sequence(future.await?) })
}

pub(crate) mod destination;

pub use destination::{
    assignment_container_in_context, AssignmentStepSpec, DestinationCardinality, DestinationLayout,
    DestinationRange, PreparedSequenceDestination, PreparedSequenceEndpoint,
    SequenceDestinationBuilder, SequenceEndpointSpec,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct SequenceResolutionContext {
    current_function_outputs: Option<usize>,
    destination_cardinality: Option<usize>,
}

impl SequenceResolutionContext {
    pub const fn current_function_outputs(count: usize) -> Self {
        Self {
            current_function_outputs: Some(count),
            destination_cardinality: None,
        }
    }

    pub const fn destination_cardinality(count: usize) -> Self {
        Self {
            current_function_outputs: None,
            destination_cardinality: Some(count),
        }
    }

    pub const fn current_function_output_count(self) -> Option<usize> {
        self.current_function_outputs
    }

    pub const fn destination_count(self) -> Option<usize> {
        self.destination_cardinality
    }
}

pub trait ResolveValueSequence {
    /// Applies the language-level selection required by the surrounding
    /// expression context.
    fn resolve(
        self,
        use_context: SequenceUse,
        context: SequenceResolutionContext,
    ) -> Result<runmat_value::OutputListStorage, RuntimeError>;
}

impl ResolveValueSequence for ValueSequence {
    fn resolve(
        self,
        use_context: SequenceUse,
        context: SequenceResolutionContext,
    ) -> Result<runmat_value::OutputListStorage, RuntimeError> {
        let count = resolved_output_count(self.len(), use_context, context)?;
        if use_context == SequenceUse::Discard {
            return Ok(runmat_value::OutputListStorage::empty());
        }
        let mut values = self.into_output_storage();
        values.truncate(count);
        Ok(values)
    }
}

/// The one sequence-selection policy, independent of physical output storage.
/// Rooted iteration and legacy storage projection use identical validation.
pub(crate) fn resolved_output_count(
    len: usize,
    use_context: SequenceUse,
    context: SequenceResolutionContext,
) -> Result<usize, RuntimeError> {
    let count = match use_context {
        SequenceUse::Discard => return Ok(0),
        SequenceUse::ExpandAll => return Ok(len),
        SequenceUse::RequireSingle => 1,
        SequenceUse::SelectPrefix { count } => count,
        SequenceUse::SelectCurrentFunctionOutputs => {
            context.current_function_outputs.ok_or_else(|| {
                semantic_error(
                    "SequenceOutputContextUnavailable",
                    "the current function output count is unavailable for this value sequence",
                )
            })?
        }
        SequenceUse::SelectDestinationCardinality => {
            context.destination_cardinality.ok_or_else(|| {
                semantic_error(
                    "SequenceDestinationContextUnavailable",
                    "the destination cardinality is unavailable for this value sequence",
                )
            })?
        }
    };
    runmat_value::validate_output_count(count).map_err(sequence_error_to_runtime)?;
    if use_context == SequenceUse::RequireSingle && len != 1 {
        return Err(semantic_error(
            "CommaSeparatedListRequiresSingleValue",
            format!(
                "this expression requires one value, but the comma-separated list contains {len}"
            ),
        ));
    }
    if len < count {
        return Err(semantic_error(
            "CommaSeparatedListOutputShortage",
            format!("the comma-separated list contains {len} values, but {count} were requested"),
        ));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_rejects_plain_nonsingleton_and_selects_requested_prefix() {
        let error = ValueSequence::comma_separated(vec![Value::Num(1.0), Value::Num(2.0)])
            .unwrap()
            .resolve(
                SequenceUse::RequireSingle,
                SequenceResolutionContext::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires one value"));

        assert_eq!(
            ValueSequence::comma_separated(
                vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0),]
            )
            .unwrap()
            .resolve(
                SequenceUse::SelectPrefix { count: 2 },
                SequenceResolutionContext::default(),
            )
            .unwrap()
            .as_slice(),
            &[Value::Num(1.0), Value::Num(2.0)]
        );
    }

    #[test]
    fn empty_and_zero_output_contexts_are_deterministic() {
        assert!(ValueSequence::empty()
            .resolve(SequenceUse::Discard, SequenceResolutionContext::default())
            .unwrap()
            .is_empty());
        assert!(ValueSequence::empty()
            .resolve(
                SequenceUse::SelectPrefix { count: 0 },
                SequenceResolutionContext::default(),
            )
            .unwrap()
            .is_empty());
        assert!(ValueSequence::empty()
            .resolve(
                SequenceUse::RequireSingle,
                SequenceResolutionContext::default()
            )
            .is_err());
    }

    #[test]
    fn selection_rejects_output_counts_outside_the_artifact_domain() {
        let error = ValueSequence::empty()
            .resolve(
                SequenceUse::SelectPrefix {
                    count: runmat_value::MAX_VALUE_SEQUENCE_OUTPUTS + 1,
                },
                SequenceResolutionContext::default(),
            )
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:OutputSequenceLimitExceeded")
        );
    }
}
