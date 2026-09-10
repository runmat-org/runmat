use runmat_types::SequenceUse;
use runmat_value::Value;

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(crate) mod destination;

pub use destination::{
    AssignmentStepSpec, DestinationCardinality, DestinationLayout, DestinationRange,
    PreparedSequenceDestination, PreparedSequenceEndpoint, SequenceDestinationBuilder,
    SequenceEndpointSpec,
};

/// A source-level value sequence before its surrounding expression context
/// selects, expands, or discards entries.
#[derive(Debug)]
pub enum ValueSequence {
    Single(Value),
    CommaSeparated(Vec<Value>),
}

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

impl ValueSequence {
    pub fn single(value: Value) -> Self {
        Self::Single(value)
    }

    pub fn comma_separated(values: Vec<Value>) -> Self {
        Self::CommaSeparated(values)
    }

    /// Adapt the legacy callable return carrier exactly once at the execution
    /// boundary. `OutputList` must never escape this conversion as a language
    /// value or be stored in a local/workspace slot.
    pub(crate) fn from_callable_result(value: Value, requested_outputs: usize) -> Self {
        match value {
            Value::OutputList(values) => Self::CommaSeparated(values),
            _ if requested_outputs == 0 => Self::CommaSeparated(Vec::new()),
            value => Self::Single(value),
        }
    }

    pub fn resolve(
        self,
        use_context: SequenceUse,
        context: SequenceResolutionContext,
    ) -> Result<Vec<Value>, RuntimeError> {
        match use_context {
            SequenceUse::Discard => Ok(Vec::new()),
            SequenceUse::ExpandAll => Ok(self.into_values()),
            SequenceUse::RequireSingle => self.select_prefix(1, true),
            SequenceUse::SelectPrefix { count } => self.select_prefix(count, false),
            SequenceUse::SelectCurrentFunctionOutputs => {
                let count = context.current_function_outputs.ok_or_else(|| {
                    semantic_error(
                        "SequenceOutputContextUnavailable",
                        "the current function output count is unavailable for this value sequence",
                    )
                })?;
                self.select_prefix(count, false)
            }
            SequenceUse::SelectDestinationCardinality => {
                let count = context.destination_cardinality.ok_or_else(|| {
                    semantic_error(
                        "SequenceDestinationContextUnavailable",
                        "the destination cardinality is unavailable for this value sequence",
                    )
                })?;
                self.select_prefix(count, false)
            }
        }
    }

    fn into_values(self) -> Vec<Value> {
        match self {
            Self::Single(value) => vec![value],
            Self::CommaSeparated(values) => values,
        }
    }

    fn select_prefix(
        self,
        count: usize,
        require_exact_single: bool,
    ) -> Result<Vec<Value>, RuntimeError> {
        let mut values = self.into_values();
        if require_exact_single && values.len() != 1 {
            return Err(semantic_error(
                "CommaSeparatedListRequiresSingleValue",
                format!(
                    "this expression requires one value, but the comma-separated list contains {}",
                    values.len()
                ),
            ));
        }
        if values.len() < count {
            return Err(semantic_error(
                "CommaSeparatedListOutputShortage",
                format!(
                    "the comma-separated list contains {} values, but {count} were requested",
                    values.len()
                ),
            ));
        }
        values.truncate(count);
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_rejects_plain_nonsingleton_and_selects_requested_prefix() {
        let error = ValueSequence::comma_separated(vec![Value::Num(1.0), Value::Num(2.0)])
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
            .resolve(
                SequenceUse::SelectPrefix { count: 2 },
                SequenceResolutionContext::default(),
            )
            .unwrap(),
            vec![Value::Num(1.0), Value::Num(2.0)]
        );
    }

    #[test]
    fn empty_and_zero_output_contexts_are_deterministic() {
        assert!(ValueSequence::comma_separated(Vec::new())
            .resolve(SequenceUse::Discard, SequenceResolutionContext::default())
            .unwrap()
            .is_empty());
        assert!(ValueSequence::comma_separated(Vec::new())
            .resolve(
                SequenceUse::SelectPrefix { count: 0 },
                SequenceResolutionContext::default(),
            )
            .unwrap()
            .is_empty());
        assert!(ValueSequence::comma_separated(Vec::new())
            .resolve(
                SequenceUse::RequireSingle,
                SequenceResolutionContext::default()
            )
            .is_err());
    }
}
