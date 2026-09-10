use crate::runtime_error::semantic_error;
use crate::sequence::SequenceResolutionContext;
use crate::RuntimeError;

#[derive(Clone, Debug)]
pub struct SubscriptReadRequest {
    pub sequence_use: runmat_types::SequenceUse,
    pub sequence_context: SequenceResolutionContext,
    pub indexing_context: runmat_types::ObjectIndexingContext,
    pub access: super::super::ObjectAccessContext,
}

impl SubscriptReadRequest {
    pub fn require_single(access: super::super::ObjectAccessContext) -> Self {
        Self {
            sequence_use: runmat_types::SequenceUse::RequireSingle,
            sequence_context: SequenceResolutionContext::default(),
            indexing_context: runmat_types::ObjectIndexingContext::Expression,
            access,
        }
    }

    pub(super) fn known_requested_outputs(&self) -> Result<Option<usize>, RuntimeError> {
        use runmat_types::SequenceUse;
        match self.sequence_use {
            SequenceUse::Discard => Ok(Some(0)),
            SequenceUse::RequireSingle => Ok(Some(1)),
            SequenceUse::SelectPrefix { count } => Ok(Some(count)),
            SequenceUse::SelectCurrentFunctionOutputs => self
                .sequence_context
                .current_function_output_count()
                .map(Some)
                .ok_or_else(|| unavailable("current function output count")),
            SequenceUse::SelectDestinationCardinality => self
                .sequence_context
                .destination_count()
                .map(Some)
                .ok_or_else(|| unavailable("destination cardinality")),
            SequenceUse::ExpandAll => Ok(None),
        }
    }
}

fn unavailable(context: &str) -> RuntimeError {
    semantic_error(
        "SequenceOutputContextUnavailable",
        format!("the {context} is unavailable for this subscript read"),
    )
}
