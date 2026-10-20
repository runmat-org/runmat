use crate::builtins::common::uniform_scalar_output::{UniformScalarError, UniformScalarFailure};
mod collector;
pub(super) use collector::{normalize, Collector};

pub(super) fn map_error(error: UniformScalarError) -> crate::RuntimeError {
    match error.into_failure() {
        Err(error) => error,
        Ok(UniformScalarFailure::Materialization(reason)) => {
            super::error::internal(format!("cellfun: {reason}"))
        }
        Ok(UniformScalarFailure::InvalidStorage(kind)) => {
            super::error::internal(format!("cellfun: {kind} has no storage value"))
        }
        Ok(UniformScalarFailure::CharacterRank) => super::error::uniform(
            "cellfun: character outputs with UniformOutput=true must be 2-D",
        ),
        Ok(UniformScalarFailure::CharacterCount) => {
            super::error::uniform("cellfun: callback returned the wrong number of characters")
        }
        Ok(UniformScalarFailure::SizeOverflow) => {
            super::error::internal("cellfun: character output size exceeds platform limits")
        }
        Ok(UniformScalarFailure::NonScalar) => super::error::uniform(
            "cellfun: callback must return scalar numeric, logical, character, or complex values when UniformOutput is true",
        ),
        Ok(UniformScalarFailure::Heterogeneous) => super::error::uniform(
            "cellfun: callback outputs with UniformOutput=true must have the same data type on every invocation",
        ),
    }
}
