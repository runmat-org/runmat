use crate::builtins::common::uniform_scalar_output::UniformScalarError;

pub(super) fn map_error(error: UniformScalarError) -> crate::RuntimeError {
    match error {
        UniformScalarError::Materialization(reason) => {
            super::error::internal(format!("cellfun: {reason}"))
        }
        UniformScalarError::InvalidStorage(kind) => {
            super::error::internal(format!("cellfun: {kind} has no storage value"))
        }
        UniformScalarError::CharacterRank => super::error::uniform(
            "cellfun: character outputs with UniformOutput=true must be 2-D",
        ),
        UniformScalarError::CharacterCount => {
            super::error::uniform("cellfun: callback returned the wrong number of characters")
        }
        UniformScalarError::SizeOverflow => {
            super::error::internal("cellfun: character output size exceeds platform limits")
        }
        UniformScalarError::NonScalar => super::error::uniform(
            "cellfun: callback must return scalar numeric, logical, character, or complex values when UniformOutput is true",
        ),
        UniformScalarError::Heterogeneous => super::error::uniform(
            "cellfun: callback outputs with UniformOutput=true must have the same data type on every invocation",
        ),
    }
}
