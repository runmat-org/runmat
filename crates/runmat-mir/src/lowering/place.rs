use crate::MirPlace;
use runmat_hir::{HirPlace, SemanticError};

#[allow(dead_code)]
pub(crate) fn lower_place(_place: &HirPlace) -> Result<Option<MirPlace>, SemanticError> {
    Ok(None)
}
