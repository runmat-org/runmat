use crate::MirRvalue;
use runmat_hir::{HirExpr, SemanticError};

#[allow(dead_code)]
pub(crate) fn lower_expr(_expr: &HirExpr) -> Result<Option<MirRvalue>, SemanticError> {
    Ok(None)
}
