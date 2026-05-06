use crate::MirStmt;
use runmat_hir::{HirStmt, SemanticError};

#[allow(dead_code)]
pub(crate) fn lower_stmt(_stmt: &HirStmt) -> Result<Vec<MirStmt>, SemanticError> {
    Ok(Vec::new())
}
