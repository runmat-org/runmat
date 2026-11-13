use super::{MirExpressionRegion, MirExpressionStep, MirPreparedExpressionRegion};
use crate::{MirOperand, MirStmt, MirStmtKind};

impl MirExpressionRegion {
    pub fn from_lowered(
        statements: Vec<MirStmt>,
        result: MirOperand,
    ) -> Result<Self, runmat_hir::HirError> {
        let steps = Self::steps_from_lowered(statements)?;
        let region = Self { steps, result };
        region.validate().map_err(runmat_hir::HirError::new)?;
        Ok(region)
    }

    pub(crate) fn steps_from_lowered(
        statements: Vec<MirStmt>,
    ) -> Result<Vec<MirExpressionStep>, runmat_hir::HirError> {
        let mut steps = Vec::with_capacity(statements.len());
        for statement in statements {
            let step = match statement.kind {
                MirStmtKind::Assign {
                    place: crate::MirPlace::Local(local),
                    value,
                } => MirExpressionStep::Let {
                    local,
                    value,
                    span: statement.span,
                },
                MirStmtKind::CaptureSequence {
                    destination,
                    source,
                } => MirExpressionStep::CaptureSequence {
                    destination,
                    source,
                    span: statement.span,
                },
                _ => {
                    return Err(runmat_hir::HirError::new(
                        "expression preparation must lower to a straight-line region",
                    ));
                }
            };
            steps.push(step);
        }
        Ok(steps)
    }
}

impl MirPreparedExpressionRegion {
    pub fn from_lowered(
        statements: Vec<MirStmt>,
        result: MirOperand,
    ) -> Result<Self, runmat_hir::HirError> {
        let steps = MirExpressionRegion::steps_from_lowered(statements)?;
        let region = Self { steps, result };
        region.validate().map_err(runmat_hir::HirError::new)?;
        Ok(region)
    }
}
