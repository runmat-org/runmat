//! One reusable physical constraint row for the AL multiplier transaction.
use super::super::{evaluation::generation::Evaluated, invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;
pub(super) struct Row {
    pub inequalities: Vec<f64>,
    pub equalities: Vec<f64>,
    extents: (usize, usize),
}
impl Row {
    pub fn new(evaluated: &Evaluated, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        let (inequalities, equalities) = evaluated.constraints.extents();
        Ok(Self {
            inequalities: POLICY.capacity(inequalities, cooperation.runtime())?,
            equalities: POLICY.capacity(equalities, cooperation.runtime())?,
            extents: (inequalities, equalities),
        })
    }
    pub async fn capture(
        &mut self,
        evaluated: &Evaluated,
        row: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let (inequalities, equalities) = evaluated.constraints.extents();
        if row >= evaluated.constraints.rows || (inequalities, equalities) != self.extents {
            return Err(invalid("AL constraint row extent changed"));
        }
        self.inequalities.clear();
        self.equalities.clear();
        for column in 0..inequalities {
            self.inequalities
                .push(evaluated.constraints.inequality(row, column));
            cooperation.completed(1).await?;
        }
        for column in 0..equalities {
            self.equalities
                .push(evaluated.constraints.equality(row, column));
            cooperation.completed(1).await?;
        }
        Ok(())
    }
}
