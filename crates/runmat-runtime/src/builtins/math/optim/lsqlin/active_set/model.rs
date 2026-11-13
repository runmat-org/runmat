//! One original residual owner and one original constraint-role assembly.
use super::*;
pub(super) struct Model {
    pub c: Matrix,
    pub d: Vec<f64>,
    pub weights: Vec<f64>,
    pub units: f64,
    pub rows: constraints::Constraints,
    pub index: CscRows,
}
impl Model {
    pub(super) async fn new(
        problem: Problem,
        units: objective::Units,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(Self, options::Controls, Vec<f64>)> {
        let Problem {
            c: matrix,
            d,
            a,
            b,
            aeq,
            beq,
            lb,
            ub,
            mut x0,
            controls,
            weights,
        } = problem;
        if x0.is_empty() || x0.len() != matrix.n {
            return Err(invalid(
                "active-set requires a nonempty x0 with one element per C column",
            ));
        }
        for ((x, &lower), &upper) in x0.iter_mut().zip(&lb).zip(&ub) {
            if !x.is_finite() {
                return Err(invalid("active-set x0 must be finite"));
            }
            *x = x.max(lower).min(upper);
            c.completed(1).await?;
        }
        let rows = constraints::Constraints::new(a, b, aeq, beq, lb, ub, c).await?;
        let index = CscRows::prepare(
            matrix.n,
            &matrix.colptr,
            &matrix.rowval,
            allocation,
            reserve,
            c,
        )
        .await?;
        let (weights, _, _) = weights.into_values(matrix.m, c).await?;
        Ok((
            Self {
                c: matrix,
                d,
                weights,
                units: units.multiplier(),
                rows,
                index,
            },
            controls,
            x0,
        ))
    }
    pub(super) fn objective(&self) -> super::super::residual::View<'_> {
        super::super::residual::View {
            c: &self.c,
            d: &self.d,
            weights: &self.weights,
            units: self.units,
        }
    }
    pub(super) fn view(&self) -> working_set::Rows<'_> {
        working_set::Rows {
            a: &self.rows.matrix,
            b: &self.rows.rhs,
            equalities: self.rows.equalities,
        }
    }
    pub(super) async fn evaluate(
        &self,
        x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<super::super::residual::Measurements>> {
        super::super::residual::measure(self.objective(), x, allocation, |_, _, _| Some(0.), c)
            .await
    }
}
