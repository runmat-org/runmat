//! Per-column data scales from original affine residual rows. Exact pullback
//! combines shared/cancelling coefficients before any absolute value is taken.
use super::*;
impl LeastSquares {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn scale(
        &self,
        program: &Program,
        origin: &exact::Evaluated,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<super::super::proof::ObjectiveScale> {
        let mut linear = runmat_value::exact_capacity_vec(program.layout.columns)
            .map_err(|e| policy.storage(e))?;
        let mut curvature = runmat_value::exact_capacity_vec(program.layout.columns)
            .map_err(|e| policy.storage(e))?;
        for _ in 0..program.layout.columns {
            linear.push(Rational::zero());
            curvature.push(Rational::zero());
            c.completed(1).await?;
        }
        let mut reverse = exact::SparsePlan::new(program, policy, c).await?;
        for term in &self.terms {
            let root = term.residual;
            let count = evaluation::element_count(
                &program
                    .source
                    .nodes
                    .get(&root)
                    .ok_or_else(|| policy.invalid("residual lost original source"))?
                    .node()
                    .shape,
                policy,
            )?;
            let weight = Rational::from_usize(2)
                .mul(&term.weight.exact, policy, c)
                .await?;
            for row in 0..count {
                let offset = origin
                    .value(root, row, policy)?
                    .ok_or_else(|| policy.invalid("original residual origin is unassigned"))?;
                let gradient = reverse.row(program, origin, root, row, policy, c).await?;
                let mut norm = Rational::zero();
                for coefficient in gradient.values() {
                    norm = norm.add(&coefficient.abs(), policy, c).await?;
                    c.completed(1).await?;
                }
                for (column, coefficient) in gradient {
                    let coefficient = weight.mul(&coefficient.abs(), policy, c).await?;
                    linear[column] = linear[column]
                        .add(&coefficient.mul(&offset.abs(), policy, c).await?, policy, c)
                        .await?;
                    curvature[column] = curvature[column]
                        .add(&coefficient.mul(&norm, policy, c).await?, policy, c)
                        .await?;
                    c.completed(1).await?;
                }
                c.completed(1).await?;
            }
        }
        super::super::proof::ObjectiveScale::from_parts(linear, curvature, policy, c).await
    }
}
