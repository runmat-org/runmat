//! Terminal original squared-residual differential, not rounded C'Cr or a
//! second graph. Affine source pullback consumes only active original rows.
use super::*;
impl LeastSquares {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn seeds(
        &self,
        point: &exact::Evaluated,
        origin: &exact::Evaluated,
        program: &Program,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(Vec<exact::Seed>, Rational)> {
        let mut seeds = Vec::new();
        let mut cost = Rational::zero();
        let two = Rational::from_float(2.).expect("finite constant");
        for term in &self.terms {
            let shape = &program
                .source
                .nodes
                .get(&term.residual)
                .ok_or_else(|| policy.invalid("residual lost original source"))?
                .node()
                .shape;
            for row in 0..evaluation::element_count(shape, policy)? {
                let value = point
                    .value(term.residual, row, policy)?
                    .ok_or_else(|| policy.invalid("original residual is unassigned"))?;
                let offset = origin
                    .value(term.residual, row, policy)?
                    .ok_or_else(|| policy.invalid("original residual origin is unassigned"))?;
                let weight = two
                    .mul(&term.weight.exact, policy, c)
                    .await?
                    .mul(&value, policy, c)
                    .await?;
                let direction = value.sub(&offset, policy, c).await?;
                cost = cost
                    .add(&weight.mul(&direction, policy, c).await?, policy, c)
                    .await?;
                seeds.try_reserve(1).map_err(|e| policy.reserve(e))?;
                seeds.push(exact::Seed {
                    root: term.residual,
                    row,
                    value: weight,
                });
                c.completed(1).await?;
            }
        }
        Ok((seeds, cost))
    }
}
