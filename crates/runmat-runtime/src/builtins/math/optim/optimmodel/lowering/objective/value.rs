//! Full exact objective value in original squared-residual units.
use super::*;
impl LeastSquares {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn value(
        &self,
        evaluated: &exact::Evaluated,
        program: &Program,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rational> {
        let mut value = self.constant.clone();
        for term in &self.terms {
            let node = program
                .source
                .nodes
                .get(&term.residual)
                .ok_or_else(|| policy.invalid("squared objective lost original residual"))?
                .node();
            for row in 0..evaluation::element_count(&node.shape, policy)? {
                let residual = evaluated
                    .value(term.residual, row, policy)?
                    .ok_or_else(|| policy.invalid("squared objective residual is unassigned"))?;
                let square = residual.mul(&residual, policy, c).await?;
                value = value
                    .add(&term.weight.exact.mul(&square, policy, c).await?, policy, c)
                    .await?;
                c.completed(1).await?;
            }
        }
        Ok(value)
    }
}
