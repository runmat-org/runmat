//! Existing conic feasibility substrate, under the same public iteration owner.
use super::*;
use clarabel::solver::{DefaultInfo, SupportedConeT};
use num_rational::BigRational;
struct Control<'a>(&'a mut options::Controls);
impl conic::solve::IterationControl for Control<'_> {
    fn accuracy(&self) -> f64 {
        self.0.feasibility.min(self.0.optimality)
    }
    fn remaining(&self) -> u32 {
        self.0.remaining()
    }
    fn record(
        &mut self,
        previous: &mut u32,
        info: &DefaultInfo<f64>,
        _scale: f64,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        self.0.record_phase(previous, info, runtime)
    }
}
struct Original<'a, V> {
    validation: &'a mut V,
    roles: &'a [conic::LinearRowRole],
}
impl<V: compiled::Validation> conic::solve::TerminalValidation for Original<'_, V> {
    async fn farkas(
        &mut self,
        weights: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation
            .certificate(
                compiled::Certificate::Farkas {
                    rows: compiled::Rows::original(self.roles),
                    multipliers: weights,
                },
                c,
            )
            .await
    }
    async fn ray(&mut self, _: &[BigRational], _: &mut ScalarCooperation) -> BuiltinResult<bool> {
        Ok(false)
    }
    async fn feasible(
        &mut self,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(false)
    }
}
pub(super) async fn solve<V: compiled::Validation>(
    model: &Model,
    seed: &[f64],
    controls: &mut options::Controls,
    validation: &mut V,
    c: &mut ScalarCooperation,
) -> BuiltinResult<conic::solve::Candidate> {
    let n = model.c.n;
    let mut pointers = capacity(
        n.checked_add(1)
            .ok_or_else(|| invalid("phase-I pointer overflow"))?,
    )?;
    let mut indices = capacity(n)?;
    let mut values = capacity(n)?;
    let mut gradient = capacity(n)?;
    pointers.push(0);
    for (column, &x) in seed.iter().enumerate() {
        indices.push(column);
        values.push(1.);
        gradient.push(-x);
        pointers.push(indices.len());
        c.completed(1).await?;
    }
    let metric = Matrix::new(n, n, pointers, indices, values);
    let mut cones = capacity(2)?;
    if model.rows.equalities != 0 {
        cones.push(SupportedConeT::ZeroConeT(model.rows.equalities));
    }
    if model.rows.rhs.len() > model.rows.equalities {
        cones.push(SupportedConeT::NonnegativeConeT(
            model.rows.rhs.len() - model.rows.equalities,
        ));
    }
    let problem = conic::Model {
        p: Some(&metric),
        a: &model.rows.matrix,
        b: &model.rows.rhs,
        f: &gradient,
        cones: &cones,
        identity: runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
        objective_scale: 1.,
    };
    conic::solve::candidate_validated(
        &problem,
        seed,
        &mut Control(controls),
        c,
        &mut Original {
            validation,
            roles: &model.rows.roles,
        },
    )
    .await
}
