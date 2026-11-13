//! Equation data owns original residual roots separately from hard constraints.
use super::*;
use runmat_gc::GcHandle;
mod assembly;
mod evidence;
mod formulation;
mod linear;
mod nonlinear;
mod publication;
mod run;
mod scalar;
mod validation;
pub(super) use formulation::Evidence;
pub(in crate::builtins::math::optim::optimmodel) use publication::publish;
pub(in crate::builtins::math::optim::optimmodel) use run::solve;
pub(super) enum Prepared {
    Linear(Program),
    Nonlinear(Model),
}
pub(in crate::builtins::math::optim::optimmodel) struct Model {
    pub(super) source: evaluation::Plan,
    pub(super) layout: layout::Layout,
    pub(super) evidence: Evidence,
}
impl Model {
    pub(super) async fn prepare(
        data: &runmat_value::optimization_model::EquationProblemData,
        requested: Option<runmat_builtins::OptimSolver>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Prepared> {
        c.completed(0).await?;
        let formulation = formulation::Formulation::prepare(data, policy, c).await?;
        let source = evaluation::Plan::prepare_roots(&formulation.roots, policy, c).await?;
        let layout = layout::Layout::prepare(&source, policy, c).await?;
        if layout.columns == 0 {
            return Err(policy.invalid("solve requires at least one original equation variable"));
        }
        let evidence = Evidence::prepare(formulation, &source, policy, c).await?;
        if requested.is_some_and(|solver| solver != runmat_builtins::OptimSolver::Lsqlin) {
            return Ok(Prepared::Nonlinear(Self {
                source,
                layout,
                evidence,
            }));
        }
        match engine::arrays(
            &source,
            &layout,
            None,
            &evidence.formulation.roots,
            policy,
            c,
        )
        .await?
        {
            Ok(engine::Arrays { arrays, facts }) => Ok(Prepared::Linear(Program {
                source,
                layout,
                selected: None,
                objective: objective::ModelObjective::Equations(evidence),
                arrays,
                facts,
            })),
            Err(Stop::NonAffine | Stop::UnsupportedFunction) => Ok(Prepared::Nonlinear(Self {
                source,
                layout,
                evidence,
            })),
            Err(stop) => Err(stop.error(policy)),
        }
    }
}
