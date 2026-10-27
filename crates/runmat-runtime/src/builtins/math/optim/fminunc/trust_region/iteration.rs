//! Trust-radius acceptance and original accepted-state/model lifetimes.
use super::super::evaluation::{ObjectiveEvaluator, ObjectiveState};
use super::*;
use crate::execution::RootedValue;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim::fminunc) enum Termination {
    Optimality,
    Step,
    FunctionChange,
    PredictedChange,
    Limit,
    Observer,
    Numerical,
    ObjectiveLimit,
}
impl Termination {
    pub(super) fn flag(self) -> i32 {
        match self {
            Self::Optimality => 1,
            Self::Step => 2,
            Self::FunctionChange => 3,
            Self::PredictedChange => 5,
            Self::Limit => 0,
            Self::Observer => -1,
            Self::Numerical => -4,
            Self::ObjectiveLimit => -3,
        }
    }
    pub(super) fn description(self) -> &'static str {
        match self {
        Self::Optimality=>"First-order optimality satisfies OptimalityTolerance.",
        Self::Step=>"Trust-region step displacement satisfies StepTolerance; first-order optimality need not be satisfied.",
        Self::FunctionChange=>"Accepted objective change satisfies FunctionTolerance; first-order optimality need not be satisfied.",
        Self::PredictedChange=>"Positive model-predicted decrease satisfies FunctionTolerance; first-order optimality need not be satisfied.",
        Self::Limit=>"Iteration or function-evaluation limit reached before a stopping tolerance was satisfied.",
        Self::Observer=>"Optimization stopped by the output function.",
        Self::ObjectiveLimit=>"The evaluated objective reached ObjectiveLimit; this is a threshold stop, not a proof of unboundedness.",
        Self::Numerical=>"RunMat numerical failure: the trust-region model or subproblem could not produce a representable descent step; no local-minimum claim is made.",
    }
    }
}

pub(in crate::builtins::math::optim::fminunc) struct Outcome {
    pub(super) state: ObjectiveState,
    pub(super) model: Option<model::Model>,
    pub(super) iterations: usize,
    pub(super) evaluations: usize,
    pub(super) cg_iterations: usize,
    pub(super) step_size: f64,
    pub(super) termination: Termination,
}

enum Prepared {
    Cg(cg::Preconditioner),
    Direct(factorization::Factorization),
}
impl Prepared {
    async fn new(
        model: &mut model::Products<'_>,
        n: usize,
        controls: &Controls,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        match controls.subproblem {
            runmat_builtins::OptimSubproblemAlgorithm::Cg => {
                cg::Preconditioner::new(model, n, controls.bandwidth, cooperation)
                    .await
                    .map(|value| value.map(Self::Cg))
            }
            runmat_builtins::OptimSubproblemAlgorithm::Factorization => {
                factorization::Factorization::new(model, n, cooperation)
                    .await
                    .map(|value| value.map(Self::Direct))
            }
        }
    }
    async fn step(
        &self,
        model: &mut model::Products<'_>,
        gradient: &[f64],
        radius: f64,
        controls: &Controls,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<cg::Step>> {
        match self {
            Self::Cg(preconditioner) => {
                cg::step(
                    model,
                    preconditioner,
                    gradient,
                    radius,
                    controls,
                    cooperation,
                )
                .await
            }
            Self::Direct(factorization) => {
                factorization
                    .step(model, gradient, radius, cooperation)
                    .await
            }
        }
    }
}

async fn supplied_model(
    source: &ModelSource,
    raw: RootedValue,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<model::Model> {
    match source {
        ModelSource::Objective => model::Model::matrix(raw, n, cooperation).await,
        ModelSource::Multiply { function } => Ok(model::Model::multiply(raw, function.clone())),
        ModelSource::GradientDifferences { .. } => {
            unreachable!("difference source never requests a third callback output")
        }
    }
}

pub(in crate::builtins::math::optim::fminunc) async fn minimize(
    function: &Value,
    x: Vec<f64>,
    shape: &[usize],
    scalar: bool,
    options: &FminuncOptions,
    observer: Option<&Value>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let controls = options
        .trust_region
        .as_ref()
        .expect("trust-region dispatch has active controls");
    let n = x.len();
    let pattern = match &controls.source {
        ModelSource::GradientDifferences {
            pattern: Some(value),
        } => Some(differences::Pattern::prepare(value, n, cooperation, model::policy()).await?),
        _ => None,
    };
    let third = !matches!(controls.source, ModelSource::GradientDifferences { .. });
    let mut evaluator = ObjectiveEvaluator::new(
        function,
        shape,
        scalar,
        true,
        options.max_fun_evals,
        cooperation,
    );
    let (mut state, mut raw) = evaluator.evaluate_analytic(&x, third).await?;
    let mut radius = norm(&x).max(1.0);
    if !radius.is_finite() {
        radius = x.iter().map(|value| value.abs()).fold(1.0, f64::max);
    }
    let mut model = None;
    let mut iterations = 0;
    let mut cg_iterations = 0usize;
    let mut step_size = 0.0;
    let mut termination = None;
    if publication::observe(
        observer,
        common::output_callback::OutputState::Init,
        publication::Snapshot {
            state: &state,
            shape,
            scalar,
            iterations,
            evaluations: evaluator.func_count,
            cg_iterations,
            step_size,
            radius,
        },
        evaluator.cooperation,
    )
    .await?
    {
        termination = Some(Termination::Observer);
    }
    publication::progress(
        options.display,
        true,
        publication::Snapshot {
            state: &state,
            shape,
            scalar,
            iterations,
            evaluations: evaluator.func_count,
            cg_iterations,
            step_size,
            radius,
        },
        evaluator.cooperation.runtime(),
    )?;
    while termination.is_none() {
        if state.f <= options.objective_limit {
            termination = Some(Termination::ObjectiveLimit);
            break;
        }
        if state.grad.iter().map(|g| g.abs()).fold(0.0, f64::max) <= options.tol_fun {
            termination = Some(Termination::Optimality);
            break;
        }
        if iterations >= options.max_iter || evaluator.remaining_func_evals() == 0 {
            termination = Some(Termination::Limit);
            break;
        }
        let next = match &controls.source {
            ModelSource::GradientDifferences { .. } => {
                differences::build(&state, pattern.as_ref(), &mut evaluator).await?
            }
            source => Some(
                supplied_model(
                    source,
                    raw.take()
                        .expect("accepted objective retains its third output"),
                    n,
                    evaluator.cooperation,
                )
                .await?,
            ),
        };
        let Some(next) = next else {
            termination = Some(Termination::Limit);
            break;
        };
        model = Some(next);
        {
            // A view borrows this exact immutable coefficient generation. No
            // product below repeats CSC validation or owns a coefficient copy.
            let mut products = model
                .as_mut()
                .expect("constructed model")
                .products(evaluator.cooperation)
                .await?;
            if let Some(prepared) =
                Prepared::new(&mut products, n, controls, evaluator.cooperation).await?
            {
                loop {
                    if iterations >= options.max_iter || evaluator.remaining_func_evals() == 0 {
                        termination = Some(Termination::Limit);
                        break;
                    }
                    let Some(step) = prepared
                        .step(
                            &mut products,
                            &state.grad,
                            radius,
                            controls,
                            evaluator.cooperation,
                        )
                        .await?
                    else {
                        termination = Some(Termination::Numerical);
                        break;
                    };
                    cg_iterations = cg_iterations
                        .checked_add(step.iterations)
                        .ok_or_else(|| invalid("CG iteration count overflow"))?;
                    step_size = norm(&step.displacement);
                    if step_size <= options.tol_x * (1.0 + norm(&state.x)) {
                        termination = Some(Termination::Step);
                        break;
                    }
                    let mut product = zeros(n, evaluator.cooperation).await?;
                    products
                        .apply(&step.displacement, &mut product, evaluator.cooperation)
                        .await?;
                    let predicted = -dot(&state.grad, &step.displacement)
                        - 0.5 * dot(&step.displacement, &product);
                    if !(predicted > 0.0 && predicted.is_finite()) {
                        termination = Some(Termination::Numerical);
                        break;
                    }
                    if predicted <= options.function_tolerance * state.f.abs().max(1.0) {
                        termination = Some(Termination::PredictedChange);
                        break;
                    }
                    let mut trial = buffer(n)?;
                    for (x, step) in state.x.chunks(1024).zip(step.displacement.chunks(1024)) {
                        for (&x, &step) in x.iter().zip(step) {
                            let next = x + step;
                            if !next.is_finite() {
                                termination = Some(Termination::Numerical);
                                break;
                            }
                            trial.push(next);
                        }
                        if termination.is_some() {
                            break;
                        }
                        evaluator.cooperation.completed(x.len() as u64).await?;
                    }
                    if termination.is_some() {
                        break;
                    }
                    if evaluator.remaining_func_evals() == 0 {
                        termination = Some(Termination::Limit);
                        break;
                    }
                    let (candidate, candidate_model) =
                        evaluator.evaluate_analytic(&trial, third).await?;
                    iterations += 1;
                    let reduction = state.f - candidate.f;
                    let ratio = reduction / predicted;
                    let reached_limit = candidate.f <= options.objective_limit;
                    if !ratio.is_finite() && !reached_limit {
                        termination = Some(Termination::Numerical);
                        break;
                    }
                    if ratio < 0.25 {
                        radius *= 0.25;
                    } else if ratio > 0.75 && step.boundary {
                        radius = (radius * 2.0).min(f64::MAX / 4.0);
                    }
                    let accepted = reached_limit || ratio > 0.1;
                    if accepted {
                        let scale = state.f.abs().max(1.0);
                        state = candidate;
                        raw = candidate_model;
                        if reached_limit {
                            termination = Some(Termination::ObjectiveLimit);
                        } else if publication::optimality(&state.grad) <= options.tol_fun {
                            termination = Some(Termination::Optimality);
                        } else if reduction.abs() <= options.function_tolerance * scale {
                            termination = Some(Termination::FunctionChange);
                        }
                    }
                    // Rejected trial H/Hinfo drops here, never replacing the accepted
                    // model's owner. Observer state remains the actual accepted point.
                    publication::progress(
                        options.display,
                        false,
                        publication::Snapshot {
                            state: &state,
                            shape,
                            scalar,
                            iterations,
                            evaluations: evaluator.func_count,
                            cg_iterations,
                            step_size,
                            radius,
                        },
                        evaluator.cooperation.runtime(),
                    )?;
                    if publication::observe(
                        observer,
                        common::output_callback::OutputState::Iter,
                        publication::Snapshot {
                            state: &state,
                            shape,
                            scalar,
                            iterations,
                            evaluations: evaluator.func_count,
                            cg_iterations,
                            step_size,
                            radius,
                        },
                        evaluator.cooperation,
                    )
                    .await?
                    {
                        termination = Some(Termination::Observer);
                    }

                    if accepted || termination.is_some() {
                        break;
                    }
                }
            } else {
                termination = Some(Termination::Numerical);
            }
        }
        // The product-view scope has ended: every coefficient/Hinfo-slot
        // borrow is gone before model replacement, Outcome or publication.
    }
    publication::observe(
        observer,
        common::output_callback::OutputState::Done,
        publication::Snapshot {
            state: &state,
            shape,
            scalar,
            iterations,
            evaluations: evaluator.func_count,
            cg_iterations,
            step_size,
            radius,
        },
        evaluator.cooperation,
    )
    .await?;
    Ok(Outcome {
        state,
        model,
        iterations,
        evaluations: evaluator.func_count,
        cg_iterations,
        step_size,
        termination: termination.expect("terminal trust-region state"),
    })
}
