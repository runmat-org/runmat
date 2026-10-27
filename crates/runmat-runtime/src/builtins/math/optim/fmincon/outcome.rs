//! Shared accepted-state result and observer ownership, not an algorithm dispatcher.
use super::{
    derivatives::Difference,
    evaluation::{Counts, Evaluator, Point},
    model::Linearization,
    observation::{self, Snapshot},
};
use crate::builtins::math::optim::common::output_callback::OutputState;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_value::Value;

pub(in crate::builtins::math::optim::fmincon) struct Parameters<'a> {
    pub max_iterations: usize,
    pub feasibility: f64,
    pub optimality: f64,
    pub step: f64,
    pub function: Option<f64>,
    pub differences: Difference,
    pub callback: Option<&'a Value>,
    pub display: runmat_builtins::OptimDisplay,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim::fmincon) enum Termination {
    Optimality,
    Step,
    Function,
    Limit,
    Observer,
    NoFeasiblePoint,
    Numerical,
    Time,
}
impl Termination {
    pub fn flag(self) -> i32 {
        match self {
            Self::Optimality => 1,
            Self::Step => 2,
            Self::Function => 3,
            Self::Limit | Self::Time => 0,
            Self::Observer => -1,
            Self::NoFeasiblePoint => -2,
            Self::Numerical => -4,
        }
    }
    pub fn description(self) -> &'static str {
        match self {
        Self::Optimality=>"Original nonlinear constraints and Lagrangian first-order checks satisfy the requested tolerances.",
        Self::Step=>"The bound-feasible step satisfies the selected algorithm's StepTolerance criterion and original constraints satisfy ConstraintTolerance; first-order optimality need not be satisfied.",
        Self::Function=>"The objective change satisfies the selected algorithm's FunctionTolerance criterion and original constraints satisfy ConstraintTolerance; first-order optimality need not be satisfied.",
        Self::Limit=>"An iteration or evaluated-point limit was reached.",
        Self::Time=>"The selected internal feasibility deadline was reached.",
        Self::Observer=>"The output function requested termination.",
        Self::NoFeasiblePoint=>"Local feasibility restoration found no improving feasible point; this is not a global nonlinear infeasibility certificate.",
        Self::Numerical=>"A reliable numerical constrained step could not be obtained; no local minimum is claimed.",
    }
    }
}
pub(in crate::builtins::math::optim::fmincon) struct Outcome {
    pub algorithm: runmat_builtins::OptimAlgorithm,
    pub x: Vec<f64>,
    pub point: Option<Point>,
    pub constraints: Option<Linearization>,
    pub multipliers: Vec<f64>,
    pub multipliers_available: bool,
    pub hessian: Option<CscMatrix<f64>>,
    pub iterations: usize,
    pub counts: Counts,
    pub termination: Termination,
    pub violation: Option<f64>,
    pub optimality: Option<f64>,
    pub step: f64,
    pub alpha: Option<f64>,
}

impl Outcome {
    pub(super) async fn observe(
        &self,
        evaluator: &mut Evaluator<'_>,
        parameters: &Parameters<'_>,
        phase: OutputState,
        direction: &[f64],
        alpha: Option<f64>,
    ) -> BuiltinResult<bool> {
        if let Some(kind) = evaluator.multiobjective_kind() {
            return super::multiobjective::observation::notify(
                kind, self, evaluator, parameters, phase, direction, alpha,
            )
            .await;
        }
        if evaluator.semi_infinite.is_some() {
            return super::semi_infinite::observe(
                self, evaluator, parameters, phase, direction, alpha,
            )
            .await;
        }
        // Shape is a small owned adapter fact. No evaluator borrow crosses the
        // observer's mutable original cooperation port.
        let shape = evaluator.shape.clone();
        let scalar = evaluator.scalar;
        if parameters.display == runmat_builtins::OptimDisplay::Iter
            || (phase == OutputState::Done
                && parameters.display == runmat_builtins::OptimDisplay::Final)
        {
            let _entry = evaluator
                .cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            crate::console::record_console_line(crate::console::ConsoleStream::Stdout,
                format!("fmincon {}: iteration={} funcCount={} fval={:?} constraintViolation={:?} firstOrderOptimality={:?}{}",
                    self.algorithm.name(),self.iterations,evaluator.budget.counts().points,self.point.as_ref().map(|point|point.f),
                    self.violation,self.optimality,if phase==OutputState::Done {self.termination.description()}else{""}))?;
        }
        observation::notify(
            parameters.callback,
            phase,
            Snapshot {
                x: &self.x,
                point: self.point.as_ref(),
                shape: &shape,
                scalar,
                iterations: self.iterations,
                counts: evaluator.budget.counts(),
                violation: self.violation,
                optimality: self.optimality,
                step: self.step,
                alpha,
                direction,
            },
            evaluator.cooperation,
        )
        .await
    }
}
