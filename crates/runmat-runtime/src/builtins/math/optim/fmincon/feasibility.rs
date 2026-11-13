//! Internal nonlinear feasibility uses the existing constrained mechanisms.
//! The constant objective is mathematical model data, never a user observation.
use super::super::common::{
    global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation,
};
use super::{evaluation, model, numeric, options, outcome};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use futures::TryFutureExt;
use runmat_builtins::OptimAlgorithm;
use runmat_value::Value;

enum Algorithm {
    Sqp,
    WorkingSet,
    InteriorPoint,
}

fn original_runtime(
    cooperation: &ScalarCooperation,
) -> BuiltinResult<crate::context::RuntimeContext> {
    cooperation.runtime().cloned().ok_or_else(|| {
        let mut error = crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::Builtin,
            "nonlinear feasibility requires its selected execution context",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

pub(in crate::builtins::math::optim) struct Limit {
    pub timing: SolverTiming,
    pub maximum: TimeLimit,
}
impl Limit {
    pub fn expired(&self) -> bool {
        self.maximum.reached(self.timing.elapsed())
    }
}
/// These same buffers return to the caller when creation ends. The facade
/// does not retain a second original matrix or clone geometry per start.
pub(in crate::builtins::math::optim) struct Input {
    pub inequality: (CscMatrix<f64>, Vec<f64>),
    pub equality: (CscMatrix<f64>, Vec<f64>),
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
pub(in crate::builtins::math::optim) struct Prepared(options::Controls);
impl Prepared {
    pub fn value(&self) -> Value {
        self.0.prepared_value()
    }
}
pub(in crate::builtins::math::optim) async fn prepare_options(
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    let builder = match value {
        Value::Struct(value) => runmat_value::UnboundedStructValueBuilder::from_structure(value),
        Value::Tensor(value) if value.is_empty() => {
            runmat_value::UnboundedStructValueBuilder::new()
        }
        _ => {
            return Err(super::invalid(
                "SolverOpts must be a scalar fmincon options struct or []",
            ))
        }
    };
    let original = builder.finish();
    let fields = super::super::common::ScalarOptionFields::from_struct(Some(&original));
    let default_algorithm = fields
        .exact(runmat_builtins::OptimOptionField::Algorithm)
        .is_none();
    let default_display = fields
        .exact(runmat_builtins::OptimOptionField::Display)
        .is_none();
    let mut builder = runmat_value::UnboundedStructValueBuilder::from_structure(original);
    if default_algorithm {
        builder.insert("Algorithm", Value::from("sqp"));
    }
    if default_display {
        builder.insert("Display", Value::from("off"));
    }
    let controls =
        options::Controls::parse(Value::Struct(builder.finish()), true, cooperation).await?;
    if controls.algorithm == OptimAlgorithm::TrustRegionReflective {
        return Err(super::invalid(
            "nonlinear feasible creation cannot use trust-region-reflective",
        ));
    }
    Ok(Prepared(controls))
}
pub(in crate::builtins::math::optim) struct Domain {
    geometry: model::Geometry,
    controls: options::Controls,
    limit: Limit,
    extents: Option<(usize, usize)>,
}
pub(in crate::builtins::math::optim) struct Candidate {
    pub x: Vec<f64>,
    pub violation: Option<f64>,
    pub constraint_calls: usize,
    pub points: usize,
    pub timed_out: bool,
    pub extents: Option<(usize, usize)>,
}
impl Domain {
    pub fn new(input: Input, options: Prepared, limit: Limit) -> BuiltinResult<Self> {
        let Input {
            inequality: (a, b),
            equality: (aeq, beq),
            lower,
            upper,
        } = input;
        let n = lower.len();
        if n == 0
            || upper.len() != n
            || a.n != n
            || aeq.n != n
            || a.m != b.len()
            || aeq.m != beq.len()
        {
            return Err(super::invalid("invalid transferred feasibility geometry"));
        }
        Ok(Self {
            geometry: model::Geometry {
                variables: n,
                inequality: model::LinearConstraints { matrix: a, rhs: b },
                equality: model::LinearConstraints {
                    matrix: aeq,
                    rhs: beq,
                },
                lower,
                upper,
            },
            controls: options.0,
            limit,
            extents: None,
        })
    }
    pub fn run<'a>(
        &'a mut self,
        x: Vec<f64>,
        constraints: &'a Value,
        selected: &'a crate::call::descriptor::RepeatableCallable,
        policy: numeric::Policy,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<Candidate>> + 'a {
        futures::future::lazy(move |_| {
            let runtime = original_runtime(cooperation)?;
            // The same selected continuation owns the actual result. No
            // async forwarding/out-slot frame remains during user callbacks.
            crate::execution::admit_native_hostcall_with_contextual_error(&runtime, move || {
                self.run_inner(x, constraints, selected, policy, cooperation)
            })
        })
        .try_flatten()
    }
    async fn run_inner(
        &mut self,
        x: Vec<f64>,
        constraints: &Value,
        selected: &crate::call::descriptor::RepeatableCallable,
        policy: numeric::Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Candidate> {
        if x.len() != self.geometry.variables {
            return Err(super::invalid("feasibility point extent changed"));
        }
        let limit = match self.controls.point_limit {
            Some(limit) => limit,
            None => x
                .len()
                .checked_mul(100)
                .ok_or_else(|| super::invalid("default feasibility point limit overflow"))?,
        };
        let parameters = outcome::Parameters {
            function: None,
            max_iterations: self.controls.max_iterations,
            feasibility: self.controls.feasibility,
            optimality: self.controls.optimality,
            step: self.controls.step,
            differences: self.controls.differences,
            callback: self.controls.callback(),
            display: self.controls.display,
        };
        let mut evaluator = evaluation::Evaluator {
            internal: None,
            attainment: None,
            semi_infinite: None,
            scaling: self.controls.scaling.clone(),
            cooperation,
            objective: evaluation::Objective::ConstantFeasibility,
            feasibility_limit: Some(&self.limit),
            constraints: Some(constraints),
            selected_constraints: Some(selected),
            policy,
            shape: vec![1, x.len()],
            scalar: false,
            objective_gradient: true,
            constraint_gradient: self.controls.constraint_gradient,
            budget: evaluation::Budget::new(limit),
            constraint_extents: self.extents,
        };
        self.controls
            .scaling
            .validate_variables(policy.owner.name, x.len())?;
        let algorithm = match self.controls.algorithm {
            OptimAlgorithm::Sqp => Algorithm::Sqp,
            OptimAlgorithm::ActiveSet => Algorithm::WorkingSet,
            OptimAlgorithm::InteriorPoint => Algorithm::InteriorPoint,
            _ => return Err(super::invalid("unsupported nonlinear creation algorithm")),
        };
        let runtime = original_runtime(evaluator.cooperation)?;
        let geometry = &self.geometry;
        let original_parameters = &parameters;
        let original_evaluator = &mut evaluator;
        let selected_runtime = &runtime;
        let result =
            crate::execution::admit_native_hostcall_with_contextual_error(&runtime, move || {
                use futures::future::Either;
                match algorithm {
                    Algorithm::Sqp => Either::Left(super::sqp::minimize_staged(
                        selected_runtime,
                        x,
                        geometry,
                        original_parameters,
                        original_evaluator,
                        super::sqp::Strategy::Conic,
                    )),
                    Algorithm::WorkingSet => {
                        Either::Right(Either::Left(super::sqp::minimize_staged(
                            selected_runtime,
                            x,
                            geometry,
                            original_parameters,
                            original_evaluator,
                            super::sqp::Strategy::WorkingSet,
                        )))
                    }
                    Algorithm::InteriorPoint => {
                        Either::Right(Either::Right(super::interior_point::minimize(
                            x,
                            geometry,
                            original_parameters,
                            original_evaluator,
                        )))
                    }
                }
            })?
            .await?;
        debug_assert_eq!(result.counts.objective_calls, 0);
        self.extents = evaluator.constraint_extents;
        let timed_out = self.limit.expired() || result.termination == outcome::Termination::Time;
        let violation = if let Some(point) = result.point.as_ref() {
            self.geometry
                .violation(&result.x, point, policy.owner, evaluator.cooperation)
                .await?
                .map(|(maximum, _)| maximum)
        } else {
            None
        };
        drop(result.point);
        Ok(Candidate {
            x: result.x,
            violation,
            constraint_calls: result.counts.constraint_calls,
            points: result.counts.points,
            timed_out,
            extents: self.extents,
        })
    }
    pub fn into_input(self) -> Input {
        let model::Geometry {
            inequality,
            equality,
            lower,
            upper,
            ..
        } = self.geometry;
        Input {
            inequality: (inequality.matrix, inequality.rhs),
            equality: (equality.matrix, equality.rhs),
            lower,
            upper,
        }
    }
}
