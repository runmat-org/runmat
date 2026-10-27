//! The original invocation owns callback dispatch, numerical interpretation and
//! evaluated-point accounting. Numeric undefinedness is not an exception filter.
mod attainment;
mod attainment_merit;
mod budget;
pub(in crate::builtins::math::optim) mod internal;
mod jacobian;
mod objective;
pub(in crate::builtins::math::optim::fmincon) use attainment::{
    Attainment, Binding, ObjectiveValues,
};

pub(super) use super::numeric::Policy;
use super::numeric::{self, Geometry, Numeric};
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::BuiltinResult;
pub(super) use budget::{Budget, Counts};
use clarabel::algebra::CscMatrix;
use jacobian::Jacobian;
use runmat_value::Value;

pub(super) struct Point {
    pub semi_infinite: Option<super::semi_infinite::Payload>,
    pub objectives: Option<ObjectiveValues>,
    pub f: f64,
    pub inequality: Vec<f64>,
    pub equality: Vec<f64>,
    pub gradient: Option<Vec<f64>>,
    pub inequality_jacobian: Option<CscMatrix<f64>>,
    pub equality_jacobian: Option<CscMatrix<f64>>,
    pub projected_objective: bool,
    pub projected_constraints: bool,
}
impl Point {
    pub fn full_derivatives(&self) -> bool {
        !self.projected_objective && !self.projected_constraints
    }
}

pub(super) fn select_sequence(
    returned: crate::execution::RootedValueSequence,
    width: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    returned
        .resolve_sequence(
            runmat_types::SequenceUse::from_requested_outputs(
                runmat_types::RequestedOutputCount::Exactly(width),
            ),
            crate::sequence::SequenceResolutionContext::default(),
        )
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
fn selected_values(
    returned: crate::execution::RootedValueSequence,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::execution::RootedValuesIntoIter> {
    let _entry = runtime.map(crate::context::RuntimeContext::enter);
    returned.try_into_values().map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

pub(super) type Evaluation = std::result::Result<Point, EvaluationStop>;
pub(in crate::builtins::math::optim) enum EvaluationStop {
    Undefined,
    BudgetExhausted,
    Time,
}

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum ProbeRole {
    Full,
    ConstraintsOnly,
}
pub(in crate::builtins::math::optim) enum Probe {
    Full {
        objective: f64,
        inequality: Vec<f64>,
        equality: Vec<f64>,
    },
    ConstraintsOnly {
        inequality: Vec<f64>,
        equality: Vec<f64>,
    },
}
impl Probe {
    pub fn actual_objective(&self) -> BuiltinResult<f64> {
        match self {
            Self::Full { objective, .. } => Ok(*objective),
            Self::ConstraintsOnly { .. } => Err(super::invalid(
                "constraint-only probe has no objective measurement",
            )),
        }
    }
    pub fn inequality(&self) -> &[f64] {
        match self {
            Self::Full { inequality, .. } => inequality,
            Self::ConstraintsOnly { inequality, .. } => inequality,
        }
    }
    pub fn equality(&self) -> &[f64] {
        match self {
            Self::Full { equality, .. } => equality,
            Self::ConstraintsOnly { equality, .. } => equality,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Objective<'a> {
    User(&'a Value),
    ConstantFeasibility,
    Internal,
}
impl Objective<'_> {
    fn user(&self) -> BuiltinResult<&Value> {
        match self {
            Self::User(value) => Ok(value),
            Self::ConstantFeasibility => Err(super::invalid(
                "constant feasibility objective has no user callable",
            )),
            Self::Internal => Err(super::invalid(
                "internal modeled objective has no user callable",
            )),
        }
    }
}
pub(super) struct Evaluator<'a> {
    pub feasibility_limit: Option<&'a super::feasibility::Limit>,
    pub internal: Option<&'a mut dyn internal::Oracle>,
    pub semi_infinite: Option<super::semi_infinite::Domain>,
    pub attainment: Option<Attainment>,
    pub scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
    pub cooperation: &'a mut ScalarCooperation,
    pub objective: Objective<'a>,
    pub constraints: Option<&'a Value>,
    pub selected_constraints: Option<&'a crate::call::descriptor::RepeatableCallable>,
    pub policy: Policy,
    pub shape: Vec<usize>,
    pub scalar: bool,
    pub objective_gradient: bool,
    pub constraint_gradient: bool,
    pub budget: budget::Budget,
    // Set by the first defined evaluation; a later extent change is an invalid
    // callable contract, not a numerical trial to shrink past.
    pub constraint_extents: Option<(usize, usize)>,
}

impl Evaluator<'_> {
    pub fn time_limit_reached(&self) -> bool {
        self.feasibility_limit.is_some_and(|limit| limit.expired())
    }
    /// One original objective generation, with the selected real third result
    /// rooted before its original tuple lease retires. No extra model call.
    pub async fn evaluate_reflective(
        &mut self,
        x: &[f64],
        third: bool,
    ) -> BuiltinResult<
        std::result::Result<(Point, Option<crate::execution::RootedValue>), EvaluationStop>,
    > {
        if self.attainment.is_some()
            || self.internal.is_some()
            || self.semi_infinite.is_some()
            || self.constraints.is_some()
            || !self.objective_gradient
        {
            return Err(super::invalid("reflective objective evaluation requires a scalar gradient-only objective without nonlinear constraints"));
        }
        if !self.budget.begin_point() {
            return Ok(Err(EvaluationStop::BudgetExhausted));
        }
        self.cooperation.completed(x.len() as u64).await?;
        let argument =
            common::vector_to_value(self.policy.owner.name, x.to_vec(), &self.shape, self.scalar)?;
        let width = if third { 3 } else { 2 };
        if self.time_limit_reached() {
            return Ok(Err(EvaluationStop::Time));
        }
        let returned = common::call_function_sequence_in_context(
            self.cooperation.runtime(),
            self.objective.user()?,
            [argument],
            width,
        )
        .await?;
        self.budget.completed_objective();
        let returned = select_sequence(returned, width, self.cooperation.runtime())?;
        let Some(objective::Decoded::Scalar {
            value,
            gradient,
            model,
        }) = objective::read(
            returned,
            x.len(),
            if third {
                objective::Shape::ScalarWithModel
            } else {
                objective::Shape::Scalar
            },
            true,
            &self.policy,
            self.cooperation,
        )
        .await?
        else {
            return Ok(Err(EvaluationStop::Undefined));
        };
        Ok(Ok((
            Point {
                semi_infinite: None,
                objectives: None,
                f: value,
                gradient,
                inequality: Vec::new(),
                equality: Vec::new(),
                inequality_jacobian: Some(CscMatrix::zeros((0, x.len()))),
                equality_jacobian: Some(CscMatrix::zeros((0, x.len()))),
                projected_objective: false,
                projected_constraints: false,
            },
            model,
        )))
    }
    pub async fn evaluate_probe(
        &mut self,
        x: &[f64],
        role: ProbeRole,
    ) -> BuiltinResult<std::result::Result<Probe, EvaluationStop>> {
        if self.time_limit_reached() {
            return Ok(Err(EvaluationStop::Time));
        }
        if self.internal.is_some() {
            return self.evaluate_internal(x, role).await;
        }
        if matches!(role, ProbeRole::ConstraintsOnly) && self.semi_infinite.is_some() {
            return super::semi_infinite::probe(self, x).await;
        }
        self.evaluate(x).await.map(|result| {
            result.map(|point| Probe::Full {
                objective: point.f,
                inequality: point.inequality,
                equality: point.equality,
            })
        })
    }
    pub async fn evaluate(&mut self, x: &[f64]) -> BuiltinResult<Evaluation> {
        if self.time_limit_reached() {
            return Ok(Err(EvaluationStop::Time));
        }
        if self.internal.is_some() {
            return self.internal_point(x).await;
        }
        if matches!(self.objective, Objective::Internal) {
            return Err(super::invalid(
                "internal objective lost its original numerical oracle",
            ));
        }
        let callback_x = self.callback_point(x)?;
        let callback_variables = callback_x.len();
        if !self.budget.begin_point() {
            return Ok(Err(EvaluationStop::BudgetExhausted));
        }
        self.cooperation
            .completed(callback_variables as u64)
            .await?;
        let policy = &self.policy;
        let objective = match self.objective {
            Objective::User(target) => {
                let argument = common::vector_to_value(
                    self.policy.owner.name,
                    callback_x.to_vec(),
                    &self.shape,
                    self.scalar,
                )?;
                let width = if self.objective_gradient { 2 } else { 1 };
                if self.time_limit_reached() {
                    return Ok(Err(EvaluationStop::Time));
                }
                let returned = common::call_function_sequence_in_context(
                    self.cooperation.runtime(),
                    target,
                    [argument],
                    width,
                )
                .await?;
                self.budget.completed_objective();
                let returned = select_sequence(returned, width, self.cooperation.runtime())?;
                let shape = if self.attainment.is_some() {
                    objective::Shape::Vector
                } else {
                    objective::Shape::Scalar
                };
                if self.semi_infinite.is_some() {
                    for value in returned.as_slice().iter() {
                        super::semi_infinite::admit_double(value)?;
                    }
                }
                let Some(objective) = objective::read(
                    returned,
                    callback_variables,
                    shape,
                    self.objective_gradient,
                    policy,
                    self.cooperation,
                )
                .await?
                else {
                    if self
                        .semi_infinite
                        .as_ref()
                        .is_some_and(|domain| domain.controls.check_values)
                    {
                        return Err(numeric::invalid(
                            policy,
                            "objective value",
                            "FunValCheck rejected an undefined value",
                        ));
                    }
                    return Ok(Err(EvaluationStop::Undefined));
                };
                objective
            }
            Objective::Internal => {
                unreachable!("internal oracle was selected before callback dispatch")
            }
            Objective::ConstantFeasibility => {
                let mut gradient = super::workspace::capacity(callback_variables, policy.owner)?;
                for _ in 0..callback_variables {
                    gradient.push(0.);
                    self.cooperation.completed(1).await?;
                }
                objective::Decoded::Scalar {
                    value: 0.,
                    gradient: Some(gradient),
                    model: None,
                }
            }
        };
        if self.semi_infinite.is_some() {
            if self.attainment.is_some() || self.constraint_gradient {
                return Err(super::invalid("semi-infinite evaluation requires scalar objective and numerical constraint derivatives"));
            }
            let objective::Decoded::Scalar {
                value,
                gradient,
                model: _,
            } = objective
            else {
                return Err(super::invalid("semi-infinite objective must be scalar"));
            };
            let point = Point {
                semi_infinite: None,
                objectives: None,
                f: value,
                gradient,
                inequality: Vec::new(),
                equality: Vec::new(),
                inequality_jacobian: None,
                equality_jacobian: None,
                projected_objective: false,
                projected_constraints: false,
            };
            return super::semi_infinite::point(self, x, point).await;
        }
        let Some(callback) = self.constraints else {
            return self
                .finish_objective(
                    x,
                    objective,
                    Point {
                        semi_infinite: None,
                        objectives: None,
                        f: 0.0,
                        inequality: Vec::new(),
                        equality: Vec::new(),
                        gradient: None,
                        inequality_jacobian: Some(CscMatrix::zeros((0, callback_variables))),
                        equality_jacobian: Some(CscMatrix::zeros((0, callback_variables))),
                        projected_objective: false,
                        projected_constraints: false,
                    },
                )
                .await;
        };
        let width = if self.constraint_gradient { 4 } else { 2 };
        let argument = common::vector_to_value(
            self.policy.owner.name,
            callback_x.to_vec(),
            &self.shape,
            self.scalar,
        )?;
        if self.time_limit_reached() {
            return Ok(Err(EvaluationStop::Time));
        }
        let callback = self.selected_constraints.map_or(
            common::callback_source::Callable::Dynamic(callback),
            common::callback_source::Callable::Selected,
        );
        let returned = callback
            .invoke(self.cooperation.runtime(), [argument], width)
            .await?;
        self.budget.completed_constraints();
        let cooperation = &mut *self.cooperation;
        let returned = select_sequence(returned, width, cooperation.runtime())?;
        let mut extents = [0usize; 2];
        for (index, value) in returned.as_slice().iter().take(2).enumerate() {
            if let Some(shape) = numeric::shape(value) {
                extents[index] = shape
                    .iter()
                    .try_fold(1usize, |count, &axis| count.checked_mul(axis))
                    .ok_or_else(|| {
                        numeric::invalid(policy, "nonlinear constraint", "extent overflow")
                    })?;
            }
        }
        if self
            .constraint_extents
            .is_some_and(|expected| expected != (extents[0], extents[1]))
        {
            return Err(numeric::invalid(
                policy,
                "nonlinear constraint",
                "changed its output extent",
            ));
        }
        for (index, value) in returned.as_slice().iter().enumerate() {
            let geometry = if index < 2 {
                Geometry::Array
            } else {
                Geometry::Matrix(callback_variables, extents[index - 2])
            };
            if !numeric::preflight(
                value,
                "nonlinear callback output",
                geometry,
                policy,
                cooperation.runtime(),
            )? {
                return Ok(Err(EvaluationStop::Undefined));
            }
            if index < 2 {
                extents[index] = numeric::shape(value)
                    .expect("concrete numeric carrier")
                    .iter()
                    .try_fold(1usize, |count, &axis| count.checked_mul(axis))
                    .ok_or_else(|| {
                        numeric::invalid(policy, "nonlinear constraint", "extent overflow")
                    })?;
            }
        }
        let mut values = selected_values(returned, cooperation.runtime())?;
        let constraints = async {
            let Numeric::Real {
                values: inequality, ..
            } = numeric::read(
                values.next().expect("validated nonlinear inequality"),
                "nonlinear inequality",
                Geometry::Array,
                policy,
                cooperation,
            )
            .await?
            else {
                return Ok::<_, crate::RuntimeError>(None);
            };
            let Numeric::Real {
                values: equality, ..
            } = numeric::read(
                values.next().expect("validated nonlinear equality"),
                "nonlinear equality",
                Geometry::Array,
                policy,
                cooperation,
            )
            .await?
            else {
                return Ok(None);
            };
            let (inequality_jacobian, equality_jacobian) = if width == 4 {
                let Jacobian::Real(inequality_jacobian) = jacobian::read(
                    values.next().expect("validated inequality Jacobian"),
                    callback_variables,
                    inequality.len(),
                    "nonlinear inequality Jacobian",
                    policy,
                    cooperation,
                )
                .await?
                else {
                    return Ok(None);
                };
                let Jacobian::Real(equality_jacobian) = jacobian::read(
                    values.next().expect("validated equality Jacobian"),
                    callback_variables,
                    equality.len(),
                    "nonlinear equality Jacobian",
                    policy,
                    cooperation,
                )
                .await?
                else {
                    return Ok(None);
                };
                (Some(inequality_jacobian), Some(equality_jacobian))
            } else {
                (None, None)
            };
            Ok(Some((
                inequality,
                equality,
                inequality_jacobian,
                equality_jacobian,
            )))
        }
        .await?;
        drop(values);
        let Some((inequality, equality, inequality_jacobian, equality_jacobian)) = constraints
        else {
            return Ok(Err(EvaluationStop::Undefined));
        };
        let extents = (inequality.len(), equality.len());
        if self
            .constraint_extents
            .is_some_and(|expected| expected != extents)
        {
            return Err(numeric::invalid(
                policy,
                "nonlinear constraint",
                "changed its output extent",
            ));
        }
        self.constraint_extents = Some(extents);
        self.finish_objective(
            x,
            objective,
            Point {
                semi_infinite: None,
                objectives: None,
                f: 0.0,
                inequality,
                equality,
                gradient: None,
                inequality_jacobian,
                equality_jacobian,
                projected_objective: false,
                projected_constraints: false,
            },
        )
        .await
    }
}
