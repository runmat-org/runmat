//! Frozen-generation borrowed merit oracle for the shared SQP controller.
//! Model probes are distinct from expensive objective callbacks and counters.
use super::{
    domain::BoxView,
    invalid,
    merit::Criterion,
    rbf::{Model, Prediction, Training},
    IDENTITY, INVALID_ARGUMENT, NAME,
};
use crate::builtins::math::optim::{
    common::{global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation},
    fmincon::internal::{EvaluationStop, Oracle, Probe, ProbeRole},
};
use crate::BuiltinResult;
use futures::future::LocalBoxFuture;

pub(super) struct MeritOracle<'a, T, E> {
    model: &'a Model,
    training: &'a T,
    evaluated: &'a E,
    domain: BoxView<'a>,
    criterion: Criterion,
    inequality_start: usize,
    tolerance: f64,
    clock: &'a SolverTiming,
    time_limit: TimeLimit,
    coordinates: Vec<f64>,
    prediction: Vec<f64>,
}
pub(super) struct ModelView<'a, T, E> {
    pub model: &'a Model,
    pub training: &'a T,
    pub evaluated: &'a E,
    pub domain: BoxView<'a>,
    pub criterion: Criterion,
    pub inequality_start: usize,
}
pub(super) struct Controls<'a> {
    pub tolerance: f64,
    pub clock: &'a SolverTiming,
    pub time_limit: TimeLimit,
}
async fn workspace(count: usize, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = runmat_value::exact_capacity_vec(count).map_err(|source| {
        crate::build_runtime_error("surrogateopt: merit-oracle workspace allocation failed")
            .with_builtin(NAME)
            .with_source(source)
            .build()
    })?;
    while values.len() < count {
        let next = count.min(values.len().saturating_add(1024));
        let work = next - values.len();
        values.resize(next, 0.0);
        cooperation.completed(work as u64).await?;
    }
    Ok(values)
}
impl<'a, T: Training, E: Training> MeritOracle<'a, T, E> {
    pub(super) fn expired(&self) -> bool {
        self.time_limit.reached(self.clock.elapsed())
    }
    pub(super) async fn new(
        view: ModelView<'a, T, E>,
        controls: Controls<'a>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let ModelView {
            model,
            training,
            evaluated,
            domain,
            criterion,
            inequality_start,
        } = view;
        let Controls {
            tolerance,
            clock,
            time_limit,
        } = controls;
        let time_limit =
            time_limit.validate(crate::builtins::math::optim::common::global_time::Policy {
                identity: IDENTITY,
                error: &INVALID_ARGUMENT,
            })?;
        if !training.valid()
            || !evaluated.valid()
            || evaluated.rows().is_empty()
            || training.dimensions() != domain.dimensions()
            || evaluated.dimensions() != domain.dimensions()
            || inequality_start > training.lanes()
            || inequality_start > 1
            || matches!(criterion, Criterion::Objective(_)) && inequality_start != 1
            || !tolerance.is_finite()
            || tolerance < 0.0
        {
            return Err(invalid("invalid borrowed merit model geometry or schema"));
        }
        let dimensions = domain.dimensions();
        Ok(Self {
            model,
            training,
            evaluated,
            domain,
            criterion,
            inequality_start,
            tolerance,
            clock,
            time_limit,
            coordinates: workspace(dimensions, cooperation).await?,
            prediction: workspace(training.lanes(), cooperation).await?,
        })
    }

    async fn probe(
        &mut self,
        x: &[f64],
        role: ProbeRole,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::result::Result<Probe, EvaluationStop>> {
        cooperation.completed(0).await?;
        self.clock.checked_elapsed()?;
        if self.expired() {
            return Ok(Err(EvaluationStop::Time));
        }
        if x.len() != self.domain.variables() {
            return Err(invalid("merit probe changed the original variable extent"));
        }
        for (axis, slot) in self.coordinates.iter_mut().enumerate() {
            let column = self.domain.free_columns()[axis];
            *slot = self.domain.normalize_coordinate(x[column], axis);
            if !slot.is_finite() || !(0.0..=1.0).contains(slot) {
                return Ok(Err(EvaluationStop::Undefined));
            }
            cooperation.completed(1).await?;
        }
        let constraints_only = matches!(role, ProbeRole::ConstraintsOnly);
        let start = if constraints_only || matches!(self.criterion, Criterion::Feasibility) {
            self.inequality_start
        } else {
            0
        };
        if self
            .model
            .predict_lanes_into(
                self.training,
                &self.coordinates,
                start..self.prediction.len(),
                &mut self.prediction[start..],
                cooperation,
            )
            .await?
            == Prediction::Undefined
        {
            return Ok(Err(EvaluationStop::Undefined));
        }
        let mut inequality =
            runmat_value::exact_capacity_vec(self.prediction.len() - self.inequality_start)
                .map_err(|source| {
                    crate::build_runtime_error(
                        "surrogateopt: modeled constraint output allocation failed",
                    )
                    .with_builtin(NAME)
                    .with_source(source)
                    .build()
                })?;
        for &value in &self.prediction[self.inequality_start..] {
            let residual = value - self.tolerance;
            if !residual.is_finite() {
                return Ok(Err(EvaluationStop::Undefined));
            }
            inequality.push(residual);
            cooperation.completed(1).await?;
        }
        if constraints_only {
            return Ok(Ok(Probe::ConstraintsOnly {
                inequality,
                equality: Vec::new(),
            }));
        }
        let objective = match self.criterion {
            // This declared local problem is feasibility-only. Zero is its
            // mathematical constant objective, never a substitute user Fval.
            Criterion::Feasibility => 0.0,
            Criterion::Objective(scale) => {
                let mut nearest = f64::INFINITY;
                for row in 0..self.evaluated.rows().len() {
                    let mut distance = 0.0_f64;
                    for (axis, &coordinate) in self.coordinates.iter().enumerate() {
                        distance =
                            distance.hypot(coordinate - self.evaluated.coordinate(row, axis));
                        cooperation.completed(1).await?;
                    }
                    nearest = nearest.min(distance);
                }
                let Some(value) = scale.value(self.prediction[0], nearest) else {
                    return Ok(Err(EvaluationStop::Undefined));
                };
                value
            }
        };
        Ok(Ok(Probe::Full {
            objective,
            inequality,
            equality: Vec::new(),
        }))
    }
}
impl<T: Training, E: Training> Oracle for MeritOracle<'_, T, E> {
    fn evaluate<'a>(
        &'a mut self,
        x: &'a [f64],
        role: ProbeRole,
        cooperation: &'a mut ScalarCooperation,
    ) -> LocalBoxFuture<'a, BuiltinResult<std::result::Result<Probe, EvaluationStop>>> {
        Box::pin(self.probe(x, role, cooperation))
    }
}
