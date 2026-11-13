//! Genetic fitness is the enclosing current Pattern merit, evaluated through
//! its one physical callback state and effective original callback route.
use super::super::super::{
    augmented::Augmented,
    buffer,
    evaluation::{Context, Evaluator, Point, State},
    invalid,
    linear::Linear,
    timing::AuxiliaryTiming,
};
use super::super::merit::{Admission, Best, Transaction};
use super::Stop;
use crate::builtins::math::optim::{
    common::{
        global_time::TimeLimit,
        population::{Fitness, Measurement},
        ScalarCooperation,
    },
    ga::evaluation::source::{Request, Routes, Wave},
};
use crate::{execution::CallbackParallelMode, BuiltinResult};
use runmat_value::{NumericDType, Value};
use std::ops::ControlFlow;

pub(super) struct Oracle<'a> {
    pub context: Context<'a>,
    pub state: &'a mut State,
    pub execution: Option<super::super::super::evaluation::batch::Execution>,
    pub base: &'a Point,
    pub initial: f64,
    pub geometry: &'a Linear,
    pub tolerance: f64,
    pub augmented: Option<&'a Augmented>,
    pub timing: &'a AuxiliaryTiming<'a>,
    pub maximum: TimeLimit,
    pub best: Option<Best>,
    pub started: CountsAtEntry,
}
pub(super) struct CountsAtEntry {
    pub points: usize,
    pub constraints: usize,
    pub callbacks: super::super::super::evaluation::CallbackCounts,
}
impl Oracle<'_> {
    pub fn refresh_counts(
        &self,
        objective: &mut crate::builtins::math::optim::common::population::Counts,
        nonlinear: &mut crate::builtins::math::optim::common::population::Counts,
    ) -> BuiltinResult<()> {
        let difference = |now: usize, before: usize| {
            now.checked_sub(before).ok_or_else(|| {
                invalid("genetic search original observation counters moved backwards")
            })
        };
        objective.points = difference(self.state.objective_calls, self.started.points)?;
        objective.callbacks = difference(
            self.state.callbacks.objective,
            self.started.callbacks.objective,
        )?;
        nonlinear.points = difference(self.state.constraint_calls, self.started.constraints)?;
        nonlinear.callbacks = difference(
            self.state.callbacks.nonlinear,
            self.started.callbacks.nonlinear,
        )?;
        Ok(())
    }
    pub async fn seed_initial<
        P: crate::builtins::math::optim::common::population::projection::Arguments,
    >(
        &mut self,
        population: &P,
        supplied: &mut [Option<crate::builtins::math::optim::ga::evaluation::fitness::Supplied>],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        use crate::builtins::math::optim::ga::evaluation::fitness::{Origin, Supplied};
        // The receipt is the retained physical Point and its fixed current
        // merit, not a score guessed from a matching numeric coordinate.
        let Some(numeric) = population.known_coordinates() else {
            return Ok(());
        };
        if numeric.variables != self.base.x.len() || supplied.len() != numeric.members {
            return Err(invalid("initial genetic observation dimension changed"));
        }
        // Preserve the ordinary duplicate-alias rule: an explicit score on
        // any identical seed row may supply its missing representative. Do
        // not pre-empt that authority with the retained physical observation.
        for (row, score) in supplied.iter().enumerate() {
            if score.is_some() && same_seed(numeric, row, &self.base.x, cooperation).await? {
                return Ok(());
            }
        }
        for (row, score) in supplied.iter_mut().enumerate() {
            if score.is_none() && same_seed(numeric, row, &self.base.x, cooperation).await? {
                *score = Some(Supplied {
                    value: Measurement::Real(self.initial),
                    origin: Origin::Measured,
                });
            }
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    pub fn select_routes(
        &mut self,
        _projected_merit: &Value,
        nonlinear: Option<&Value>,
        vectorized: bool,
        parallel: CallbackParallelMode,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Routes> {
        if nonlinear.is_some() {
            return Err(invalid(
                "genetic search uses the current Pattern merit, not a second nonlinear problem",
            ));
        }
        self.execution = Some(
            super::super::super::evaluation::batch::Execution::select_with(
                vectorized,
                parallel,
                self.context.objective,
                self.context.constraints,
                cooperation.runtime(),
            )?,
        );
        // The original callbacks, not the local scoped merit port, own these
        // real worker/vectorized waves. They are dispatched in next below.
        Ok(Routes {
            objective: None,
            nonlinear: None,
        })
    }
    pub fn end<P: crate::builtins::math::optim::common::population::projection::Arguments>(
        &self,
        request: &Request<'_, P>,
    ) -> BuiltinResult<ControlFlow<Stop, usize>> {
        if request.outputs != 1 || request.start >= request.selected.len() {
            return Err(invalid("invalid genetic merit wave request"));
        }
        if let Some(stop) = self.timing.stop(self.maximum) {
            return Ok(ControlFlow::Break(stop.into()));
        }
        let remaining = self.state.limit.saturating_sub(self.state.objective_calls);
        if remaining == 0 {
            return Ok(ControlFlow::Break(Stop::Budget));
        }
        let execution = self
            .execution
            .as_ref()
            .ok_or_else(|| invalid("genetic merit route was not selected"))?;
        if execution.vectorized != request.vectorized || request.parallel.is_some() {
            return Err(invalid(
                "genetic merit route differs from its selected original callback owner",
            ));
        }
        let width = execution
            .width(true)
            .min(remaining)
            .min(request.selected.len() - request.start);
        if width == 0 {
            return Err(invalid("genetic merit route has zero width"));
        }
        let end = request.start + width;
        Ok(ControlFlow::Continue(end))
    }
    pub async fn next<
        P: crate::builtins::math::optim::common::population::projection::Arguments,
    >(
        &mut self,
        request: Request<'_, P>,
        end: usize,
        projected: Option<crate::builtins::math::optim::common::population::projection::Argument>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<Stop, Wave>> {
        cooperation.completed(0).await?;
        let current_end = match self.end(&request)? {
            ControlFlow::Continue(end) => end,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        if current_end < end {
            return Ok(ControlFlow::Break(Stop::Budget));
        }
        if current_end != end {
            return Err(invalid("prepared genetic projection extent changed"));
        }
        let execution = self
            .execution
            .as_ref()
            .ok_or_else(|| invalid("genetic merit route was not selected"))?;
        let width = end - request.start;
        let rows = &request.selected[request.start..end];
        let numeric = request.population.numeric_rows();
        let (value, roots) = match (numeric.is_none(), projected) {
            (true, Some(argument)) => {
                let (value, roots) = argument.into_parts();
                (Some(value), roots)
            }
            (false, None) => (None, None),
            _ => return Err(invalid("genetic prepared projection role changed")),
        };
        let n = self.base.x.len();
        let mut transaction = Transaction::new(width)?;
        let admission = Admission {
            base: self.base,
            initial: self.initial,
            geometry: self.geometry,
            tolerance: self.tolerance,
            reuse: super::super::merit::Reuse::RequestedMeasurement,
        };
        let value = match value {
            Some(value) => {
                Some(super::projection::coordinates(value, width, n, cooperation).await?)
            }
            None => None,
        };
        let before_points = self.state.objective_calls;
        let before_calls = self.state.callbacks.objective;
        let mut evaluator = Evaluator {
            context: Context {
                execution: Some(execution),
                demand: super::super::super::evaluation::ObservationDemand::Fresh,
                ..self.context
            },
            state: &mut *self.state,
            cooperation,
        };
        if let Some(numeric) = numeric {
            if numeric.variables != n || rows.iter().any(|&row| row >= numeric.members) {
                return Err(invalid(
                    "genetic numeric row view has changed coordinate geometry",
                ));
            }
            for &row in rows {
                let mut point = buffer(n)?;
                for column in 0..n {
                    point.push(numeric.get(row, column));
                    evaluator.cooperation.completed(1).await?;
                }
                transaction.push(point, &admission, &mut evaluator).await?;
            }
        } else {
            match value.expect("custom projection retained its actual value") {
                Value::Tensor(value)
                    if value.numeric_dtype() == NumericDType::F64
                        && (value.shape() == [width, n]
                            || width == 1
                                && value.len() == n
                                && value.shape().iter().filter(|&&axis| axis != 1).count()
                                    <= 1) =>
                {
                    let values = value.as_f64_slice().expect("double coordinate backing");
                    for row in 0..width {
                        let mut point = buffer(n)?;
                        for column in 0..n {
                            point.push(values[row + width * column]);
                            evaluator.cooperation.completed(1).await?;
                        }
                        transaction.push(point, &admission, &mut evaluator).await?;
                    }
                }
                Value::Num(value) if width == 1 && n == 1 => {
                    let mut point = buffer(1)?;
                    point.push(value);
                    transaction.push(point, &admission, &mut evaluator).await?;
                }
                _ => {
                    return Err(invalid(
                        "genetic search genomes must project to double coordinate rows",
                    ))
                }
            }
        }
        let augmented = self.augmented;
        let timing = self.timing;
        let maximum = self.maximum;
        let returned = transaction
            .finish(
                &mut self.best,
                self.initial,
                &mut async |point: &Point, cooperation: &mut ScalarCooperation| match augmented {
                    Some(augmented) => augmented.merit(point, cooperation).await,
                    None => Ok(Some(point.f)),
                },
                &mut evaluator,
                || timing.stop(maximum).map(Stop::from),
                Stop::Budget,
            )
            .await?;
        request.counts.points = request
            .counts
            .points
            .checked_add(evaluator.state.objective_calls - before_points)
            .ok_or_else(|| invalid("genetic search point count overflow"))?;
        request.counts.callbacks = request
            .counts
            .callbacks
            .checked_add(evaluator.state.callbacks.objective - before_calls)
            .ok_or_else(|| invalid("genetic search callback count overflow"))?;
        drop(roots);
        let scores = match returned {
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            ControlFlow::Continue(scores) => scores,
        };
        let mut measured = request
            .policy
            .capacity(width, evaluator.cooperation.runtime())?;
        for score in scores {
            measured.push(Measurement::Real(score));
            evaluator.cooperation.completed(1).await?;
        }
        let values =
            Fitness::from_measurements(request.policy, measured, evaluator.cooperation).await?;
        Ok(ControlFlow::Continue(Wave::Measured {
            start: request.start,
            end,
            values,
        }))
    }
}

async fn same_seed(
    numeric: crate::builtins::math::optim::common::population::projection::Coordinates<'_>,
    row: usize,
    seed: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let mut same = true;
    for (column, &value) in seed.iter().enumerate() {
        same &= numeric.get(row, column).to_bits() == value.to_bits();
        cooperation.completed(1).await?;
    }
    Ok(same)
}
