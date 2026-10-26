//! Original selected worker ports and shape-preserving complete-poll waves.
use super::*;
use crate::context::RuntimeContext;
use crate::execution::{CallbackBatch, CallbackParallelMode, RootedValueSequence};
use numeric::{Geometry, Numeric};
use runmat_value::Tensor;

mod execution;
pub(in crate::builtins::math::optim::patternsearch) use execution::Execution;
impl Evaluator<'_> {
    pub fn wave_width(&self, complete: bool) -> usize {
        self.execution().width(complete).min(self.remaining())
    }

    pub async fn evaluate_wave(&mut self, points: Vec<Vec<f64>>) -> BuiltinResult<Vec<Evaluation>> {
        if points.is_empty() || points.len() > self.remaining() {
            return Err(invalid("callback wave exceeds its admitted point budget"));
        }
        if self.execution().direct() {
            let mut results = buffer(points.len())?;
            for point in points {
                results.push(self.evaluate_scalar(point).await?);
            }
            return Ok(results);
        }
        let mut slots = buffer(points.len())?;
        let mut selected = buffer(points.len())?;
        for point in points {
            let cached = if let Some(cache) = self
                .state
                .cache
                .as_ref()
                .filter(|_| self.context.demand == super::ObservationDemand::MaySkipVisited)
            {
                cache.contains(&point, self.cooperation).await?
            } else {
                false
            };
            if point.iter().any(|value| !value.is_finite()) {
                slots.push(Err(Stop::Undefined));
            } else if cached {
                slots.push(Err(Stop::Cached));
            } else {
                selected.push((slots.len(), point));
                slots.push(Err(Stop::Undefined));
            }
        }
        if selected.is_empty() {
            return Ok(slots);
        }
        if self.execution().vectorized {
            self.vectorized_wave(selected, &mut slots).await?;
        } else if self.execution().objective.is_some() {
            self.worker_wave(selected, &mut slots).await?;
        } else {
            for (slot, point) in selected {
                slots[slot] = self.evaluate_scalar(point).await?;
            }
        }
        Ok(slots)
    }

    async fn worker_wave(
        &mut self,
        points: Vec<(usize, Vec<f64>)>,
        slots: &mut [Evaluation],
    ) -> BuiltinResult<()> {
        let mut arguments = buffer(points.len())?;
        let mut carriers = buffer(points.len())?;
        for (_, point) in &points {
            // Only our own Num/Tensor coordinate carrier is cloned here;
            // the same immutable backing serves the later nonlinear call.
            let argument = self.argument(point).await?;
            if self.context.demand == super::ObservationDemand::Fresh {
                if let Some(cache) = &mut self.state.cache {
                    cache.record(&argument)?;
                }
            }
            carriers.push(argument.clone());
            arguments.push(vec![argument]);
        }
        let start = self.state.objective_calls;
        self.state.objective_calls = self
            .state
            .objective_calls
            .checked_add(points.len())
            .ok_or_else(|| invalid("objective point count overflow"))?;
        if let Some(history) = &mut self.state.history {
            history.attempted(self.state.objective_calls)?;
        }
        self.state
            .callbacks
            .charge(CallbackRole::Objective, points.len())?;
        let returned = {
            let execution = self.context.execution.unwrap_or(&self.state.execution);
            let batch = execution
                .objective
                .as_ref()
                .expect("selected objective port");
            let (_, budget) = self.cooperation.runtime_budget();
            batch
                .execute(
                    arguments,
                    1,
                    budget
                        .ok_or_else(|| invalid("parallel route lacks its original CPU budget"))?,
                )?
                .await?
        };
        if returned.len() != points.len() {
            return Err(invalid("worker objective result extent changed"));
        }
        let mut measured = buffer(points.len())?;
        for (ordinal, (((slot, x), carrier), result)) in
            points.into_iter().zip(carriers).zip(returned).enumerate()
        {
            if let Some(f) = self.objective_result(result).await? {
                measured.push((slot, x, f, carrier, start + ordinal + 1));
            }
        }
        if self.execution().constraints.is_some() {
            let mut arguments = buffer(measured.len())?;
            for (_, _, _, carrier, _) in &measured {
                arguments.push(vec![carrier.clone()]);
            }
            if !arguments.is_empty() {
                self.state.constraint_calls = self
                    .state
                    .constraint_calls
                    .checked_add(arguments.len())
                    .ok_or_else(|| invalid("constraint callback count overflow"))?;
                self.state
                    .callbacks
                    .charge(CallbackRole::NonlinearConstraints, arguments.len())?;
                let returned = {
                    let execution = self.context.execution.unwrap_or(&self.state.execution);
                    let batch = execution
                        .constraints
                        .as_ref()
                        .expect("selected constraint port");
                    let (_, budget) = self.cooperation.runtime_budget();
                    batch
                        .execute(
                            arguments,
                            2,
                            budget.ok_or_else(|| {
                                invalid("parallel constraints lack original CPU budget")
                            })?,
                        )?
                        .await?
                };
                if returned.len() != measured.len() {
                    return Err(invalid("worker constraint result extent changed"));
                }
                for ((slot, x, f, carrier, ordinal), result) in measured.into_iter().zip(returned) {
                    if let Some((inequality, equality)) = self.constraint_result(result).await? {
                        let value = self
                            .complete_measurement(
                                &carrier,
                                ordinal,
                                x.len(),
                                f,
                                inequality,
                                equality,
                            )
                            .await?;
                        slots[slot] = Ok(Point {
                            x,
                            f: value.f,
                            inequality: value.inequality,
                            equality: value.equality,
                        });
                    }
                }
            }
        } else {
            for (slot, x, f, carrier, ordinal) in measured {
                let value = self
                    .complete_measurement(&carrier, ordinal, x.len(), f, Vec::new(), Vec::new())
                    .await?;
                slots[slot] = Ok(Point {
                    x,
                    f: value.f,
                    inequality: value.inequality,
                    equality: value.equality,
                });
            }
        }
        Ok(())
    }

    async fn vectorized_wave(
        &mut self,
        points: Vec<(usize, Vec<f64>)>,
        slots: &mut [Evaluation],
    ) -> BuiltinResult<()> {
        let width = points.len();
        let n = points[0].1.len();
        let point_axis = self.point_axis();
        let columns = point_axis == PointAxis::Columns;
        let mut values = buffer(
            width
                .checked_mul(n)
                .ok_or_else(|| invalid("vectorized argument extent overflow"))?,
        )?;
        if columns {
            for (_, point) in &points {
                values.extend_from_slice(point);
                self.cooperation.completed(n as u64).await?;
            }
        } else {
            for coordinate in 0..n {
                for (_, point) in &points {
                    values.push(point[coordinate]);
                    self.cooperation.completed(1).await?;
                }
            }
        }
        let shape = point_axis.argument_shape(n, width).to_vec();
        let backing = Tensor::new(values, shape).map_err(invalid)?;
        if let Some(cache) = &mut self.state.cache {
            for ordinal in 0..width {
                cache.record_matrix(&backing, ordinal, width, n, columns)?;
                self.cooperation.completed(1).await?;
            }
        }
        let retained = self.state.history.as_ref().map(|_| backing.clone());
        let argument = Value::Tensor(backing);
        let start = self.state.objective_calls;
        self.state.objective_calls = self
            .state
            .objective_calls
            .checked_add(width)
            .ok_or_else(|| invalid("objective point count overflow"))?;
        if let Some(history) = &mut self.state.history {
            history.attempted(self.state.objective_calls)?;
        }
        self.state.callbacks.charge(CallbackRole::Objective, 1)?;
        let returned = common::call_function_sequence_in_context(
            self.cooperation.runtime(),
            self.context.objective,
            [argument.clone()],
            1,
        )
        .await?;
        let mut returned = self.selected_values(returned, 1)?;
        let objective = self
            .matrix_result(
                returned.next().expect("selected objective"),
                width,
                Some(1),
                columns,
            )
            .await?;
        drop(returned);
        let Some(objective) = objective else {
            return Ok(());
        };
        let constraints = if let Some(callback) = self.context.constraints {
            self.state.constraint_calls = self
                .state
                .constraint_calls
                .checked_add(1)
                .ok_or_else(|| invalid("constraint point count overflow"))?;
            self.state
                .callbacks
                .charge(CallbackRole::NonlinearConstraints, 1)?;
            let returned = common::call_function_sequence_in_context(
                self.cooperation.runtime(),
                callback,
                [argument],
                2,
            )
            .await?;
            let returned = select(returned, 2, self.cooperation.runtime())?;
            // Both concrete roles, including their point axes and component
            // extents, are admitted before the first provider transfer.
            let mut real = true;
            for (index, value) in returned.as_slice().iter().enumerate() {
                real &= numeric::preflight(
                    value,
                    "vectorized nonlinear constraints",
                    Geometry::MatrixAny,
                    &self.context.policy,
                    self.cooperation.runtime(),
                )?;
                components(
                    &numeric::shape(value).expect("preflight carrier"),
                    width,
                    self.state
                        .extents
                        .map(|known| if index == 0 { known.0 } else { known.1 }),
                    columns,
                )?;
            }
            if !real {
                return Ok(());
            }
            let mut returned = self.rooted_values(returned)?;
            let inequality = self
                .matrix_result(
                    returned.next().expect("selected inequalities"),
                    width,
                    self.state.extents.map(|x| x.0),
                    columns,
                )
                .await?;
            let equality = self
                .matrix_result(
                    returned.next().expect("selected equalities"),
                    width,
                    self.state.extents.map(|x| x.1),
                    columns,
                )
                .await?;
            drop(returned);
            let (Some(inequality), Some(equality)) = (inequality, equality) else {
                return Ok(());
            };
            self.state.extents = Some((inequality.1, equality.1));
            Some((inequality, equality))
        } else {
            None
        };
        for (ordinal, (slot, x)) in points.into_iter().enumerate() {
            let f = objective.0[ordinal];
            let mut inequality = buffer(constraints.as_ref().map_or(0, |value| value.0 .1))?;
            let mut equality = buffer(constraints.as_ref().map_or(0, |value| value.1 .1))?;
            if let Some((inequalities, equalities)) = &constraints {
                for component in 0..inequalities.1 {
                    inequality.push(
                        inequalities.0[if columns {
                            ordinal * inequalities.1 + component
                        } else {
                            component * width + ordinal
                        }],
                    );
                }
                for component in 0..equalities.1 {
                    equality.push(
                        equalities.0[if columns {
                            ordinal * equalities.1 + component
                        } else {
                            component * width + ordinal
                        }],
                    );
                }
            }
            if f.is_finite() && inequality.iter().chain(&equality).all(|v| v.is_finite()) {
                let value = if let Some(backing) = &retained {
                    self.complete_batch_measurement(
                        backing,
                        point_axis,
                        ordinal,
                        width,
                        n,
                        start + ordinal + 1,
                        f,
                        inequality,
                        equality,
                    )
                    .await?
                } else {
                    Measurement {
                        f,
                        inequality: inequality.into(),
                        equality: equality.into(),
                    }
                };
                slots[slot] = Ok(Point {
                    x,
                    f: value.f,
                    inequality: value.inequality,
                    equality: value.equality,
                });
            }
            self.cooperation.completed(1).await?;
        }
        Ok(())
    }
    fn selected_values(
        &self,
        returned: RootedValueSequence,
        outputs: usize,
    ) -> BuiltinResult<crate::execution::RootedValuesIntoIter> {
        let returned = select(returned, outputs, self.cooperation.runtime())?;
        self.rooted_values(returned)
    }
    fn rooted_values(
        &self,
        returned: RootedValueSequence,
    ) -> BuiltinResult<crate::execution::RootedValuesIntoIter> {
        let _scope = self.cooperation.runtime().map(RuntimeContext::enter);
        returned.try_into_values().map_err(consumer_refusal)
    }
    async fn matrix_result(
        &mut self,
        value: Value,
        width: usize,
        expected: Option<usize>,
        columns: bool,
    ) -> BuiltinResult<Option<(Vec<f64>, usize)>> {
        let Some(value) = numeric::prepare(
            value,
            "vectorized result",
            Geometry::MatrixAny,
            &self.context.policy,
            self.cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        let Numeric::Real { shape, values } = numeric::materialize_raw(
            value,
            "vectorized result",
            &self.context.policy,
            self.cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        let components = components(&shape, width, expected, columns)?;
        Ok(Some((values, components)))
    }
}

fn components(
    shape: &[usize],
    width: usize,
    expected: Option<usize>,
    columns: bool,
) -> BuiltinResult<usize> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, &axis| count.checked_mul(axis))
        .ok_or_else(|| invalid("vectorized result extent overflow"))?;
    let components = if count == 0 {
        0
    } else {
        let rows = shape.first().copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(1);
        if (if columns { cols } else { rows }) != width {
            return Err(invalid(
                "vectorized result has wrong point orientation or extent",
            ));
        }
        if columns {
            rows
        } else {
            cols
        }
    };
    if expected.is_some_and(|known| known != components) {
        return Err(invalid("vectorized result component extent changed"));
    }
    Ok(components)
}

mod invoke;
