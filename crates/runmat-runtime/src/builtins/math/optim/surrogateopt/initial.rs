//! Initial rows keep their original numeric headers and optional measured
//! lanes. Missing schema waits for a real new evaluation, not a guessed
//! no-constraint result or a replay solely to discover the return shape.
use super::{
    evaluation::{Completed, Seed},
    history::Point,
    invalid,
    numeric::{self, DoubleArray, Geometry},
    observation::{self, Form, Known, ObjectiveKind, Observation, Schema},
    proposal::{self, Prepared},
    public_values,
};
use crate::{
    builtins::math::optim::{common::ScalarCooperation, intlinprog::feasibility::Domain},
    execution::{RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::Value;
use std::{num::NonZeroUsize, rc::Rc};
mod archive;
pub(super) use archive::{ArchivedInitial, BorrowedInitial};

enum Measurements {
    Missing,
    Complete(Vec<Observation>),
    Partial {
        supplied: Known,
        values: Vec<Observation>,
    },
}
pub(super) struct InitialData {
    points: DoubleArray,
    measurements: Measurements,
    rows: usize,
    variables: usize,
}
#[derive(Clone)]
pub(super) struct Initial {
    data: Rc<InitialData>,
    next: usize,
}
impl std::ops::Deref for Initial {
    type Target = InitialData;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
pub(super) enum Next {
    Known(Completed),
    Evaluate { point: Vec<f64>, seed: Option<Seed> },
    NeedSchema,
    Done,
    LinearInfeasible,
    TimeLimit,
}
#[derive(Clone, Copy)]
pub(super) enum Discovery {
    WaitForNewSite,
    AtUniqueSite,
}
impl Initial {
    pub(super) async fn prepare(
        value: Value,
        variables: usize,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let source = RootedValue::in_optional_context(value, c.runtime())?;
        let (x, fval, ineq) = match source.value() {
            Value::Struct(fields) => {
                for name in fields.fields.keys() {
                    if !matches!(name, "X" | "Fval" | "Ineq") {
                        return Err(invalid("InitialPoints has an unsupported field"));
                    }
                }
                (
                    fields
                        .fields
                        .get("X")
                        .ok_or_else(|| invalid("InitialPoints structure requires X"))?,
                    fields.fields.get("Fval"),
                    fields.fields.get("Ineq"),
                )
            }
            value => (value, None, None),
        };
        // Preflight every selected carrier before the first possible gather.
        numeric::preflight(x, "InitialPoints.X", Geometry::MatrixAny, c)?;
        for (value, role, geometry) in [
            (fval, "InitialPoints.Fval", Geometry::Vector),
            (ineq, "InitialPoints.Ineq", Geometry::MatrixAny),
        ] {
            if let Some(value) = value {
                numeric::preflight(value, role, geometry, c)?;
            }
        }
        let points = numeric::prepare(x.clone(), "InitialPoints.X", Geometry::MatrixAny, c).await?;
        let rows = points.rows();
        if points.columns() != variables && !(rows == 0 && points.columns() == 0) {
            return Err(invalid(
                "InitialPoints.X must have one column per original variable",
            ));
        }
        for index in 0..points.stored_len() {
            if !points.stored_value(index).is_finite() {
                return Err(invalid("InitialPoints.X must be finite"));
            }
            c.completed(1).await?;
        }
        if rows == 0 {
            if let Some(value) = fval {
                if numeric::prepare(value.clone(), "InitialPoints.Fval", Geometry::Vector, c)
                    .await?
                    .len()
                    != 0
                {
                    return Err(invalid(
                        "InitialPoints.Fval must match the empty X row extent",
                    ));
                }
            }
            if let Some(value) = ineq {
                if numeric::prepare(value.clone(), "InitialPoints.Ineq", Geometry::MatrixAny, c)
                    .await?
                    .rows()
                    != 0
                {
                    return Err(invalid(
                        "InitialPoints.Ineq must match the empty X row extent",
                    ));
                }
            }
        }
        let measurements = if rows == 0 || fval.is_none() && ineq.is_none() {
            Measurements::Missing
        } else {
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            if let Some(value) = fval {
                fields.insert("Fval", value.clone());
            }
            if let Some(value) = ineq {
                fields.insert("Ineq", value.clone());
            }
            let sequence = runmat_value::ValueSequence::single(Value::Struct(fields.finish()))
                .map_err(crate::sequence::sequence_error_to_runtime)?;
            let returned = match c.runtime() {
                Some(runtime) => RootedValueSequence::in_context(sequence, runtime)?,
                None => RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())?,
            };
            let values = observation::decode(
                returned,
                Form::Initial(NonZeroUsize::new(rows).expect("nonempty initial rows")),
                tolerance,
                c,
            )
            .await?;
            match (fval.is_some(), ineq.is_some()) {
                (true, true) => Measurements::Complete(values),
                (true, false) => Measurements::Partial {
                    supplied: Known::Objective,
                    values,
                },
                (false, true) => Measurements::Partial {
                    supplied: Known::Inequalities,
                    values,
                },
                (false, false) => unreachable!("missing initial measurements"),
            }
        };
        Ok(Self {
            data: Rc::new(InitialData {
                points,
                measurements,
                rows,
                variables,
            }),
            next: 0,
        })
    }
    pub(super) fn done(&self) -> bool {
        self.next == self.rows
    }
    pub(super) fn requires_schema(&self) -> bool {
        matches!(self.measurements, Measurements::Partial { .. }) && !self.done()
    }
    pub(super) async fn next(
        &mut self,
        schema: Option<Schema>,
        domain: &mut Domain,
        linear_tolerance: f64,
        discovery: Discovery,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Next> {
        loop {
            if self.done() {
                return Ok(Next::Done);
            }
            if self.requires_schema()
                && schema.is_none()
                && matches!(discovery, Discovery::WaitForNewSite)
            {
                return Ok(Next::NeedSchema);
            }
            let row = self.next;
            let original = Point::batch_row(self.points.clone(), row, self.rows, self.variables)?;
            let mut point = public_values::capacity(self.variables)?;
            for column in 0..self.variables {
                point.push(original.coordinate(column));
                c.completed(1).await?;
            }
            let point = match proposal::prepare(point, domain, linear_tolerance, c).await? {
                Prepared::Point(point) => point,
                Prepared::LinearInfeasible => return Ok(Next::LinearInfeasible),
                Prepared::TimeLimit => return Ok(Next::TimeLimit),
                Prepared::Unverified => {
                    self.discard();
                    continue;
                }
            };
            let mut unchanged = true;
            for (column, value) in point.iter().enumerate() {
                unchanged &= value.to_bits() == original.coordinate(column).to_bits();
                c.completed(1).await?;
            }
            self.next += 1;
            let (known, supplied) = match &self.measurements {
                Measurements::Missing => (None, None),
                Measurements::Complete(values) => (values.get(row).cloned(), None),
                Measurements::Partial { supplied, values } => {
                    (values.get(row).cloned(), Some(*supplied))
                }
            };
            if !unchanged {
                return Ok(Next::Evaluate { point, seed: None });
            }
            let Some(observation) = known else {
                return Ok(Next::Evaluate { point, seed: None });
            };
            let Some(supplied) = supplied else {
                return Ok(Next::Known(Completed {
                    point: original,
                    observation,
                }));
            };
            let known = observation.schema();
            let Some(actual) = schema else {
                let provided = match supplied {
                    Known::Objective => known.objective == ObjectiveKind::Present,
                    Known::Inequalities => known.inequalities != 0,
                };
                return Ok(Next::Evaluate {
                    point,
                    seed: provided.then_some(Seed {
                        point: original,
                        observation,
                        supplied,
                    }),
                });
            };
            let complete = match supplied {
                Known::Objective => {
                    if known.objective != actual.objective {
                        if known.objective == ObjectiveKind::Absent {
                            return Ok(Next::Evaluate { point, seed: None });
                        }
                        return Err(invalid(
                            "InitialPoints.Fval conflicts with the actual objective schema",
                        ));
                    }
                    actual.inequalities == 0
                }
                Known::Inequalities => {
                    if known.inequalities != actual.inequalities {
                        if known.inequalities == 0 {
                            return Ok(Next::Evaluate { point, seed: None });
                        }
                        return Err(invalid(
                            "InitialPoints.Ineq conflicts with the actual constraint extent",
                        ));
                    }
                    actual.objective == ObjectiveKind::Absent
                }
            };
            if complete {
                return Ok(Next::Known(Completed {
                    point: original,
                    observation,
                }));
            }
            return Ok(Next::Evaluate {
                point,
                seed: Some(Seed {
                    point: original,
                    observation,
                    supplied,
                }),
            });
        }
    }
    fn discard(&mut self) {
        self.next += 1;
    }
}
