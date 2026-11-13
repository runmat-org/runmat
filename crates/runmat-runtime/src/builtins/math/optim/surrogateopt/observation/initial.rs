//! A partial initial measurement substitutes only its supplied lanes at the
//! exact same original point. Each projection retains actual numeric headers;
//! no objective is replayed or copied into another point's result.
use super::*;
#[derive(Clone, Copy)]
pub(in super::super) enum Known {
    Objective,
    Inequalities,
}
impl Observation {
    /// Initial generations are decoded as one shared batch before any
    /// same-point projection. The archive borrows those original headers.
    pub(in super::super) fn initial_headers(
        &self,
    ) -> BuiltinResult<(Option<&DoubleArray>, Option<&DoubleArray>)> {
        let objective = match &self.values.objective {
            Objectives::Absent => None,
            Objectives::Measured(values) => Some(values),
            Objectives::Row { .. } => {
                return Err(invalid("initial template has a projected objective"))
            }
        };
        let constraints = match &self.values.constraints {
            Constraints::Absent => None,
            Constraints::Batch { values, .. } => Some(values),
            _ => return Err(invalid("initial template has projected constraints")),
        };
        Ok((objective, constraints))
    }
    fn objective_projection(&self) -> Objectives {
        match &self.values.objective {
            Objectives::Measured(values) => Objectives::Row {
                values: values.clone(),
                row: self.row,
            },
            value => value.clone(),
        }
    }
    fn constraint_projection(&self) -> Constraints {
        match &self.values.constraints {
            Constraints::Batch { values, columns } => Constraints::Row {
                values: values.clone(),
                row: self.row,
                rows: self.values.rows,
                columns: *columns,
            },
            value => value.clone(),
        }
    }
    pub(in super::super) async fn with_known(
        self,
        known: &Observation,
        supplied: Known,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let actual = self.schema();
        let supplied_schema = known.schema();
        let (objective, constraints) = match supplied {
            Known::Objective => {
                if actual.objective != supplied_schema.objective {
                    return Err(invalid(
                        "InitialPoints.Fval conflicts with the actual objective schema",
                    ));
                }
                (known.objective_projection(), self.constraint_projection())
            }
            Known::Inequalities => {
                if actual.inequalities != supplied_schema.inequalities {
                    return Err(invalid(
                        "InitialPoints.Ineq conflicts with the actual constraint extent",
                    ));
                }
                (self.objective_projection(), known.constraint_projection())
            }
        };
        let values = Rc::new(Values {
            objective,
            constraints,
            schema: actual,
            rows: 1,
        });
        let mut result = Observation {
            values,
            row: 0,
            quality: Quality::Finite,
            feasibility: Feasibility {
                maximum: (actual.inequalities != 0).then_some(f64::NEG_INFINITY),
                violated: 0,
            },
            stored_constraints: actual.inequalities,
        };
        if let Objective::Measured(value) = result.objective() {
            if !value.is_finite() {
                result.quality = Quality::RejectedNaN;
            }
        }
        for column in 0..actual.inequalities {
            let value = result.inequality(column);
            if value.is_nan() {
                result.quality = Quality::RejectedNaN;
                result.feasibility.maximum = Some(f64::NAN);
            } else {
                result.feasibility.maximum = result.feasibility.maximum.map(|previous| {
                    if previous.is_nan() {
                        previous
                    } else {
                        previous.max(value)
                    }
                });
                result.feasibility.violated += usize::from(value > tolerance);
            }
            c.completed(1).await?;
        }
        Ok(result)
    }
}
