//! One rooted callback generation, decoded once without replay or row copies.
use super::{
    invalid,
    numeric::{self, DoubleArray, Geometry},
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::execution::RootedValueSequence;
use crate::BuiltinResult;
use runmat_value::Value;
use std::{num::NonZeroUsize, rc::Rc};
mod initial;
pub(super) use initial::Known;

#[derive(Clone, Copy)]
pub(super) enum Form {
    Point,
    Vectorized(NonZeroUsize),
    Initial(NonZeroUsize),
}
impl Form {
    fn rows(self) -> usize {
        match self {
            Self::Point => 1,
            Self::Vectorized(n) | Self::Initial(n) => n.get(),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ObjectiveKind {
    Absent,
    Present,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Schema {
    pub objective: ObjectiveKind,
    pub inequalities: usize,
}
impl Schema {
    pub(super) fn lanes(self) -> Option<usize> {
        self.inequalities
            .checked_add(usize::from(self.objective == ObjectiveKind::Present))
    }
}
#[derive(Clone, Copy)]
pub(super) enum Objective {
    Absent,
    Measured(f64),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Quality {
    Finite,
    RejectedNaN,
}
#[derive(Clone, Copy)]
pub(super) struct Feasibility {
    /// Raw max(Ineq), including implicit zeros; absent for no constraints.
    pub maximum: Option<f64>,
    pub violated: usize,
}
#[derive(Clone, Copy)]
pub(super) struct Measurement {
    pub objective: Objective,
    pub nonlinear: Feasibility,
}
#[derive(Clone)]
enum Objectives {
    Absent,
    Measured(DoubleArray),
    Row { values: DoubleArray, row: usize },
}
#[derive(Clone)]
enum Constraints {
    Absent,
    Point(DoubleArray),
    Batch {
        values: DoubleArray,
        columns: usize,
    },
    Row {
        values: DoubleArray,
        row: usize,
        rows: usize,
        columns: usize,
    },
}
struct Values {
    objective: Objectives,
    constraints: Constraints,
    schema: Schema,
    rows: usize,
}
#[derive(Clone)]
pub(super) struct Observation {
    values: Rc<Values>,
    row: usize,
    pub quality: Quality,
    feasibility: Feasibility,
    stored_constraints: usize,
}
impl Observation {
    /// Only a complete finite callback row can participate in incumbent
    /// ranking. Rejected rows retain their actual payload for trials output.
    pub(super) fn measurement(&self) -> Option<Measurement> {
        (self.quality == Quality::Finite).then(|| Measurement {
            objective: self.objective(),
            nonlinear: self.feasibility,
        })
    }
    pub(super) fn schema(&self) -> Schema {
        self.values.schema
    }
    pub(super) fn maximum(&self) -> Option<f64> {
        self.feasibility.maximum
    }
    pub(super) fn objective(&self) -> Objective {
        match &self.values.objective {
            Objectives::Absent => Objective::Absent,
            Objectives::Measured(values) => Objective::Measured(values.get(self.row)),
            Objectives::Row { values, row } => Objective::Measured(values.get(*row)),
        }
    }
    pub(super) fn inequality(&self, column: usize) -> f64 {
        assert!(
            column < self.values.schema.inequalities,
            "validated constraint column"
        );
        match &self.values.constraints {
            Constraints::Absent => unreachable!("empty constraint extent"),
            Constraints::Point(values) => values.get(column),
            Constraints::Batch { values, columns } => {
                assert!(column < *columns, "validated batch constraint column");
                values.get(column * self.values.rows + self.row)
            }
            Constraints::Row {
                values,
                row,
                rows,
                columns,
            } => {
                assert!(column < *columns, "validated projected constraint column");
                values.get(column * *rows + *row)
            }
        }
    }
}

/// Shape checks for ALL selected fields precede the first transfer. Empty
/// Fval is legal only as the documented structured feasibility-only form.
fn shape(value: &Value, role: &str) -> BuiltinResult<(Vec<usize>, usize)> {
    let dimensions = crate::builtins::math::optim::common::constrained_numeric::shape(value)
        .ok_or_else(|| invalid(format!("{role} must be a concrete numeric array")))?;
    let count = dimensions
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| invalid(format!("{role} extent overflow")))?;
    Ok((dimensions, count))
}
fn matrix(dimensions: &[usize], rows: usize, columns: Option<usize>) -> bool {
    dimensions.first().copied().unwrap_or(1) == rows
        && columns.is_none_or(|n| dimensions.get(1).copied().unwrap_or(1) == n)
        && dimensions.iter().skip(2).all(|&n| n == 1)
}
fn vector(dimensions: &[usize]) -> bool {
    dimensions.iter().filter(|&&n| n != 1).count() <= 1
}
pub(super) async fn decode(
    returned: RootedValueSequence,
    form: Form,
    constraint_tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Observation>> {
    if !constraint_tolerance.is_finite() || constraint_tolerance < 0.0 {
        return Err(invalid("invalid nonlinear constraint tolerance"));
    }
    cooperation.completed(0).await?;
    let returned = {
        let _entry = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned
            .resolve_sequence(
                runmat_types::SequenceUse::from_requested_outputs(
                    runmat_types::RequestedOutputCount::Exactly(1),
                ),
                crate::sequence::SequenceResolutionContext::default(),
            )?
            .try_into_single()?
    };
    let rows = form.rows();
    let (objective, constraints, structured) = match returned.value() {
        Value::Struct(value) => {
            let objective = value.fields.get("Fval");
            let constraints = value.fields.get("Ineq");
            if objective.is_none() && constraints.is_none() {
                return Err(invalid("objective structure requires Fval or Ineq"));
            }
            (objective, constraints, true)
        }
        value => (Some(value), None, false),
    };
    let mut empty_objective = objective.is_none();
    if let Some(value) = objective {
        let (dimensions, count) = shape(value, "Fval")?;
        empty_objective = structured && count == 0;
        if !empty_objective
            && match form {
                Form::Point => count != 1,
                Form::Vectorized(_) => !matrix(&dimensions, rows, Some(1)),
                Form::Initial(_) => count != rows || !vector(&dimensions),
            }
        {
            return Err(invalid("Fval must be scalar or an actual batch column"));
        }
    }
    let mut columns = 0;
    if let Some(value) = constraints {
        let (dimensions, count) = shape(value, "Ineq")?;
        if count != 0 {
            columns = match form {
                Form::Point => {
                    if !vector(&dimensions) {
                        return Err(invalid("Ineq must be a real vector"));
                    }
                    count
                }
                Form::Vectorized(_) | Form::Initial(_) => {
                    if !matrix(&dimensions, rows, None) {
                        return Err(invalid(
                            "vectorized Ineq must have one row per actual point",
                        ));
                    }
                    dimensions.get(1).copied().unwrap_or(1)
                }
            };
        }
    }
    columns
        .checked_add(usize::from(!empty_objective))
        .ok_or_else(|| invalid("objective/constraint lane extent overflow"))?;
    for (value, role) in [(objective, "Fval"), (constraints, "Ineq")] {
        if let Some(value) = value {
            numeric::preflight(value, role, Geometry::Array, cooperation)?;
        }
    }
    let objective = if empty_objective {
        Objectives::Absent
    } else {
        let values = numeric::prepare(
            objective.expect("nonempty objective").clone(),
            "Fval",
            Geometry::Elements(rows),
            cooperation,
        )
        .await?;
        if matches!(form, Form::Vectorized(_)) && !values.matrix_matches(rows, Some(1)) {
            return Err(invalid(
                "Fval changed its actual batch column during preparation",
            ));
        }
        Objectives::Measured(values)
    };
    let constraints = if columns == 0 {
        Constraints::Absent
    } else {
        let values = numeric::prepare(
            constraints.expect("nonempty constraints").clone(),
            "Ineq",
            Geometry::Elements(
                rows.checked_mul(columns)
                    .ok_or_else(|| invalid("constraint batch extent overflow"))?,
            ),
            cooperation,
        )
        .await?;
        match form {
            Form::Point => {
                if !values.vector_like() {
                    return Err(invalid("Ineq changed its vector extent during preparation"));
                }
                Constraints::Point(values)
            }
            Form::Vectorized(_) | Form::Initial(_) => {
                if !values.matrix_matches(rows, Some(columns)) {
                    return Err(invalid(
                        "Ineq changed its actual batch rows during preparation",
                    ));
                }
                Constraints::Batch { values, columns }
            }
        }
    };
    let schema = Schema {
        objective: match &objective {
            Objectives::Absent => ObjectiveKind::Absent,
            Objectives::Measured(_) | Objectives::Row { .. } => ObjectiveKind::Present,
        },
        inequalities: columns,
    };
    schema
        .lanes()
        .ok_or_else(|| invalid("objective/constraint lane extent overflow"))?;
    let values = Rc::new(Values {
        objective,
        constraints,
        schema,
        rows,
    });
    // The rooted returned aggregate stays alive through every preparation
    // above. Only selected numeric storage survives in Values; ignored fields
    // are never inspected and no raw historic callback workspace is retained.
    let mut observations = runmat_value::exact_capacity_vec(rows).map_err(|source| {
        crate::build_runtime_error("surrogateopt: observation metadata allocation failed")
            .with_builtin(super::NAME)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })?;
    for row in 0..rows {
        let mut observation = Observation {
            values: values.clone(),
            row,
            quality: Quality::Finite,
            feasibility: Feasibility {
                maximum: (columns != 0).then_some(f64::NEG_INFINITY),
                violated: 0,
            },
            stored_constraints: 0,
        };
        if let Objective::Measured(value) = observation.objective() {
            if value.is_infinite() {
                return Err(invalid("Fval must be finite or NaN"));
            }
            if value.is_nan() {
                observation.quality = Quality::RejectedNaN;
            }
        }
        observations.push(observation);
        cooperation.completed(1).await?;
    }
    match &values.constraints {
        Constraints::Row { .. } => {
            unreachable!("callback decode never creates an initial projection")
        }
        Constraints::Absent => {}
        Constraints::Point(array) => {
            quality_scan(
                array,
                true,
                constraint_tolerance,
                &mut observations,
                cooperation,
            )
            .await?
        }
        Constraints::Batch { values: array, .. } => {
            quality_scan(
                array,
                false,
                constraint_tolerance,
                &mut observations,
                cooperation,
            )
            .await?
        }
    }
    for observation in &mut observations {
        if observation.stored_constraints < columns {
            observation.feasibility.maximum =
                observation
                    .feasibility
                    .maximum
                    .map(|v| if v.is_nan() { v } else { v.max(0.0) });
        }
        cooperation.completed(1).await?;
    }
    Ok(observations)
}

async fn quality_scan(
    values: &DoubleArray,
    point: bool,
    tolerance: f64,
    observations: &mut [Observation],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    for index in 0..values.stored_len() {
        let value = values.stored_value(index);
        let row = if point { 0 } else { values.stored_row(index) };
        let observation = &mut observations[row];
        observation.stored_constraints = observation
            .stored_constraints
            .checked_add(1)
            .ok_or_else(|| invalid("stored constraint count overflow"))?;
        if value.is_infinite() {
            return Err(invalid("Ineq must be finite or NaN"));
        }
        if value.is_nan() {
            observation.quality = Quality::RejectedNaN;
            observation.feasibility.maximum = Some(f64::NAN);
        } else {
            observation.feasibility.maximum =
                observation
                    .feasibility
                    .maximum
                    .map(|v| if v.is_nan() { v } else { v.max(value) });
            if value > tolerance {
                observation.feasibility.violated = observation
                    .feasibility
                    .violated
                    .checked_add(1)
                    .ok_or_else(|| invalid("violated constraint count overflow"))?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
