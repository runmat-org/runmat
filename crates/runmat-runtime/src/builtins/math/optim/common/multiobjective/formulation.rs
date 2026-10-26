//! Original objective identities and their attainment rows. This is a smooth
//! epigraph model, not a nonsmooth max callback or a fixed penalty objective.
use super::{invalid, Kind};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

pub(crate) enum Specification {
    Minimax {
        absolute: usize,
    },
    Goal {
        goals: Vec<f64>,
        weights: Vec<f64>,
        equal: usize,
    },
}

/// Desired side of an original goal, separate from attainment magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    AtMost,
    AtLeast,
}
impl Direction {
    pub(crate) const fn coefficient(self) -> f64 {
        match self {
            Self::AtMost => 1.0,
            Self::AtLeast => -1.0,
        }
    }
    const fn opposite(self) -> Self {
        match self {
            Self::AtMost => Self::AtLeast,
            Self::AtLeast => Self::AtMost,
        }
    }
}
/// Original objective identity survives first-k absolute/equality expansion.
pub(crate) struct Row {
    pub objective: usize,
    pub direction: Direction,
    pub goal: f64,
    /// Nonnegative attainment magnitude, distinct from the original raw weight.
    pub weight: f64,
}
pub(crate) struct Formulation {
    pub kind: Kind,
    pub objectives: usize,
    pub rows: Vec<Row>,
    pub(super) first_row: Vec<usize>,
}

pub(crate) fn capacity<T>(length: usize, kind: Kind) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(length).map_err(|source| {
        crate::build_runtime_error(format!(
            "{}: epigraph workspace allocation failed",
            kind.name()
        ))
        .with_builtin(kind.name())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
    })
}

impl Specification {
    pub(crate) async fn bind(
        &self,
        kind: Kind,
        objectives: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Formulation> {
        if objectives == 0 {
            return Err(invalid(
                kind,
                "the objective must return a nonempty real array",
            ));
        }
        let count = match self {
            Self::Minimax { absolute } => *absolute,
            Self::Goal {
                goals,
                weights,
                equal,
            } => {
                if goals.len() != objectives || weights.len() != objectives {
                    return Err(invalid(
                        kind,
                        "goal and weight must match the number of objectives",
                    ));
                }
                *equal
            }
        };
        if count > objectives {
            return Err(invalid(
                kind,
                "the first-objective count exceeds the objective extent",
            ));
        }
        let rows = objectives
            .checked_add(count)
            .ok_or_else(|| invalid(kind, "epigraph row extent overflow"))?;
        let mut bound = capacity(rows, kind)?;
        let mut first_row = capacity(
            objectives
                .checked_add(1)
                .ok_or_else(|| invalid(kind, "objective extent overflow"))?,
            kind,
        )?;
        for objective in 0..objectives {
            first_row.push(bound.len());
            let (goal, weight) = match self {
                Self::Minimax { .. } => (0.0, 1.0),
                Self::Goal { goals, weights, .. } => (goals[objective], weights[objective]),
            };
            if !goal.is_finite() || !weight.is_finite() {
                return Err(invalid(
                    kind,
                    "goals and weights must be finite real values",
                ));
            }
            let direction = if weight < 0.0 {
                Direction::AtLeast
            } else {
                Direction::AtMost
            };
            let magnitude = weight.abs();
            bound.push(Row {
                objective,
                direction,
                goal,
                weight: magnitude,
            });
            if objective < count {
                bound.push(Row {
                    objective,
                    direction: direction.opposite(),
                    goal,
                    weight: magnitude,
                });
            }
            cooperation.completed(1).await?;
        }
        first_row.push(bound.len());
        Ok(Formulation {
            kind,
            objectives,
            rows: bound,
            first_row,
        })
    }
}

impl Formulation {
    pub(crate) fn validate_extent(&self, values: &[f64]) -> BuiltinResult<()> {
        if values.len() != self.objectives {
            return Err(invalid(
                self.kind,
                "objective extent changed during optimization",
            ));
        }
        Ok(())
    }

    /// Defined original coefficients can still overflow at a trial. Only this
    /// numerical absence is shrinkable; operational failures remain outer errors.
    pub(crate) async fn residuals(
        &self,
        values: &[f64],
        gamma: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        self.validate_extent(values)?;
        if !gamma.is_finite() {
            return Ok(None);
        }
        let mut result = capacity(self.rows.len(), self.kind)?;
        for row in &self.rows {
            let value = row.direction.coefficient() * (values[row.objective] - row.goal)
                - row.weight * gamma;
            if !value.is_finite() {
                return Ok(None);
            }
            result.push(value);
            cooperation.completed(1).await?;
        }
        Ok(Some(result))
    }

    /// The normalized worst objective is independent of the free attainment
    /// coordinate. Zero-weight rows remain hard constraints, never divisions.
    pub(crate) async fn worst(
        &self,
        values: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        self.validate_extent(values)?;
        let mut worst: Option<f64> = None;
        for row in &self.rows {
            if row.weight != 0.0 {
                let value =
                    row.direction.coefficient() * (values[row.objective] - row.goal) / row.weight;
                if !value.is_finite() {
                    return Ok(None);
                }
                worst = Some(worst.map_or(value, |old| old.max(value)));
            }
            cooperation.completed(1).await?;
        }
        Ok(worst)
    }
}

#[cfg(test)]
#[path = "formulation/tests.rs"]
mod tests;
