use super::*;

mod prepare;
mod response;

pub(super) use self::prepare::prepare_data;
#[cfg(test)]
use self::prepare::{compact_complete_cases, CompleteCaseBuffers};
pub(super) use self::response::response_outcomes;
pub(super) use self::response::ResponseExactness;
#[cfg(test)]
use self::response::{count_outcomes, numeric_outcomes};
#[cfg(test)]
use runmat_value::Tensor;

#[derive(Clone, Debug)]
pub(super) struct PreparedData {
    pub(super) design: Vec<f64>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) outcomes: Vec<Vec<f64>>,
    pub(super) weights: Vec<f64>,
    pub(super) classes: ClassNames,
    pub(super) observations: f64,
}

pub(super) struct ResponseOutcomes {
    outcomes: Vec<Vec<f64>>,
    classes: ClassNames,
    weights: Vec<f64>,
}

pub(super) fn validate_option_weight_length(count: usize, rows: usize) -> BuiltinResult<()> {
    if count != rows {
        return Err(invalid(
            "mnrfit: Weights vector length must match the number of rows in X",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) enum ClassNames {
    Numeric(Vec<f64>),
    Text(Vec<String>),
}

impl ClassNames {
    pub(super) fn len(&self) -> usize {
        match self {
            ClassNames::Numeric(values) => values.len(),
            ClassNames::Text(values) => values.len(),
        }
    }
}

fn checked_finite_add(total: f64, value: f64, role: &str) -> BuiltinResult<f64> {
    let next = total + value;
    if next.is_finite() {
        Ok(next)
    } else {
        Err(invalid(format!(
            "mnrfit: {role} total exceeds finite double range"
        )))
    }
}

#[cfg(test)]
#[path = "data/tests.rs"]
mod tests;
