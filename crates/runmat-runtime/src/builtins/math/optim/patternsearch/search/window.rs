//! Finite proposal windows; original Linear rows retain feasibility authority.
use super::super::invalid;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(in crate::builtins::math::optim::patternsearch) struct Window {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl Window {
    /// Moves the existing decoded bound storage. Default direct polling does
    /// not retain these arrays or create any sampling coordinates.
    pub async fn new(
        mut lower: Vec<f64>,
        mut upper: Vec<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if lower.len() != upper.len() {
            return Err(invalid("search bound extent mismatch"));
        }
        for (a, b) in lower.iter_mut().zip(&mut upper) {
            if a.is_nan() || b.is_nan() || *a > *b || *a == f64::INFINITY || *b == f64::NEG_INFINITY
            {
                return Err(invalid("search bounds are inconsistent"));
            }
            match (a.is_finite(), b.is_finite()) {
                (false, false) => {
                    *a = -1.0;
                    *b = 1.0;
                }
                (true, false) => {
                    let trial = *a + 2.0;
                    *b = if trial > *a && trial.is_finite() {
                        trial
                    } else {
                        let adjacent = a.next_up();
                        if adjacent.is_finite() {
                            adjacent
                        } else {
                            *a
                        }
                    };
                }
                (false, true) => {
                    let trial = *b - 2.0;
                    *a = if trial < *b && trial.is_finite() {
                        trial
                    } else {
                        let adjacent = b.next_down();
                        if adjacent.is_finite() {
                            adjacent
                        } else {
                            *b
                        }
                    };
                }
                (true, true) => {}
            }
            cooperation.completed(1).await?;
        }
        Ok(Self { lower, upper })
    }
}
