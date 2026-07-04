pub const MODULE_PURPOSE: &str = "curve endpoint, projection, length, growth, and loop checks";

mod checks;
mod types;

#[cfg(test)]
mod tests;

pub use checks::validate_curve_discretization;
pub use types::{CurveValidationError, CurveValidationOptions, CurveValidationReport};
