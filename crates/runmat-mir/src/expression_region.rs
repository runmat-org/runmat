#[cfg(feature = "compiler")]
mod construction;
mod model;
mod traversal;
mod validation;

pub(crate) fn validate_preparation_steps(steps: &[MirExpressionStep]) -> Result<(), String> {
    validation::validate_steps_only(steps)
}

pub(crate) fn validate_fixed_preparation_nesting(value: &crate::MirRvalue) -> Result<(), String> {
    validation::reject_nested_fixed_preparation(value)
}

pub use model::{
    MirExpressionRegion, MirExpressionRegionRef, MirExpressionStep, MirPreparedExpressionRegion,
};

#[cfg(all(test, feature = "compiler"))]
#[path = "expression_region/tests.rs"]
mod tests;
