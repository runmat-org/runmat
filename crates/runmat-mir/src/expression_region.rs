mod construction;
mod model;
mod traversal;
mod validation;

pub use model::{MirExpressionRegion, MirExpressionStep};

#[cfg(test)]
#[path = "expression_region/tests.rs"]
mod tests;
