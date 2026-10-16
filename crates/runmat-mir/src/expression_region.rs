#[cfg(feature = "compiler")]
mod construction;
mod model;
mod traversal;
mod validation;

pub use model::{MirExpressionRegion, MirExpressionStep};

#[cfg(all(test, feature = "compiler"))]
#[path = "expression_region/tests.rs"]
mod tests;
