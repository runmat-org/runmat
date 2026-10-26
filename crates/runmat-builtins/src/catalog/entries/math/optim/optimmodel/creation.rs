//! Authenticated class-creation targets preserve the ordinary constructor ABI.
use super::*;
use crate::*;
use runmat_types::StaticMethodName;
pub const OPTIMEXPR_ZEROS_LIKE_METHOD: StaticMethodName = StaticMethodName::new("zerosLike");
pub const OPTIMEXPR_ZEROS_METHOD: StaticMethodName = StaticMethodName::new("zeros");
pub const OPTIMEXPR_ZEROS_LIKE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.zerosLike",
    &descriptor::ZEROS_LIKE,
    documentation::method(
        "OptimizationExpression.zerosLike",
        "optimization-expression-zeros-like",
    ),
    inference::expression,
);
pub const OPTIMEXPR_ZEROS_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.zeros",
    &descriptor::VARIADIC,
    documentation::method(
        "OptimizationExpression.zeros",
        "optimization-expression-zeros",
    ),
    inference::expression,
);
