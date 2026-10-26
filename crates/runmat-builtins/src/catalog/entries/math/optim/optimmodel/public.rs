use super::*;
use crate::*;
pub const OPTIMEXPR_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optimexpr",
    &OPTIMEXPR_DESCRIPTOR,
    documentation::DOCUMENTATION,
    inference::constructor,
);
pub const OPTIMMODEL_EVALUATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "evaluate",
    &descriptor::BINARY,
    documentation::EVALUATE,
    inference::numeric,
);
pub const OPTIMEXPR_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.subsref",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.subsref",
        "optimization-expression-subsref",
    ),
    inference::read,
);
pub const OPTIMEXPR_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.subsasgn",
    &descriptor::WRITE,
    documentation::method(
        "OptimizationExpression.subsasgn",
        "optimization-expression-subsasgn",
    ),
    inference::expression,
);
pub const OPTIMEXPR_EVALUATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.evaluate",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.evaluate",
        "optimization-expression-evaluate",
    ),
    inference::numeric,
);
pub const OPTIMCONSTRAINT_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationConstraint.subsref",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationConstraint.subsref",
        "optimization-constraint-subsref",
    ),
    inference::read,
);
pub const OPTIMCONSTRAINT_SUBSASGN_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationConstraint.subsasgn",
    &descriptor::WRITE,
    documentation::method(
        "OptimizationConstraint.subsasgn",
        "optimization-constraint-subsasgn",
    ),
    inference::read,
);
pub const OPTIMCONSTRAINT_EVALUATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationConstraint.evaluate",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationConstraint.evaluate",
        "optimization-constraint-evaluate",
    ),
    inference::numeric,
);
