use super::*;
use crate::*;
pub const OPTIMEXPR_UPLUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.uplus",
    &descriptor::UNARY,
    documentation::method(
        "OptimizationExpression.uplus",
        "optimization-expression-uplus",
    ),
    inference::expression,
);
pub const OPTIMEXPR_UMINUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.uminus",
    &descriptor::UNARY,
    documentation::method(
        "OptimizationExpression.uminus",
        "optimization-expression-uminus",
    ),
    inference::expression,
);
pub const OPTIMEXPR_TRANSPOSE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.transpose",
    &descriptor::UNARY,
    documentation::method(
        "OptimizationExpression.transpose",
        "optimization-expression-transpose",
    ),
    inference::expression,
);
pub const OPTIMEXPR_CTRANSPOSE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.ctranspose",
    &descriptor::UNARY,
    documentation::method(
        "OptimizationExpression.ctranspose",
        "optimization-expression-ctranspose",
    ),
    inference::expression,
);
pub const OPTIMEXPR_PLUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.plus",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.plus",
        "optimization-expression-plus",
    ),
    inference::expression,
);
pub const OPTIMEXPR_MINUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.minus",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.minus",
        "optimization-expression-minus",
    ),
    inference::expression,
);
pub const OPTIMEXPR_MTIMES_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.mtimes",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.mtimes",
        "optimization-expression-mtimes",
    ),
    inference::expression,
);
pub const OPTIMEXPR_TIMES_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.times",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.times",
        "optimization-expression-times",
    ),
    inference::expression,
);
pub const OPTIMEXPR_MPOWER_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.mpower",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.mpower",
        "optimization-expression-mpower",
    ),
    inference::expression,
);
pub const OPTIMEXPR_POWER_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.power",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.power",
        "optimization-expression-power",
    ),
    inference::expression,
);
pub const OPTIMEXPR_MLDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.mldivide",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.mldivide",
        "optimization-expression-mldivide",
    ),
    inference::expression,
);
pub const OPTIMEXPR_MRDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.mrdivide",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.mrdivide",
        "optimization-expression-mrdivide",
    ),
    inference::expression,
);
pub const OPTIMEXPR_RDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.rdivide",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.rdivide",
        "optimization-expression-rdivide",
    ),
    inference::expression,
);
pub const OPTIMEXPR_LDIVIDE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.ldivide",
    &descriptor::BINARY,
    documentation::method(
        "OptimizationExpression.ldivide",
        "optimization-expression-ldivide",
    ),
    inference::expression,
);
pub const OPTIMEXPR_LE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.le",
    &descriptor::BINARY,
    documentation::method("OptimizationExpression.le", "optimization-expression-le"),
    inference::inequality,
);
pub const OPTIMEXPR_GE_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.ge",
    &descriptor::BINARY,
    documentation::method("OptimizationExpression.ge", "optimization-expression-ge"),
    inference::inequality,
);
pub const OPTIMEXPR_EQ_CATALOG_ENTRY: BuiltinCatalogEntry = entry::entry(
    "optim.problemdef.OptimizationExpression.eq",
    &descriptor::BINARY,
    documentation::method("OptimizationExpression.eq", "optimization-expression-eq"),
    inference::equality,
);
