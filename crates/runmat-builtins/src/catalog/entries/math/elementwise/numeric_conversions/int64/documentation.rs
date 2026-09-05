use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "int64",
    "Convert supported values to signed 64-bit integer storage.",
    "`int64(X)` performs shape-preserving, rounded, saturating conversion to native signed 64-bit storage.",
    "-9,223,372,036,854,775,808 through 9,223,372,036,854,775,807",
    "values = int64([-Inf -1.5 0 1.5 Inf])",
    "values contains int64 minimum, -2, 0, 2, and int64 maximum",
    "assert(isa(values, \"int64\"));\nassert(isequal(values, [intmin(\"int64\") int64(-2) int64(0) int64(2) intmax(\"int64\")]));",
    "[-2 0; 1 3]"
);
