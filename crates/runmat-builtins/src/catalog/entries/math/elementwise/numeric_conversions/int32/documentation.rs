use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "int32",
    "Convert supported values to signed 32-bit integer storage.",
    "`int32(X)` performs shape-preserving, rounded, saturating conversion to native signed 32-bit storage.",
    "-2,147,483,648 through 2,147,483,647",
    "values = int32([-Inf -1.5 0 1.5 Inf])",
    "values = int32([-2147483648 -2 0 2 2147483647])",
    "assert(isa(values, \"int32\"));\nassert(isequal(values, [intmin(\"int32\") int32(-2) int32(0) int32(2) intmax(\"int32\")]));",
    "[-2 0; 1 3]"
);
