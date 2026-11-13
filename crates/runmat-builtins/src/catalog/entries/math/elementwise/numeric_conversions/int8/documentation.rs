use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "int8",
    "Convert supported values to signed 8-bit integer storage.",
    "`int8(X)` performs shape-preserving, rounded, saturating conversion to native signed 8-bit storage.",
    "-128 through 127",
    "values = int8([-Inf -1.5 0 1.5 Inf])",
    "values = int8([-128 -2 0 2 127])",
    "assert(isa(values, \"int8\"));\nassert(isequal(values, [intmin(\"int8\") int8(-2) int8(0) int8(2) intmax(\"int8\")]));",
    "[-2 0; 1 3]"
);
