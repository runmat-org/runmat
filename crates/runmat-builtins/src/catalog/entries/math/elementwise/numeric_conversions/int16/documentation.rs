use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "int16",
    "Convert supported values to signed 16-bit integer storage.",
    "`int16(X)` performs shape-preserving, rounded, saturating conversion to native signed 16-bit storage.",
    "-32,768 through 32,767",
    "values = int16([-Inf -1.5 0 1.5 Inf])",
    "values = int16([-32768 -2 0 2 32767])",
    "assert(isa(values, \"int16\"));\nassert(isequal(values, [intmin(\"int16\") int16(-2) int16(0) int16(2) intmax(\"int16\")]));",
    "[-2 0; 1 3]"
);
