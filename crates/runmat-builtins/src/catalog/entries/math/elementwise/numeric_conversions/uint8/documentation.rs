use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "uint8",
    "Convert supported values to unsigned 8-bit integer storage.",
    "`uint8(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 8-bit storage.",
    "0 through 255",
    "values = uint8([-Inf 0 3.5 Inf])",
    "values = uint8([0 0 4 255])",
    "assert(isa(values, \"uint8\"));\nassert(isequal(values, [uint8(0) uint8(0) uint8(4) intmax(\"uint8\")]));",
    "[0 2; 1 3]"
);
