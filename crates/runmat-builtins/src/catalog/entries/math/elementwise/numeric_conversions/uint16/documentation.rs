use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "uint16",
    "Convert supported values to unsigned 16-bit integer storage.",
    "`uint16(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 16-bit storage.",
    "0 through 65,535",
    "values = uint16([-Inf 0 3.5 Inf])",
    "values = uint16([0 0 4 65535])",
    "assert(isa(values, \"uint16\"));\nassert(isequal(values, [uint16(0) uint16(0) uint16(4) intmax(\"uint16\")]));",
    "[0 2; 1 3]"
);
