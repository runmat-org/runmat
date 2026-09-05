use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "uint32",
    "Convert supported values to unsigned 32-bit integer storage.",
    "`uint32(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 32-bit storage.",
    "0 through 4,294,967,295",
    "values = uint32([-Inf 0 3.5 Inf])",
    "values = uint32([0 0 4 4294967295])",
    "assert(isa(values, \"uint32\"));\nassert(isequal(values, [uint32(0) uint32(0) uint32(4) intmax(\"uint32\")]));",
    "[0 2; 1 3]"
);
