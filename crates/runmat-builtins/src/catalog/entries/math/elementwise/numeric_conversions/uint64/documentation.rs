use super::super::documentation_contract::*;

define_integer_conversion_documentation!(
    "uint64",
    "Convert supported values to unsigned 64-bit integer storage.",
    "`uint64(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 64-bit storage.",
    "0 through 18,446,744,073,709,551,615",
    "values = uint64([-Inf 0 3.5 Inf])",
    "values contains 0, 0, 4, and uint64 maximum",
    "assert(isa(values, \"uint64\"));\nassert(isequal(values, [uint64(0) uint64(0) uint64(4) intmax(\"uint64\")]));",
    "[0 2; 1 3]"
);
