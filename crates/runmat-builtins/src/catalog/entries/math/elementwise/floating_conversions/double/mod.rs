mod documentation;

use super::contract::*;
use documentation::DOUBLE_DOCUMENTATION;

define_floating_conversion_contract!(
    "double",
    "DOUBLE",
    NumericClass::Double,
    DOUBLE_DOCUMENTATION,
    "Double-precision output value.",
    "binary64 storage",
    "double-like-prototype",
    "double(X, 'like', prototype) is a RunMat extension",
    "RunMat:compatibility:DoubleLikePrototypeExtension",
    BuiltinIntegerBackendRule::GatherFallback,
    BuiltinIntegerOutputClassRule::Double,
    "Every fixed-width integer class converts directly to IEEE binary64; wide int64 and uint64 values may round.",
    "Host conversion reads authoritative integer storage. Resident conversion uses the owning provider only when it can produce true F64 and otherwise gathers."
);
