mod documentation;

use super::contract::*;
use documentation::SINGLE_DOCUMENTATION;

define_floating_conversion_contract!(
    "single",
    "SINGLE",
    NumericClass::Single,
    SINGLE_DOCUMENTATION,
    "Single-precision output value.",
    "binary32 storage",
    "single-like-output",
    "single with a like output prototype is a RunMat extension",
    "RunMat:compatibility:SingleLikeOutputExtension",
    BuiltinIntegerBackendRule::HostAndGpu,
    BuiltinIntegerOutputClassRule::FunctionSpecific,
    "Every fixed-width integer class converts directly from native integer storage to IEEE binary32 without an intermediate binary64 materialization.",
    "The output uses native single storage with conversion rounding; supported complex integer storage converts each component directly."
);
