use crate::*;

pub(super) const ERRORBAR_Y_ERR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    documented_integer_input!(
        "Y",
        "Y accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "E",
        "E accepts every built-in integer class as symmetric error lengths."
    ),
];

pub(super) const ERRORBAR_X_Y_ERR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    documented_integer_input!(
        "X",
        "X accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "Y",
        "Y accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "E",
        "E accepts every built-in integer class as symmetric error lengths."
    ),
];

pub(super) const ERRORBAR_ASYMMETRIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    documented_integer_input!(
        "X",
        "X accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "Y",
        "Y accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "YNeg",
        "Negative error lengths accept every built-in integer class."
    ),
    documented_integer_input!(
        "YPos",
        "Positive error lengths accept every built-in integer class."
    ),
];

pub(super) const ERRORBAR_BOTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 6] = [
    documented_integer_input!(
        "X",
        "X accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "Y",
        "Y accepts every built-in integer class as coordinate data."
    ),
    documented_integer_input!(
        "YNeg",
        "Negative vertical lengths accept every built-in integer class."
    ),
    documented_integer_input!(
        "YPos",
        "Positive vertical lengths accept every built-in integer class."
    ),
    documented_integer_input!(
        "XNeg",
        "Negative horizontal lengths accept every built-in integer class."
    ),
    documented_integer_input!(
        "XPos",
        "Positive horizontal lengths accept every built-in integer class."
    ),
];

pub const ERRORBAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = errorbar(integer_Y, integer_E)",
        inputs: &ERRORBAR_Y_ERR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative integer storage crosses one explicit client graphics conversion boundary and returns opaque graphics handles.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = errorbar(integer_X, integer_Y, integer_E)",
        inputs: &ERRORBAR_X_Y_ERR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Coordinate and error inputs remain exact until the client graphics conversion boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = errorbar(integer_X, integer_Y, integer_YNeg, integer_YPos)",
        inputs: &ERRORBAR_ASYMMETRIC_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Asymmetric lengths remain exact until the client graphics conversion boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = errorbar(integer_X, integer_Y, integer_YNeg, integer_YPos, integer_XNeg, integer_XPos)",
        inputs: &ERRORBAR_BOTH_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Both-direction integer lengths gather authoritatively before client graphics conversion.",
    },
];
