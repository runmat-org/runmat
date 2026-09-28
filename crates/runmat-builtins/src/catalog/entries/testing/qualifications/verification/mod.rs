//! Non-aborting verification qualifications.

use super::*;

mod documentation;
#[cfg(test)]
mod tests;

const FLOOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "floor",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Comparison threshold; the current runtime compares numeric scalar values only.",
};
const GREATER_INPUTS: &[BuiltinParamDescriptor] = &[RECEIVER, ACTUAL, FLOOR, DIAGNOSTIC];
const EMPTY_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "verifyEmpty(testCase, actual, diagnostic...)",
    inputs: TRUE_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const EQUAL_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "verifyEqual(testCase, actual, expected, diagnostic...)",
    inputs: EQUAL_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const FALSE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "verifyFalse(testCase, actual, diagnostic...)",
    inputs: TRUE_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const GREATER_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "verifyGreaterThan(testCase, actual, floor, diagnostic...)",
    inputs: GREATER_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const VERIFICATION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.VERIFICATION_FAILED",
    identifier: Some("RunMat:Test:VerificationFailed"),
    when: "A verification fails outside an active test; inside an active test, its diagnostic is recorded and execution continues.",
    message: "verification failed",
};

pub const VERIFY_EMPTY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: EMPTY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, VERIFICATION_ERROR],
};
pub const VERIFY_EQUAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: EQUAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, VERIFICATION_ERROR],
};
pub const VERIFY_FALSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: FALSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, VERIFICATION_ERROR],
};
pub const VERIFY_GREATER_THAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: GREATER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, VERIFICATION_ERROR],
};

pub const VERIFY_EMPTY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.verifyEmpty",
    documentation::VERIFY_EMPTY,
    &VERIFY_EMPTY_DESCRIPTOR,
    inference::verify_empty,
    BuiltinAsyncBehavior::MaySuspend,
);
pub const VERIFY_EQUAL_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.verifyEqual",
    documentation::VERIFY_EQUAL,
    &VERIFY_EQUAL_DESCRIPTOR,
    inference::verify_equal,
    BuiltinAsyncBehavior::NeverSuspends,
);
pub const VERIFY_FALSE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.verifyFalse",
    documentation::VERIFY_FALSE,
    &VERIFY_FALSE_DESCRIPTOR,
    inference::verify_false,
    BuiltinAsyncBehavior::NeverSuspends,
);
pub const VERIFY_GREATER_THAN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.verifyGreaterThan",
    documentation::VERIFY_GREATER_THAN,
    &VERIFY_GREATER_THAN_DESCRIPTOR,
    inference::verify_greater_than,
    BuiltinAsyncBehavior::NeverSuspends,
);
