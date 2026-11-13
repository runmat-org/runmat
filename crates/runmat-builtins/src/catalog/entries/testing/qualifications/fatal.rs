//! Fatal qualifications abort the current test run on failure.

use super::*;

mod documentation;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "fatalAssertTrue(testCase, actual, diagnostic...)",
    inputs: TRUE_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const FATAL_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.FATAL_ASSERTION_FAILED",
    identifier: Some("RunMat:Test:FatalAssertionFailed"),
    when: "The qualification fails; an active run is aborted after diagnostic recording, or an error is returned outside a test.",
    message: "fatal assertion failed",
};

pub const FATAL_ASSERT_TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, FATAL_ERROR],
};

pub const FATAL_ASSERT_TRUE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.fatalAssertTrue",
    documentation::FATAL_ASSERT_TRUE,
    &FATAL_ASSERT_TRUE_DESCRIPTOR,
    inference::fatal_assert_true,
    BuiltinAsyncBehavior::NeverSuspends,
);

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact,
        ValueKindFact,
    };

    #[test]
    fn fatal_assertion_is_method_only_and_preserves_failure_as_runtime_effect() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::new(vec![]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = inference::fatal_assert_true(&request, &FATAL_ASSERT_TRUE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        assert_eq!(
            FATAL_ASSERT_TRUE_CATALOG_ENTRY.descriptor.completion_policy,
            BuiltinCompletionPolicy::MethodOnly
        );
        assert_eq!(
            FATAL_ASSERT_TRUE_CATALOG_ENTRY.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        );
        assert!(FATAL_ASSERT_TRUE_CATALOG_ENTRY
            .descriptor
            .errors
            .iter()
            .any(|error| error.identifier == Some("RunMat:Test:FatalAssertionFailed")));
    }
}
