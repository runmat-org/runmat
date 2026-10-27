//! Assertion and assumption qualifications. The generated testing parent owns composition.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod documentation;
mod fatal;
mod inference;
mod verification;
#[cfg(test)]
mod tests;

pub use self::fatal::*;
pub use self::verification::*;

const RECEIVER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "testCase",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A TestCase object or subclass.",
};
const ACTUAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "actual",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value being qualified.",
};
const EXPECTED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "expected",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Expected value for equality qualification.",
};
const CONSTRAINT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "constraint",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A compatible matlab.unittest.constraints.Constraint object.",
};
const DIAGNOSTIC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "diagnostic",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional diagnostics; the first scalar text or Diagnostic message replaces the default failure text.",
};
const EMPTY_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty double matrix on success; failures stop or filter the current test.",
}];
const EQUAL_INPUTS: &[BuiltinParamDescriptor] = &[RECEIVER, ACTUAL, EXPECTED, DIAGNOSTIC];
const TRUE_INPUTS: &[BuiltinParamDescriptor] = &[RECEIVER, ACTUAL, DIAGNOSTIC];
const THAT_INPUTS: &[BuiltinParamDescriptor] = &[RECEIVER, ACTUAL, CONSTRAINT, DIAGNOSTIC];

const ASSERT_EQUAL_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "assertEqual(testCase, actual, expected, diagnostic...)",
    inputs: EQUAL_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const ASSERT_THAT_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "assertThat(testCase, actual, constraint, diagnostic...)",
    inputs: THAT_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const ASSERT_TRUE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "assertTrue(testCase, actual, diagnostic...)",
    inputs: TRUE_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const ASSUME_EQUAL_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "assumeEqual(testCase, actual, expected, diagnostic...)",
    inputs: EQUAL_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const ASSUME_TRUE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "assumeTrue(testCase, actual, diagnostic...)",
    inputs: TRUE_INPUTS,
    outputs: EMPTY_OUTPUT,
}];

const RECEIVER_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.QUALIFICATION_RECEIVER",
    identifier: Some("RunMat:Testing:InvalidTestCase"),
    when: "The receiver is not a TestCase object or subclass.",
    message: "first argument must be a matlab.unittest.TestCase",
};
const ASSERTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.ASSERTION_FAILED",
    identifier: Some("RunMat:Test:AssertionFailed"),
    when: "An assertion fails; the active test is aborted after diagnostic recording, or an error is returned outside a test.",
    message: "assertion failed",
};
const ASSUMPTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.ASSUMPTION_FAILED",
    identifier: Some("RunMat:Test:AssumptionFailed"),
    when: "An assumption fails; the active test is filtered after diagnostic recording, or an error is returned outside a test.",
    message: "assumption failed",
};
const CONSTRAINT_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.INVALID_CONSTRAINT",
    identifier: Some("RunMat:Testing:InvalidConstraint"),
    when: "assertThat receives a value that is not a compatible Constraint object.",
    message: "expected a matlab.unittest.constraints.Constraint",
};
const UNSUPPORTED_CONSTRAINT_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TESTING.UNSUPPORTED_CONSTRAINT",
    identifier: Some("RunMat:Testing:UnsupportedConstraint"),
    when: "assertThat receives a compatible Constraint subclass without a supported satisfaction hook.",
    message: "constraint requires an unsupported custom satisfaction hook",
};

pub const ASSERT_EQUAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: ASSERT_EQUAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, ASSERTION_ERROR],
};
pub const ASSERT_THAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: ASSERT_THAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        RECEIVER_ERROR,
        CONSTRAINT_ERROR,
        UNSUPPORTED_CONSTRAINT_ERROR,
        ASSERTION_ERROR,
    ],
};
pub const ASSERT_TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: ASSERT_TRUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, ASSERTION_ERROR],
};
pub const ASSUME_EQUAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: ASSUME_EQUAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, ASSUMPTION_ERROR],
};
pub const ASSUME_TRUE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: ASSUME_TRUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[RECEIVER_ERROR, ASSUMPTION_ERROR],
};

const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

const fn entry(
    name: &'static str,
    docs: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    async_behavior: BuiltinAsyncBehavior,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "testing/qualifications",
        documentation: docs,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior,
            purity: BuiltinPurity::Impure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: match async_behavior {
                BuiltinAsyncBehavior::NeverSuspends => {
                    &[EffectKind::EnvironmentWrite, EffectKind::MayThrow]
                }
                BuiltinAsyncBehavior::MaySuspend
                | BuiltinAsyncBehavior::MaySuspendLocally
                | BuiltinAsyncBehavior::RequiresAsyncRuntime => &[
                    EffectKind::EnvironmentWrite,
                    EffectKind::MaySuspend,
                    EffectKind::MayThrow,
                ],
            },
            capabilities: &[],
        },
        placement: PLACEMENT,
        link: LINK,
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const ASSERT_EQUAL_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.assertEqual",
    documentation::ASSERT_EQUAL,
    &ASSERT_EQUAL_DESCRIPTOR,
    inference::assert_equal,
    BuiltinAsyncBehavior::NeverSuspends,
);
pub const ASSERT_THAT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.assertThat",
    documentation::ASSERT_THAT,
    &ASSERT_THAT_DESCRIPTOR,
    inference::assert_that,
    BuiltinAsyncBehavior::NeverSuspends,
);
pub const ASSERT_TRUE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.assertTrue",
    documentation::ASSERT_TRUE,
    &ASSERT_TRUE_DESCRIPTOR,
    inference::assert_true,
    BuiltinAsyncBehavior::NeverSuspends,
);
pub const ASSUME_EQUAL_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.assumeEqual",
    documentation::ASSUME_EQUAL,
    &ASSUME_EQUAL_DESCRIPTOR,
    inference::assume_equal,
    BuiltinAsyncBehavior::NeverSuspends,
);
pub const ASSUME_TRUE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.assumeTrue",
    documentation::ASSUME_TRUE,
    &ASSUME_TRUE_DESCRIPTOR,
    inference::assume_true,
    BuiltinAsyncBehavior::NeverSuspends,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &ASSERT_EQUAL_CATALOG_ENTRY,
    &ASSERT_THAT_CATALOG_ENTRY,
    &ASSERT_TRUE_CATALOG_ENTRY,
    &ASSUME_EQUAL_CATALOG_ENTRY,
    &ASSUME_TRUE_CATALOG_ENTRY,
    &FATAL_ASSERT_TRUE_CATALOG_ENTRY,
    &VERIFY_EMPTY_CATALOG_ENTRY,
    &VERIFY_EQUAL_CATALOG_ENTRY,
    &VERIFY_FALSE_CATALOG_ENTRY,
    &VERIFY_GREATER_THAN_CATALOG_ENTRY,
    &VERIFY_LESS_THAN_CATALOG_ENTRY,
    &VERIFY_NOT_EMPTY_CATALOG_ENTRY,
    &VERIFY_NOT_EQUAL_CATALOG_ENTRY,
    &VERIFY_THAT_CATALOG_ENTRY,
    &VERIFY_TRUE_CATALOG_ENTRY,
];
