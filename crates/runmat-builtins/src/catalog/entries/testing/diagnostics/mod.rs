//! Testing objects and diagnostic recording; the generated testing parent owns composition.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const TEST_CASE_INPUTS: &[BuiltinParamDescriptor] = &[];
const TEXT_INPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "diagnosticText",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One host string scalar or character row.",
}];
const LOG_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "testCase",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A TestCase object or subclass.",
    },
    BuiltinParamDescriptor {
        name: "diagnostic",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host scalar text or a Diagnostic object.",
    },
];
const LOG_VERBOSE_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "testCase",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A TestCase object or subclass.",
    },
    BuiltinParamDescriptor {
        name: "verbosity",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric scalar level 1 to 4, or Terse, Concise, Detailed, or Verbose; Verbosity enum values are not yet supported.",
    },
    BuiltinParamDescriptor {
        name: "diagnostic",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host scalar text or a Diagnostic object.",
    },
];
const OBJECT_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "object",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Testing object with its concrete runtime class.",
}];
const EMPTY_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty double matrix; logging is a test-context side effect.",
}];

const TEST_CASE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "testCase = matlab.unittest.TestCase()",
    inputs: TEST_CASE_INPUTS,
    outputs: OBJECT_OUTPUT,
}];
const STRING_DIAGNOSTIC_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "diagnostic = matlab.unittest.diagnostics.StringDiagnostic(diagnosticText)",
    inputs: TEXT_INPUT,
    outputs: OBJECT_OUTPUT,
}];
const LOG_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "log(testCase, diagnostic)",
        inputs: LOG_INPUTS,
        outputs: EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "log(testCase, verbosity, diagnostic)",
        inputs: LOG_VERBOSE_INPUTS,
        outputs: EMPTY_OUTPUT,
    },
];

const TEST_CASE_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.TEST_CASE_INPUT",
    identifier: Some("RunMat:Testing:InvalidInput"),
    when: "The direct constructor receives an argument.",
    message: "matlab.unittest.TestCase: too many input arguments",
}];
const DIAGNOSTIC_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.STRING_DIAGNOSTIC_INPUT",
    identifier: Some("RunMat:Testing:InvalidDiagnostic"),
    when: "The input is not one host string scalar or character row.",
    message: "StringDiagnostic: expected scalar text",
}];
const LOG_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.LOG_INPUT", identifier: Some("RunMat:Testing:InvalidDiagnostic"),
    when: "The call has an invalid receiver, arity, verbosity, diagnostic, or no active test context.",
    message: "log requires a TestCase, valid verbosity, and a supported diagnostic in an active test context",
}];

pub const TEST_CASE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: TEST_CASE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: TEST_CASE_ERRORS,
};
pub const STRING_DIAGNOSTIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: STRING_DIAGNOSTIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: DIAGNOSTIC_ERRORS,
};
pub const TEST_CASE_LOG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: LOG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: LOG_ERRORS,
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

struct EntryDefinition {
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    category: &'static str,
    docs: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    effects: &'static [EffectKind],
    purity: BuiltinPurity,
}

const fn entry(definition: EntryDefinition) -> BuiltinCatalogEntry {
    let EntryDefinition {
        provenance,
        name,
        category,
        docs,
        descriptor,
        infer,
        effects,
        purity,
    } = definition;
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category,
        documentation: docs,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects,
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

pub const TEST_CASE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "matlab.unittest.TestCase",
    category: "testing",
    docs: documentation::TEST_CASE,
    descriptor: &TEST_CASE_DESCRIPTOR,
    infer: inference::test_case,
    effects: &[EffectKind::MayThrow],
    purity: BuiltinPurity::Pure,
});
pub const STRING_DIAGNOSTIC_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "matlab.unittest.diagnostics.StringDiagnostic",
    category: "testing/diagnostics",
    docs: documentation::STRING_DIAGNOSTIC,
    descriptor: &STRING_DIAGNOSTIC_DESCRIPTOR,
    infer: inference::string_diagnostic,
    effects: &[EffectKind::MayThrow],
    purity: BuiltinPurity::Pure,
});
pub const TEST_CASE_LOG_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "matlab.unittest.TestCase.log",
    category: "testing/diagnostics",
    docs: documentation::LOG,
    descriptor: &TEST_CASE_LOG_DESCRIPTOR,
    infer: inference::log,
    effects: &[EffectKind::EnvironmentWrite, EffectKind::MayThrow],
    purity: BuiltinPurity::Impure,
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &TEST_CASE_CATALOG_ENTRY,
    &STRING_DIAGNOSTIC_CATALOG_ENTRY,
    &TEST_CASE_LOG_CATALOG_ENTRY,
];
