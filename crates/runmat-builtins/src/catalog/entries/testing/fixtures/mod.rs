//! Test fixture construction and lifecycle methods.

use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const PATH_INPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One host path. Multiple folders and name-value options are not yet supported.",
}];
const NO_INPUTS: &[BuiltinParamDescriptor] = &[];
const RECEIVER_AND_FIXTURE: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "testCase",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "TestCase or subclass in an active test lifecycle.",
    },
    BuiltinParamDescriptor {
        name: "fixture",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Handle fixture object.",
    },
];
const TEARDOWN_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "testCase",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "TestCase or subclass in an active test lifecycle.",
    },
    BuiltinParamDescriptor {
        name: "teardownFcn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle invoked at teardown.",
    },
    BuiltinParamDescriptor {
        name: "input",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Arguments retained for the teardown callback.",
    },
];
const OBJECT_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "fixture",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Concrete handle fixture object.",
}];
const EMPTY_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty double matrix; teardown registration is an effect.",
}];

const PATH_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "fixture = matlab.unittest.fixtures.PathFixture(folder)",
    inputs: PATH_INPUT,
    outputs: OBJECT_OUTPUT,
}];
const TEMP_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "fixture = matlab.unittest.fixtures.TemporaryFolderFixture()",
    inputs: NO_INPUTS,
    outputs: OBJECT_OUTPUT,
}];
const APPLY_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "fixture = applyFixture(testCase, fixture)",
    inputs: RECEIVER_AND_FIXTURE,
    outputs: OBJECT_OUTPUT,
}];
const TEARDOWN_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "addTeardown(testCase, teardownFcn, input, ...)",
    inputs: TEARDOWN_INPUTS,
    outputs: EMPTY_OUTPUT,
}];
const FIXTURE_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.INVALID_FIXTURE",
    identifier: Some("RunMat:Testing:InvalidFixture"),
    when: "The fixture form, receiver, handle or lifecycle is invalid, or setup fails.",
    message: "A supported fixture and an active test lifecycle are required",
}];
const TEARDOWN_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.ADD_TEARDOWN",
    identifier: Some("RunMat:Testing:AddTeardown"),
    when: "The callback or active lifecycle is invalid, or the command limit is reached.",
    message: "A function handle and an active test lifecycle are required",
}];

pub const PATH_FIXTURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: PATH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: FIXTURE_ERRORS,
};
pub const TEMPORARY_FOLDER_FIXTURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: TEMP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: FIXTURE_ERRORS,
};
pub const APPLY_FIXTURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: APPLY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: FIXTURE_ERRORS,
};
pub const ADD_TEARDOWN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: TEARDOWN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: TEARDOWN_ERRORS,
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
    effects: &'static [EffectKind],
    capabilities: &'static [CapabilityRequirement],
    async_behavior: BuiltinAsyncBehavior,
    environment_effect: Option<BuiltinEnvironmentEffect>,
    purity: BuiltinPurity,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "testing/fixtures",
        documentation: docs,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior,
            purity,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect,
            effects,
            capabilities,
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

pub const PATH_FIXTURE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.fixtures.PathFixture",
    documentation::PATH_FIXTURE,
    &PATH_FIXTURE_DESCRIPTOR,
    inference::path_fixture,
    &[EffectKind::MayThrow],
    &[],
    BuiltinAsyncBehavior::MaySuspend,
    None,
    BuiltinPurity::Pure,
);
pub const TEMPORARY_FOLDER_FIXTURE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.fixtures.TemporaryFolderFixture",
    documentation::TEMPORARY_FOLDER_FIXTURE,
    &TEMPORARY_FOLDER_FIXTURE_DESCRIPTOR,
    inference::temporary_folder_fixture,
    &[EffectKind::MayThrow],
    &[],
    BuiltinAsyncBehavior::MaySuspend,
    None,
    BuiltinPurity::Pure,
);
pub const APPLY_FIXTURE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.applyFixture",
    documentation::APPLY_FIXTURE,
    &APPLY_FIXTURE_DESCRIPTOR,
    inference::apply_fixture,
    &[
        EffectKind::EnvironmentRead,
        EffectKind::EnvironmentWrite,
        EffectKind::FilesystemRead,
        EffectKind::FilesystemWrite,
        EffectKind::MaySuspend,
        EffectKind::MayThrow,
    ],
    &[CapabilityRequirement::Filesystem],
    BuiltinAsyncBehavior::MaySuspend,
    Some(BuiltinEnvironmentEffect::PathMutation),
    BuiltinPurity::Impure,
);
pub const ADD_TEARDOWN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.TestCase.addTeardown",
    documentation::ADD_TEARDOWN,
    &ADD_TEARDOWN_DESCRIPTOR,
    inference::add_teardown,
    &[
        EffectKind::HostCallback,
        EffectKind::Unknown,
        EffectKind::MayThrow,
    ],
    &[],
    BuiltinAsyncBehavior::NeverSuspends,
    None,
    BuiltinPurity::Impure,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &PATH_FIXTURE_CATALOG_ENTRY,
    &TEMPORARY_FOLDER_FIXTURE_CATALOG_ENTRY,
    &APPLY_FIXTURE_CATALOG_ENTRY,
    &ADD_TEARDOWN_CATALOG_ENTRY,
];
