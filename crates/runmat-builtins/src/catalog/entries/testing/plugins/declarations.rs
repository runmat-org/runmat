//! Plugin constructor descriptors and catalog identities.
use super::{documentation, inference};
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

const NO_INPUTS: &[BuiltinParamDescriptor] = &[];
const FOLDER_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "folder",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "At least one nonempty host string scalar, character row, or string array of folders.",
    },
    BuiltinParamDescriptor {
        name: "additionalFoldersOrOptions",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Further folders or the supported 'IncludingSubfolders' name-value pair.",
    },
];
const OBJECT_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "plugin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A scalar plugin object of the declared concrete class.",
}];
const BASE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "plugin = matlab.unittest.plugins.TestRunnerPlugin()",
    inputs: NO_INPUTS,
    outputs: OBJECT_OUTPUT,
}];
const COVERAGE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "plugin = matlab.unittest.plugins.CodeCoveragePlugin.forFolder(folder, ...)",
    inputs: FOLDER_INPUTS,
    outputs: OBJECT_OUTPUT,
}];
const BASE_ERRORS: &[BuiltinErrorDescriptor] = &[BuiltinErrorDescriptor {
    code: "RM.TESTING.PLUGIN_ARITY",
    identifier: Some("RunMat:Testing:InvalidPlugin"),
    when: "The TestRunnerPlugin constructor receives any input.",
    message: "TestRunnerPlugin: too many input arguments",
}];
const COVERAGE_ERRORS: &[BuiltinErrorDescriptor] = &[
    BuiltinErrorDescriptor {
        code: "RM.TESTING.COVERAGE_PLUGIN_INPUT",
        identifier: Some("RunMat:Testing:InvalidPlugin"),
        when: "No nonempty folder is supplied, a folder is not supported host text, or IncludingSubfolders is absent or not scalar logical or numeric 0/1.",
        message: "CodeCoveragePlugin.forFolder requires supported nonempty folders and a valid IncludingSubfolders option",
    },
    BuiltinErrorDescriptor {
        code: "RM.TESTING.COVERAGE_UNSUPPORTED_OPTION",
        identifier: Some("RunMat:Testing:UnsupportedPluginOption"),
        when: "Producing, MetricLevel, or Filter appears after a folder.",
        message: "CodeCoveragePlugin.forFolder does not support that option",
    },
];

pub const TEST_RUNNER_PLUGIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: BASE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: BASE_ERRORS,
};
pub const CODE_COVERAGE_FOR_FOLDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: COVERAGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: COVERAGE_ERRORS,
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
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "testing/plugins",
        documentation: docs,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[EffectKind::MayThrow],
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

pub const TEST_RUNNER_PLUGIN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.plugins.TestRunnerPlugin",
    documentation::TEST_RUNNER_PLUGIN,
    &TEST_RUNNER_PLUGIN_DESCRIPTOR,
    inference::test_runner_plugin,
);
pub const CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "matlab.unittest.plugins.CodeCoveragePlugin.forFolder",
    documentation::CODE_COVERAGE_FOR_FOLDER,
    &CODE_COVERAGE_FOR_FOLDER_DESCRIPTOR,
    inference::code_coverage_for_folder,
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &TEST_RUNNER_PLUGIN_CATALOG_ENTRY,
    &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
];
