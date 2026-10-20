//! Function-test discovery and runner execution contracts.
//! The generated testing parent owns composition.

use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

mod documentation;
mod inference;
#[cfg(test)]
mod tests;

const HANDLES: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "functions",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of named local function handles, normally from localfunctions.",
}];
const OPTIONS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor { name: "nameValue", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "Reserved name-value arguments; all nonempty option lists currently return UnsupportedRunnerOption." }];
const RUNNER_AND_PLUGIN: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "runner",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "TestRunner handle object.",
    },
    BuiltinParamDescriptor {
        name: "plugin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "TestRunnerPlugin object or subclass.",
    },
];
const RUNNER_AND_SUITE: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "runner",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "TestRunner handle object.",
    },
    BuiltinParamDescriptor {
        name: "suite",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "TestSuite selection resolved by the active Core test executor.",
    },
];
const SUITE_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "suite",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "TestSuite scalar or row array containing functions whose names begin with test.",
}];
const RUNNER_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "runner",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "TestRunner handle; addPlugin returns the same handle.",
}];
const RESULT_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structured TestResult scalar or row array, one element per executed test.",
}];

pub const FUNCTIONTESTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "suite = functiontests(functions)",
        inputs: HANDLES,
        outputs: SUITE_OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[BuiltinErrorDescriptor {
        code: "RM.TESTING.FUNCTIONTESTS_INPUT",
        identifier: Some("RunMat:functiontests:InvalidInput"),
        when: "Input is not a cell array or any element is not a named function handle.",
        message: "functiontests requires a cell array of named function handles",
    }],
};
pub const WITH_TEXT_OUTPUT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "runner = matlab.unittest.TestRunner.withTextOutput(nameValue...)",
        inputs: OPTIONS,
        outputs: RUNNER_OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.TESTING.RUNNER_OPTIONS",
            identifier: Some("RunMat:Testing:InvalidRunner"),
            when: "The number of option arguments is odd or runner allocation fails.",
            message: "TestRunner.withTextOutput options must be name-value pairs",
        },
        BuiltinErrorDescriptor {
            code: "RM.TESTING.UNSUPPORTED_RUNNER_OPTION",
            identifier: Some("RunMat:Testing:UnsupportedRunnerOption"),
            when: "Any name-value argument is supplied; runner options are not implemented yet.",
            message: "TestRunner.withTextOutput options are not supported yet",
        },
    ],
};
pub const ADD_PLUGIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor { signatures: &[BuiltinSignatureDescriptor { label: "runner = runner.addPlugin(plugin)", inputs: RUNNER_AND_PLUGIN, outputs: RUNNER_OUTPUT }], output_mode: BuiltinOutputMode::Fixed, completion_policy: BuiltinCompletionPolicy::MethodOnly, errors: &[BuiltinErrorDescriptor { code: "RM.TESTING.RUNNER_PLUGIN", identifier: Some("RunMat:Testing:InvalidRunner"), when: "The receiver is not a TestRunner handle, the plugin is incompatible, or plugin storage cannot be updated.", message: "addPlugin requires a TestRunner handle and TestRunnerPlugin" }] };
pub const TEST_RUNNER_RUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "results = runner.run(suite)",
        inputs: RUNNER_AND_SUITE,
        outputs: RESULT_OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        BuiltinErrorDescriptor {
            code: "RM.TESTING.RUNNER_RUN",
            identifier: Some("RunMat:Testing:InvalidRunner"),
            when: "The receiver is not a TestRunner handle.",
            message: "TestRunner.run requires a TestRunner handle",
        },
        BuiltinErrorDescriptor {
            code: "RM.TESTING.EXECUTOR_REQUIRED",
            identifier: Some("RunMat:Testing:RequiresExecutor"),
            when: "No Core test executor is installed for this runtime session.",
            message: "test execution requires an active Core test executor",
        },
    ],
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
    name: &'static str,
    docs: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    async_behavior: BuiltinAsyncBehavior,
    purity: BuiltinPurity,
    effects: &'static [EffectKind],
    capabilities: &'static [CapabilityRequirement],
}

const fn entry(definition: EntryDefinition) -> BuiltinCatalogEntry {
    let EntryDefinition {
        name,
        docs,
        descriptor,
        infer,
        async_behavior,
        purity,
        effects,
        capabilities,
    } = definition;
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "testing",
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
            environment_effect: None,
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

pub const FUNCTIONTESTS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    name: "functiontests",
    docs: documentation::FUNCTIONTESTS,
    descriptor: &FUNCTIONTESTS_DESCRIPTOR,
    infer: inference::functiontests,
    async_behavior: BuiltinAsyncBehavior::NeverSuspends,
    // Labels come from the invocation's already-bound source record. This
    // reads neither session environment variables nor host/filesystem state.
    purity: BuiltinPurity::DeterministicReadOnly,
    effects: &[EffectKind::MayThrow],
    capabilities: &[],
});
pub const WITH_TEXT_OUTPUT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    name: "matlab.unittest.TestRunner.withTextOutput",
    docs: documentation::WITH_TEXT_OUTPUT,
    descriptor: &WITH_TEXT_OUTPUT_DESCRIPTOR,
    infer: inference::with_text_output,
    async_behavior: BuiltinAsyncBehavior::MaySuspend,
    purity: BuiltinPurity::Pure,
    effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
    capabilities: &[],
});
pub const ADD_PLUGIN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    name: "matlab.unittest.TestRunner.addPlugin",
    docs: documentation::ADD_PLUGIN,
    descriptor: &ADD_PLUGIN_DESCRIPTOR,
    infer: inference::add_plugin,
    async_behavior: BuiltinAsyncBehavior::NeverSuspends,
    purity: BuiltinPurity::Impure,
    effects: &[EffectKind::EnvironmentWrite, EffectKind::MayThrow],
    capabilities: &[],
});
pub const TEST_RUNNER_RUN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(EntryDefinition {
    name: "matlab.unittest.TestRunner.run",
    docs: documentation::RUN,
    descriptor: &TEST_RUNNER_RUN_DESCRIPTOR,
    infer: inference::run,
    async_behavior: BuiltinAsyncBehavior::MaySuspend,
    purity: BuiltinPurity::Impure,
    effects: &[
        EffectKind::EnvironmentRead,
        EffectKind::FilesystemRead,
        EffectKind::HostCallback,
        EffectKind::MaySuspend,
        EffectKind::MayThrow,
        EffectKind::Unknown,
    ],
    capabilities: &[CapabilityRequirement::Filesystem],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &FUNCTIONTESTS_CATALOG_ENTRY,
    &WITH_TEXT_OUTPUT_CATALOG_ENTRY,
    &ADD_PLUGIN_CATALOG_ENTRY,
    &TEST_RUNNER_RUN_CATALOG_ENTRY,
];
