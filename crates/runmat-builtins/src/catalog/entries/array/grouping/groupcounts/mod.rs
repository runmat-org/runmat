mod documentation;

use crate::*;
use documentation::DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array, cell array of grouping vectors, table, or timetable.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "groupbins_and_options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Grouping selectors, bin specifications, and name-value options.",
};
const B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double column vector containing group counts.",
};
const BG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BG",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grouping labels, or a cell array with one label vector per grouping role.",
};
const BP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BP",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double column vector containing group percentages.",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [A, OPTIONS];
const OUTPUTS: [BuiltinParamDescriptor; 3] = [B, BG, BP];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[B,BG,BP] = groupcounts(A,groupbins,Name,Value)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const GROUPCOUNTS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPCOUNTS.INVALID_INPUT",
    identifier: Some("RunMat:groupcounts:InvalidInput"),
    when: "Grouping data, selectors, bin specifications, or options are invalid.",
    message: "groupcounts: invalid input",
};
pub const GROUPCOUNTS_ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPCOUNTS.TOO_LARGE",
    identifier: Some("RunMat:groupcounts:TooLarge"),
    when: "The requested empty-group Cartesian product exceeds the materialization limit.",
    message: "groupcounts: result is too large",
};
pub const GROUPCOUNTS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROUPCOUNTS.INTERNAL",
    identifier: Some("RunMat:groupcounts:Internal"),
    when: "Count, percentage, label, or table output construction fails.",
    message: "groupcounts: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    GROUPCOUNTS_ERROR_INVALID_INPUT,
    GROUPCOUNTS_ERROR_TOO_LARGE,
    GROUPCOUNTS_ERROR_INTERNAL,
];
pub const GROUPCOUNTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GROUPCOUNTS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupcounts-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "groupcounts on explicit GPU-resident grouping data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupcountsResidentInputExtension"),
    };
pub const GROUPCOUNTS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupcounts-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "groupcounts with a fixed-width integer bin count or Boolean control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupcountsIntegerControlExtension"),
    };
pub const GROUPCOUNTS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    GROUPCOUNTS_RESIDENT_INPUT_EXTENSION,
    GROUPCOUNTS_INTEGER_CONTROL_EXTENSION,
];

const DATA_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "integer grouping values",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer grouping values and numeric bin edges are compared from native storage.",
}];
const CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "fixed-width integer bin count or Boolean option",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Fixed-width integer controls are accepted only in RunMat compatibility mode; equivalent double and logical controls remain compatible.",
}];
pub const GROUPCOUNTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[B,BG,BP] = groupcounts(integer_grouping_values,___)",
        inputs: &DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B and BP are double; unbinned BG values retain their source integer classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "groupcounts(A,integer_numbins,___)",
        inputs: &CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The bin count is decoded exactly and bounded before allocation.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const GROUPCOUNTS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity {
        name: "groupcounts",
    },
    category: "array/grouping",
    documentation: DOCUMENTATION,
    descriptor: &GROUPCOUNTS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Grouping(
            GroupingInferenceRule::Counts,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &GROUPCOUNTS_EXTENSIONS,
    integer_capabilities: &GROUPCOUNTS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
