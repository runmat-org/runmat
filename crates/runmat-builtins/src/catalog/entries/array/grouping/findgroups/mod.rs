mod documentation;

use crate::*;
use documentation::DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Grouping vectors, or one table containing grouping variables.",
};
const G: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based double group numbers with NaN at missing observations.",
};
const ID: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ID",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Sorted group identifiers, one output per input, or one identifier table.",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [A];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [G, ID];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[G,ID1,...,IDN] = findgroups(A1,...,AN)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const FINDGROUPS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FINDGROUPS.INVALID_INPUT",
    identifier: Some("RunMat:findgroups:InvalidInput"),
    when: "Inputs are unsupported grouping values, are not compatible vectors, or request an invalid table form.",
    message: "findgroups: invalid grouping input",
};
pub const FINDGROUPS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FINDGROUPS.INTERNAL",
    identifier: Some("RunMat:findgroups:Internal"),
    when: "A group index or identifier output cannot be constructed.",
    message: "findgroups: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [FINDGROUPS_ERROR_INVALID_INPUT, FINDGROUPS_ERROR_INTERNAL];
pub const FINDGROUPS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FINDGROUPS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "findgroups-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "findgroups on GPU-resident grouping data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FindgroupsResidentInputExtension"),
    };
pub const FINDGROUPS_MATRIX_COLUMNS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "findgroups-matrix-as-columns",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "findgroups matrix inputs interpreted as grouping columns are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FindgroupsMatrixColumnsExtension"),
    };
pub const FINDGROUPS_TABLE_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "findgroups-table-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "findgroups(T,selector) is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FindgroupsTableSelectorExtension"),
    };
pub const FINDGROUPS_TIMETABLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "findgroups-timetable-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "findgroups on timetable input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FindgroupsTimetableExtension"),
};
pub const FINDGROUPS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FINDGROUPS_RESIDENT_INPUT_EXTENSION,
    FINDGROUPS_MATRIX_COLUMNS_EXTENSION,
    FINDGROUPS_TABLE_SELECTOR_EXTENSION,
    FINDGROUPS_TIMETABLE_EXTENSION,
];

const VECTOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A1,...,AN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Each integer grouping role is compared from native storage and retains its class in the matching identifier output.",
    }];
const TABLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "integer table variables",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer table variables are exact grouping roles and retain their class in TID.",
}];
pub const FINDGROUPS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[G,ID1,...,IDN] = findgroups(integer_A1,...,integer_AN)",
        inputs: &VECTOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "G is double; each identifier output preserves the corresponding integer class and values, including values above flintmax.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[G,TID] = findgroups(T_with_integer_variables)",
        inputs: &TABLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "G is double; TID preserves variable names, integer classes, and exact group identifiers.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const FINDGROUPS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "findgroups" },
    category: "array/grouping",
    documentation: DOCUMENTATION,
    descriptor: &FINDGROUPS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Grouping(
            GroupingInferenceRule::SortedGroups,
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
    extensions: &FINDGROUPS_EXTENSIONS,
    integer_capabilities: &FINDGROUPS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
