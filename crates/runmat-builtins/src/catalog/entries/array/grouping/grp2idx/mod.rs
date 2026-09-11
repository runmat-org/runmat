mod documentation;

use crate::*;
use documentation::DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, categorical, datetime, duration, string, cellstr, or character grouping value.",
};
const G: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "g",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Double column vector of one-based group indices, with NaN for missing observations.",
};
const GN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "gN",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of character-vector group names.",
};
const GL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "gL",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Ordered group levels in the input representation, except string input returns cellstr.",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [S];
const OUTPUT_G: [BuiltinParamDescriptor; 1] = [G];
const OUTPUT_G_GN: [BuiltinParamDescriptor; 2] = [G, GN];
const OUTPUT_G_GN_GL: [BuiltinParamDescriptor; 3] = [G, GN, GL];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "g = grp2idx(s)",
        inputs: &INPUTS,
        outputs: &OUTPUT_G,
    },
    BuiltinSignatureDescriptor {
        label: "[g, gN] = grp2idx(s)",
        inputs: &INPUTS,
        outputs: &OUTPUT_G_GN,
    },
    BuiltinSignatureDescriptor {
        label: "[g, gN, gL] = grp2idx(s)",
        inputs: &INPUTS,
        outputs: &OUTPUT_G_GN_GL,
    },
];

pub const GRP2IDX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRP2IDX.INVALID_INPUT",
    identifier: Some("RunMat:grp2idx:InvalidInput"),
    when: "The input is not a supported grouping vector or its representation is malformed.",
    message: "grp2idx: invalid grouping input",
};
pub const GRP2IDX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRP2IDX.INTERNAL",
    identifier: Some("RunMat:grp2idx:Internal"),
    when: "A group output cannot be constructed or restored through its provider.",
    message: "grp2idx: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [GRP2IDX_ERROR_INVALID_INPUT, GRP2IDX_ERROR_INTERNAL];
pub const GRP2IDX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "s",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All fixed-width classes are compared from native storage.",
}];
pub const GRP2IDX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[g,gN,gL] = grp2idx(integer_s)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "g is double, gN is cellstr, and gL preserves the exact integer class and values.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const GRP2IDX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "grp2idx" },
    category: "array/grouping",
    documentation: DOCUMENTATION,
    descriptor: &GRP2IDX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Grouping(
            GroupingInferenceRule::IndexLabels,
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
        residency: BuiltinResidencyPolicy::PreserveInputs,
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
    extensions: &[],
    integer_capabilities: &GRP2IDX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
