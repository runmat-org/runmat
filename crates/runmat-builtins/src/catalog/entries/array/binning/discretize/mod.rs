mod documentation;

use crate::{
    ArrayInferenceRule, BinningInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior,
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity, BuiltinCompatibility,
    BuiltinCompletionPolicy, BuiltinContractDeclaration, BuiltinContractMaturity,
    BuiltinDescriptor, BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor,
    ALL_INTEGER_CLASSES,
};
use documentation::DISCRETIZE_DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bin indices or replacement values with the shape of X.",
};
const E: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "E",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Computed double bin-edge row vector.",
};
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric or logical values to assign to bins.",
};
const EDGES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "edges",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Increasing numeric edge vector.",
};
const COUNT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive scalar bin count.",
};
const EDGES_OR_COUNT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "edges_or_N",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Increasing numeric edge vector or positive scalar bin count.",
};
const VALUES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "values",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One numeric or text replacement value per bin.",
};
const INCLUDED_EDGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "IncludedEdge",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Literal IncludedEdge option name.",
};
const SIDE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "side",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "left or right.",
};
const REST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional replacement values and IncludedEdge pair.",
};
const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [Y];
const OUTPUT_Y_E: [BuiltinParamDescriptor; 2] = [Y, E];
const INPUT_EDGES: [BuiltinParamDescriptor; 2] = [X, EDGES];
const INPUT_COUNT: [BuiltinParamDescriptor; 2] = [X, COUNT];
const INPUT_VALUES: [BuiltinParamDescriptor; 3] = [X, EDGES_OR_COUNT, VALUES];
const INPUT_EDGE: [BuiltinParamDescriptor; 4] = [X, EDGES_OR_COUNT, INCLUDED_EDGE, SIDE];
const INPUT_REST: [BuiltinParamDescriptor; 3] = [X, COUNT, REST];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "Y = discretize(X, edges)",
        inputs: &INPUT_EDGES,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = discretize(X, N)",
        inputs: &INPUT_COUNT,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = discretize(___, values)",
        inputs: &INPUT_VALUES,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = discretize(___, \"IncludedEdge\", side)",
        inputs: &INPUT_EDGE,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, E] = discretize(X, N, ___)",
        inputs: &INPUT_REST,
        outputs: &OUTPUT_Y_E,
    },
];

pub const DISCRETIZE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISCRETIZE.INVALID_INPUT",
    identifier: Some("RunMat:discretize:InvalidInput"),
    when: "Inputs, edges, labels, options, or requested outputs are invalid or unsupported.",
    message: "discretize: invalid input",
};
pub const DISCRETIZE_ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISCRETIZE.TOO_LARGE",
    identifier: Some("RunMat:discretize:TooLarge"),
    when: "A requested computed-edge vector exceeds the materialization limit.",
    message: "discretize: requested number of bins is too large",
};
pub const DISCRETIZE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISCRETIZE.INTERNAL",
    identifier: Some("RunMat:discretize:Internal"),
    when: "Provider transfer or output construction fails.",
    message: "discretize: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    DISCRETIZE_ERROR_INVALID_INPUT,
    DISCRETIZE_ERROR_TOO_LARGE,
    DISCRETIZE_ERROR_INTERNAL,
];
pub const DISCRETIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const X_AND_EDGES: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is compared from native storage.",
    },
    BuiltinIntegerInputCapability {
        name: "edges",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer edges are validated and compared exactly.",
    },
];
const REPLACEMENT_VALUES: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "values",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer replacement values preserve class; missing assignments use exact zero.",
}];
pub const DISCRETIZE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = discretize(integer_X, integer_edges, ___)", inputs: &X_AND_EDGES, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Explicit-edge assignment remains exact across mixed integer and floating classes; default indices are double." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = discretize(X, edges, integer_values, ___)", inputs: &REPLACEMENT_VALUES, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Replacement output preserves the values class and uses exact zero outside the bins." },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const DISCRETIZE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "discretize" },
    category: "array/binning",
    documentation: DISCRETIZE_DOCUMENTATION,
    descriptor: &DISCRETIZE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Binning(
            BinningInferenceRule::Discretize,
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
        residency: BuiltinResidencyPolicy::Host,
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
    integer_capabilities: &DISCRETIZE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
