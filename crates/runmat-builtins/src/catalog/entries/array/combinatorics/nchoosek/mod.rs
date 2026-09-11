mod documentation;

use crate::{
    ArrayInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration,
    BuiltinCatalogEntry, BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, CombinatoricsInferenceRule, ALL_INTEGER_CLASSES,
};
use documentation::NCHOOSEK_DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Binomial coefficient or matrix whose rows contain all selected combinations.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n_or_v",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative integer scalar n or vector v containing choices.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative integer number of choices to select.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "b = nchoosek(n, k)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = nchoosek(v, k)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const NCHOOSEK_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NCHOOSEK.INVALID_INPUT",
    identifier: Some("RunMat:nchoosek:InvalidInput"),
    when: "Inputs are missing, malformed, unsupported, or not valid nonnegative integer choices.",
    message: "nchoosek: invalid input",
};
pub const NCHOOSEK_ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NCHOOSEK.TOO_LARGE",
    identifier: Some("RunMat:nchoosek:TooLarge"),
    when: "The combinations matrix exceeds the supported materialization limit.",
    message: "nchoosek: output is too large",
};
pub const NCHOOSEK_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NCHOOSEK.INTERNAL",
    identifier: Some("RunMat:nchoosek:Internal"),
    when: "Output allocation or container construction fails.",
    message: "nchoosek: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    NCHOOSEK_ERROR_INVALID_INPUT,
    NCHOOSEK_ERROR_TOO_LARGE,
    NCHOOSEK_ERROR_INTERNAL,
];
pub const NCHOOSEK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const COEFFICIENT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "n is a real nonnegative integer-valued scalar.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "k is a real nonnegative integer-valued scalar no greater than n.",
    },
];
const VECTOR_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "v",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "v retains its native integer storage class in the combinations.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "k is a real nonnegative integer-valued structural parameter.",
    },
];
pub const NCHOOSEK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "b = nchoosek(n, k)",
        inputs: &COEFFICIENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The coefficient is computed exactly in the selected integer class; representability limits remain form-specific.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = nchoosek(v, k)",
        inputs: &VECTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Combination rows copy authoritative storage without floating-point conversion.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const NCHOOSEK_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "nchoosek" },
    category: "array/combinatorics",
    documentation: NCHOOSEK_DOCUMENTATION,
    descriptor: &NCHOOSEK_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Combinatorics(
            CombinatoricsInferenceRule::SelectionCombinations,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
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
    integer_capabilities: &NCHOOSEK_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
