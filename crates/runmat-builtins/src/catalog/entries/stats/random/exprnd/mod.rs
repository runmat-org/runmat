use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, IdentityInferenceRule,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
mod inference;
use documentation::EXPRND_DOCUMENTATION;

pub const EXPRND_INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exprnd-integer-mean",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exprnd with a typed-integer mean is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExprndIntegerMeanExtension"),
};

pub const EXPRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exprnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exprnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExprndIntegerSizeExtension"),
};

pub const EXPRND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [EXPRND_INTEGER_MU_EXTENSION, EXPRND_INTEGER_SIZE_EXTENSION];

const EXPRND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "mu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer means are gated and enter the floating exponential computation domain only after positivity validation.",
    },
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer size controls are gated and parsed exactly before allocation.",
    },
];

pub const EXPRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "r = exprnd(mu, sz1, sz2, ...)",
        inputs: &EXPRND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer means select double output. Integer sizes preserve the mean's single/double output class. Mean and size arguments are separate RunMat-only extensions; resident mean fallback restores output only to the exact source owner when that owner physically supports the required output precision.",
    }];

const EXPRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from exponential distribution.",
}];

const EXPRND_INPUTS_MU: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exponential mean parameter (must be > 0).",
}];

const EXPRND_INPUTS_MU_SZ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exponential mean parameter (must be > 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

const EXPRND_INPUTS_MU_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exponential mean parameter (must be > 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

const EXPRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = exprnd(mu)",
        inputs: &EXPRND_INPUTS_MU,
        outputs: &EXPRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = exprnd(mu, sz)",
        inputs: &EXPRND_INPUTS_MU_SZ,
        outputs: &EXPRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = exprnd(mu, sz1, sz2, ...)",
        inputs: &EXPRND_INPUTS_MU_DIMS,
        outputs: &EXPRND_OUTPUT_R,
    },
];

pub const EXPRND_ERROR_MU_MUST_BE_POSITIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPRND.MU_MUST_BE_POSITIVE",
    identifier: Some("RunMat:exprnd:MuMustBePositive"),
    when: "mu is zero or negative.",
    message: "exprnd: mu must be greater than zero",
};

pub const EXPRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:exprnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "exprnd: invalid argument",
};

pub const EXPRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXPRND.INTERNAL",
    identifier: Some("RunMat:exprnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "exprnd: internal operation failed",
};

const EXPRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    EXPRND_ERROR_MU_MUST_BE_POSITIVE,
    EXPRND_ERROR_INVALID_ARGUMENT,
    EXPRND_ERROR_INTERNAL,
];

pub const EXPRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXPRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXPRND_ERRORS,
};

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 3] = [
    EffectKind::Randomness,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
];
pub const EXPRND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "exprnd" },
    category: "stats/random",
    documentation: EXPRND_DOCUMENTATION,
    descriptor: &EXPRND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
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
    extensions: &EXPRND_EXTENSIONS,
    integer_capabilities: &EXPRND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&EXPRND_CATALOG_ENTRY];
