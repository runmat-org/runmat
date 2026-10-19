mod documentation;
mod inference;
use crate::*;
const MAGIC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "n-by-n magic square matrix.",
}];

const MAGIC_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, or character input whose first real element selects the square order; floating values are floored.",
}];

const MAGIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "A = magic(n)",
    inputs: &MAGIC_SIG_N_INPUTS,
    outputs: &MAGIC_OUTPUT,
}];

const MAGIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are accepted; the first element is read exactly as the square order.",
    }];

pub const MAGIC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = magic(integer_n)",
        inputs: &MAGIC_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The first real element selects the order and the result is a host double matrix; explicit gpuArray input is unsupported while automatic residency gathers transparently.",
    }];

pub const MAGIC_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MAGIC.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:magic:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "magic: too many output arguments",
};
const MAGIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    MAGIC_ERROR_TOO_MANY_OUTPUTS,
    BuiltinErrorDescriptor {
        code: "RM.MAGIC.ARG_COUNT",
        identifier: None,
        when: "The builtin is called with a number of arguments other than one.",
        message: "magic: requires exactly one input argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.MAGIC.EMPTY_INPUT",
        identifier: None,
        when: "The input has no numeric elements.",
        message: "magic: input must contain a numeric value",
    },
    BuiltinErrorDescriptor {
        code: "RM.MAGIC.NON_FINITE",
        identifier: None,
        when: "The order argument is non-finite.",
        message: "magic: dimension must be finite",
    },
    BuiltinErrorDescriptor {
        code: "RM.MAGIC.NEGATIVE",
        identifier: None,
        when: "The floored order argument is negative.",
        message: "magic: dimension must be non-negative",
    },
    BuiltinErrorDescriptor {
        code: "RM.MAGIC.EXPLICIT_GPU",
        identifier: None,
        when: "The order is supplied as an explicit gpuArray.",
        message: "magic: gpuArray input is not supported",
    },
];

pub const MAGIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MAGIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MAGIC_ERRORS,
};

pub const MAGIC_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "magic" },
    category: "array/creation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MAGIC_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &MAGIC_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MAGIC_CATALOG_ENTRY];
