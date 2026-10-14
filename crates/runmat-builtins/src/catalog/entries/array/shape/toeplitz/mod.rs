mod documentation;
mod inference;
use crate::*;
pub const TOEPLITZ_COMPLEX_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "toeplitz-complex-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "toeplitz accepts native complex-integer vectors",
        error_identifier: Some("RunMat:compatibility:ToeplitzComplexIntegerInputExtension"),
    };
const TOEPLITZ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TOEPLITZ_COMPLEX_INTEGER_INPUT_EXTENSION];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Toeplitz matrix with constant diagonals.",
}];

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First row for the symmetric or Hermitian Toeplitz matrix.",
}];

const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "c",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First column.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First row.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = toeplitz(r)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = toeplitz(c, r)",
        inputs: &INPUTS_TWO,
        outputs: &OUTPUT,
    },
];

const INTEGER_VECTOR_INPUT: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "c",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The first column retains its native integer class and exact values.",
    },
    BuiltinIntegerInputCapability {
        name: "r",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes:
            "The first row must use the same native integer class as c and retains exact values.",
    },
];
const COMPLEX_INTEGER_VECTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "c and/or r",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "RunMat can structurally copy paired native integer components; the public integer domain does not support complex-integer operations.",
    }];
pub const TOEPLITZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "T = toeplitz(integer_c [, integer_r])",
        inputs: &INTEGER_VECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Toeplitz placement copies exact native values without arithmetic. Two-input integer calls require one common class; resident output returns through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "T = toeplitz(complex_integer_c [, complex_integer_r])",
        inputs: &COMPLEX_INTEGER_VECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only same-class paired integer-component construction preserves exact real and imaginary storage.",
    },
];

pub const TOEPLITZ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOEPLITZ.INVALID_INPUT",
    identifier: Some("RunMat:toeplitz:InvalidInput"),
    when: "Inputs are not vector-valued real or complex numeric arrays.",
    message: "toeplitz: invalid input",
};

pub const TOEPLITZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOEPLITZ.INTERNAL",
    identifier: Some("RunMat:toeplitz:Internal"),
    when: "Output allocation or gpuArray gather fails.",
    message: "toeplitz: internal error",
};

pub const TOEPLITZ_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOEPLITZ.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:toeplitz:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "toeplitz: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    TOEPLITZ_ERROR_INVALID_INPUT,
    TOEPLITZ_ERROR_INTERNAL,
    TOEPLITZ_ERROR_TOO_MANY_OUTPUTS,
];

pub const TOEPLITZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TOEPLITZ_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "toeplitz" },
    category: "array/shape",
    documentation: documentation::DOCUMENTATION,
    descriptor: &TOEPLITZ_DESCRIPTOR,
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
        residency: BuiltinResidencyPolicy::PreserveInputs,
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
    extensions: &TOEPLITZ_EXTENSIONS,
    integer_capabilities: &TOEPLITZ_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TOEPLITZ_CATALOG_ENTRY];
