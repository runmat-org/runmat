use crate::{
    AccelerationInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior,
    BuiltinBindingAvailability, BuiltinBindingDeclaration, BuiltinBindingIdentity,
    BuiltinCatalogEntry, BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation,
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy,
    BuiltinInferenceRule, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, BuiltinSignatureDescriptor,
    ALL_INTEGER_CLASSES,
};
use runmat_types::{CapabilityRequirement, EffectKind};

pub const GPUARRAY_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-size-arguments",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gpuArray size arguments are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArraySizeExtension"),
};
pub const GPUARRAY_DTYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-dtype-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gpuArray dtype selectors are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArrayDtypeExtension"),
};
pub const GPUARRAY_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the gpuArray \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArrayLikeExtension"),
};
pub const GPUARRAY_TEXT_UPLOAD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-text-upload",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "uploading character vectors or string scalars with gpuArray is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArrayTextUploadExtension"),
};
pub const GPUARRAY_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    GPUARRAY_SIZE_EXTENSION,
    GPUARRAY_DTYPE_EXTENSION,
    GPUARRAY_LIKE_EXTENSION,
    GPUARRAY_TEXT_UPLOAD_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes upload as exact same-class real or paired-complex gpuArray storage with the original shape.",
}];
const INTEGER_DTYPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat-only dtype selectors may explicitly convert X to any supported integer gpuArray class.",
    }];
pub const GPUARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gpuArray(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The transfer preserves exact values, class, shape, and supported complexity. An existing gpuArray input is returned unchanged and remains valid.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gpuArray(X, integer_dtype)",
        inputs: &INTEGER_DTYPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only conversion uses the requested native integer class and never consumes or invalidates a gpuArray input.",
    },
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "GPU-resident handle containing uploaded/converted data.",
}];
const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to upload or recast on GPU.",
};
const INPUTS_BASE: [BuiltinParamDescriptor; 1] = [INPUT];
const INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
    INPUT,
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Reshape dimensions (scalar dims or a single size vector tensor).",
    },
];
const INPUTS_DTYPE: [BuiltinParamDescriptor; 2] = [
    INPUT,
    BuiltinParamDescriptor {
        name: "dtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"double\""),
        description: "Class tag such as `single`, `int32`, `uint8`, `logical`, or `double`.",
    },
];
const INPUTS_LIKE: [BuiltinParamDescriptor; 3] = [
    INPUT,
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal keyword `\"like\"`.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value whose class drives output conversion.",
    },
];
const INPUTS_DIMS_OPTIONS: [BuiltinParamDescriptor; 3] = [
    INPUT,
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Reshape dimensions (scalar dims or a single size vector tensor).",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Class tags and/or `\"like\", prototype` qualifiers.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X)",
        inputs: &INPUTS_BASE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, dim, ...)",
        inputs: &INPUTS_DIMS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, dtype)",
        inputs: &INPUTS_DTYPE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, \"like\", prototype)",
        inputs: &INPUTS_LIKE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, dim, ..., option, ...)",
        inputs: &INPUTS_DIMS_OPTIONS,
        outputs: &OUTPUT,
    },
];

macro_rules! error {
    ($name:ident, $code:literal, $identifier:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            when: $when,
            message: $message,
        };
    };
}
error!(
    GPUARRAY_ERROR_NO_PROVIDER,
    "RM.GPUARRAY.NO_PROVIDER",
    "RunMat:gpuArray:NoProvider",
    "No acceleration provider is registered for host/device transfers.",
    "gpuArray: no acceleration provider registered"
);
error!(
    GPUARRAY_ERROR_OPTION_ARGUMENT,
    "RM.GPUARRAY.OPTION_ARGUMENT",
    "RunMat:gpuArray:OptionArgument",
    "Option tail contains non-text values where class tags/keywords are expected.",
    "gpuArray: invalid option argument"
);
error!(
    GPUARRAY_ERROR_LIKE_MISSING,
    "RM.GPUARRAY.LIKE_MISSING",
    "RunMat:gpuArray:LikeMissingPrototype",
    "Keyword `like` is supplied without a following prototype value.",
    "gpuArray: expected a prototype value after 'like'"
);
error!(
    GPUARRAY_ERROR_LIKE_DUPLICATE,
    "RM.GPUARRAY.LIKE_DUPLICATE",
    "RunMat:gpuArray:LikeDuplicate",
    "Keyword `like` appears more than once.",
    "gpuArray: duplicate 'like' qualifier"
);
error!(
    GPUARRAY_ERROR_CODISTRIBUTED_UNSUPPORTED,
    "RM.GPUARRAY.CODISTRIBUTED_UNSUPPORTED",
    "RunMat:gpuArray:CodistributedUnsupported",
    "Distributed/codistributed qualifiers are requested.",
    "gpuArray: codistributed arrays are not supported yet"
);
error!(
    GPUARRAY_ERROR_CONFLICTING_TYPE,
    "RM.GPUARRAY.CONFLICTING_TYPE",
    "RunMat:gpuArray:ConflictingTypeQualifiers",
    "Multiple incompatible class qualifiers are supplied.",
    "gpuArray: conflicting type qualifiers supplied"
);
error!(
    GPUARRAY_ERROR_UNKNOWN_OPTION,
    "RM.GPUARRAY.UNKNOWN_OPTION",
    "RunMat:gpuArray:UnknownOption",
    "Text option is not a recognized class/keyword token.",
    "gpuArray: unrecognised option"
);
error!(
    GPUARRAY_ERROR_SIZE_ARGUMENT,
    "RM.GPUARRAY.SIZE_ARGUMENT",
    "RunMat:gpuArray:InvalidSizeArgument",
    "Size arguments are malformed (not finite integers, negative, or invalid combinations).",
    "gpuArray: invalid size argument"
);
error!(
    GPUARRAY_ERROR_LIKE_PROTOTYPE,
    "RM.GPUARRAY.LIKE_PROTOTYPE",
    "RunMat:gpuArray:InvalidLikePrototype",
    "`like` prototype is unsupported for type inference.",
    "gpuArray: invalid 'like' prototype"
);
error!(
    GPUARRAY_ERROR_INPUT_TYPE,
    "RM.GPUARRAY.INPUT_TYPE",
    "RunMat:gpuArray:UnsupportedInputType",
    "Input value type cannot be uploaded/coerced to supported gpuArray storage.",
    "gpuArray: unsupported input type"
);
error!(
    GPUARRAY_ERROR_TYPED_INTEGER,
    "RM.GPUARRAY.TYPED_INTEGER",
    "RunMat:gpuArray:TypedIntegerUnsupported",
    "A native integer value or integer GPU class is requested without matching provider storage.",
    "gpuArray: native integer storage is not supported by the active acceleration provider"
);
error!(
    GPUARRAY_ERROR_CONVERSION,
    "RM.GPUARRAY.CONVERSION",
    "RunMat:gpuArray:ConversionFailed",
    "Requested dtype conversion cannot be performed (for example NaN->logical).",
    "gpuArray: conversion failed"
);
error!(
    GPUARRAY_ERROR_RESHAPE,
    "RM.GPUARRAY.RESHAPE",
    "RunMat:gpuArray:ReshapeMismatch",
    "Requested shape does not preserve the element count.",
    "gpuArray: cannot reshape gpuArray into requested size"
);
error!(
    GPUARRAY_ERROR_PROVIDER_IO,
    "RM.GPUARRAY.PROVIDER_IO",
    "RunMat:gpuArray:ProviderIO",
    "Provider upload/download interaction fails.",
    "gpuArray: provider I/O failed"
);
error!(
    GPUARRAY_ERROR_INTERNAL,
    "RM.GPUARRAY.INTERNAL",
    "RunMat:gpuArray:InternalError",
    "Internal tensor/container conversion fails.",
    "gpuArray: internal error"
);

const ERRORS: [BuiltinErrorDescriptor; 15] = [
    GPUARRAY_ERROR_NO_PROVIDER,
    GPUARRAY_ERROR_OPTION_ARGUMENT,
    GPUARRAY_ERROR_LIKE_MISSING,
    GPUARRAY_ERROR_LIKE_DUPLICATE,
    GPUARRAY_ERROR_CODISTRIBUTED_UNSUPPORTED,
    GPUARRAY_ERROR_CONFLICTING_TYPE,
    GPUARRAY_ERROR_UNKNOWN_OPTION,
    GPUARRAY_ERROR_SIZE_ARGUMENT,
    GPUARRAY_ERROR_LIKE_PROTOTYPE,
    GPUARRAY_ERROR_INPUT_TYPE,
    GPUARRAY_ERROR_TYPED_INTEGER,
    GPUARRAY_ERROR_CONVERSION,
    GPUARRAY_ERROR_RESHAPE,
    GPUARRAY_ERROR_PROVIDER_IO,
    GPUARRAY_ERROR_INTERNAL,
];
pub const GPUARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "gpuArray" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const CAPABILITIES: [CapabilityRequirement; 1] = [CapabilityRequirement::Accelerator];
pub const GPUARRAY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "gpuArray" },
    category: "acceleration/gpu",
    documentation: BuiltinDocumentation {
        summary: "Move data to the GPU as gpuArray values.",
        keywords: &["gpuArray", "gpu", "accelerate", "upload", "dtype", "like"],
        related: &["gather"],
        introduced: None,
        status: None,
        examples: &["G = gpuArray([1 2 3], 'single');"],
    },
    descriptor: &GPUARRAY_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Acceleration(AccelerationInferenceRule::GpuArray),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &CAPABILITIES,
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Required,
        residency: BuiltinResidencyPolicy::ProduceResident,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &GPUARRAY_EXTENSIONS,
    integer_capabilities: &GPUARRAY_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
