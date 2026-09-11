mod documentation;

use crate::*;
use documentation::SIZE_DOCUMENTATION;

const OUTPUT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double row vector containing the queried dimension extents.",
}];
const OUTPUT_SCALARS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Host double scalar for each requested output.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value whose MATLAB-visible dimensions are inspected.",
}];
const SELECTED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value whose MATLAB-visible dimensions are inspected.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Positive integer scalar selectors, or one vector or empty selector.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "sz = size(A)",
        inputs: &INPUT,
        outputs: &OUTPUT_VECTOR,
    },
    BuiltinSignatureDescriptor {
        label: "sz = size(A, dim)",
        inputs: &SELECTED_INPUTS,
        outputs: &OUTPUT_VECTOR,
    },
    BuiltinSignatureDescriptor {
        label: "sz = size(A, dim1, dim2, ...)",
        inputs: &SELECTED_INPUTS,
        outputs: &OUTPUT_VECTOR,
    },
    BuiltinSignatureDescriptor {
        label: "[sz1, ..., szN] = size(A)",
        inputs: &INPUT,
        outputs: &OUTPUT_SCALARS,
    },
    BuiltinSignatureDescriptor {
        label: "[sz1, ..., szN] = size(A, dim1, ..., dimN)",
        inputs: &SELECTED_INPUTS,
        outputs: &OUTPUT_SCALARS,
    },
];

macro_rules! size_error {
    ($name:ident, $code:literal, $suffix:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some(concat!("RunMat:size:", $suffix)),
            when: $when,
            message: $message,
        };
    };
}

size_error!(
    SIZE_ERROR_DIM_ARG_TYPE,
    "RM.SIZE.DIM_ARG_TYPE",
    "InvalidDimensionType",
    "A selector is not a numeric scalar or vector.",
    "size: dimension selectors must be numeric scalars or one numeric vector"
);
size_error!(
    SIZE_ERROR_DIM_VECTOR_SHAPE,
    "RM.SIZE.DIM_VECTOR_SHAPE",
    "InvalidDimensionVector",
    "The single vector selector is not vector-shaped.",
    "size: dimension selector must be a vector"
);
size_error!(
    SIZE_ERROR_DIM_SCALAR_LIST,
    "RM.SIZE.DIM_SCALAR_LIST",
    "InvalidDimensionList",
    "A vector occurs within a variadic scalar selector list.",
    "size: separate dimension arguments must be scalars"
);
size_error!(
    SIZE_ERROR_DIM_NON_FINITE,
    "RM.SIZE.DIM_NON_FINITE",
    "NonFiniteDimension",
    "A floating selector is not finite.",
    "size: dimension must be finite"
);
size_error!(
    SIZE_ERROR_DIM_NON_INTEGER,
    "RM.SIZE.DIM_NON_INTEGER",
    "NonIntegerDimension",
    "A floating selector is fractional.",
    "size: dimension must be an integer"
);
size_error!(
    SIZE_ERROR_DIM_LT_ONE,
    "RM.SIZE.DIM_LT_ONE",
    "NonPositiveDimension",
    "A selector is less than one.",
    "size: dimension must be at least one"
);
size_error!(
    SIZE_ERROR_DIM_RANGE,
    "RM.SIZE.DIM_RANGE",
    "DimensionOutOfRange",
    "A selector is outside the unsigned structural range.",
    "size: dimension is outside the supported structural range"
);
size_error!(
    SIZE_ERROR_OUTPUT_COUNT,
    "RM.SIZE.OUTPUT_COUNT",
    "OutputCountMismatch",
    "The number of outputs does not equal the number of explicit queried dimensions.",
    "size: output count must match queried dimension count"
);
size_error!(
    SIZE_ERROR_RESULT_NOT_EXACT,
    "RM.SIZE.RESULT_NOT_EXACT",
    "ResultNotExactDouble",
    "A result extent or collapsed product is not exactly representable as double.",
    "size: result exceeds exact double range"
);
size_error!(
    SIZE_ERROR_INTERNAL,
    "RM.SIZE.INTERNAL",
    "InternalError",
    "Shape metadata is invalid or output construction fails.",
    "size: invalid shape metadata"
);

const ERRORS: &[BuiltinErrorDescriptor] = &[
    SIZE_ERROR_DIM_ARG_TYPE,
    SIZE_ERROR_DIM_VECTOR_SHAPE,
    SIZE_ERROR_DIM_SCALAR_LIST,
    SIZE_ERROR_DIM_NON_FINITE,
    SIZE_ERROR_DIM_NON_INTEGER,
    SIZE_ERROR_DIM_LT_ONE,
    SIZE_ERROR_DIM_RANGE,
    SIZE_ERROR_OUTPUT_COUNT,
    SIZE_ERROR_RESULT_NOT_EXACT,
    SIZE_ERROR_INTERNAL,
];
pub const SIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const INTEGER_ARRAY_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Only shape metadata is inspected; the integer payload remains untouched.",
}];
const INTEGER_DIMENSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim or dimensions",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar, vector, and separate selectors are decoded exactly from native integer storage.",
    }];
pub const SIZE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "size(integer_A)",
        inputs: &INTEGER_ARRAY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Dimensions and requested-output collapse use checked structural arithmetic and exact host double outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "size(A, integer_dimensions)",
        inputs: &INTEGER_DIMENSION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Dimension values are range checked before structural lookup; an empty selector returns a 1-by-0 double row.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 1] = [runmat_types::EffectKind::MayThrow];
pub const SIZE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "size" },
    category: "array/introspection",
    documentation: SIZE_DOCUMENTATION,
    descriptor: &SIZE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Introspection(
            ArrayIntrospectionInferenceRule::ShapeQuery(ShapeQuery::Size),
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
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::InspectHandles,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &SIZE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SIZE_CATALOG_ENTRY];
