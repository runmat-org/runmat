mod documentation;

use crate::*;
use documentation::NUMEL_DOCUMENTATION;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact element count as a host double scalar.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value whose outer array elements are counted.",
}];
const SELECTED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value whose outer array elements are counted.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "RunMat-only positive integer dimension selectors.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "n = numel(A)",
        inputs: &INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "n = numel(A, dim1, dim2, ...)",
        inputs: &SELECTED_INPUTS,
        outputs: &OUTPUT,
    },
];

pub const NUMEL_DIMENSIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "numel-dimension-selectors",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Selected-dimension numel forms are a RunMat extension; the compatible public form accepts only the value.",
    error_identifier: Some("RunMat:compatibility:NumelDimensionSelectorsExtension"),
};
pub const NUMEL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NUMEL_DIMENSIONS_EXTENSION];

macro_rules! numel_error {
    ($name:ident, $code:literal, $suffix:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some(concat!("RunMat:numel:", $suffix)),
            when: $when,
            message: $message,
        };
    };
}
numel_error!(
    NUMEL_ERROR_DIM_ARG_TYPE,
    "RM.NUMEL.DIM_ARG_TYPE",
    "InvalidDimensionType",
    "A selector is not a numeric scalar or vector.",
    "numel: dimension selectors must be numeric scalars or one numeric vector"
);
numel_error!(
    NUMEL_ERROR_DIM_VECTOR_SHAPE,
    "RM.NUMEL.DIM_VECTOR_SHAPE",
    "InvalidDimensionVector",
    "The single vector selector is not vector-shaped.",
    "numel: dimension selector must be a vector"
);
numel_error!(
    NUMEL_ERROR_DIM_SCALAR_LIST,
    "RM.NUMEL.DIM_SCALAR_LIST",
    "InvalidDimensionList",
    "A vector occurs within a variadic scalar selector list.",
    "numel: separate dimension arguments must be scalars"
);
numel_error!(
    NUMEL_ERROR_DIM_EMPTY,
    "RM.NUMEL.DIM_EMPTY",
    "EmptyDimensionList",
    "No selected dimensions remain after parsing.",
    "numel: dimension list must contain at least one element"
);
numel_error!(
    NUMEL_ERROR_DIM_NON_FINITE,
    "RM.NUMEL.DIM_NON_FINITE",
    "NonFiniteDimension",
    "A floating selector is not finite.",
    "numel: dimension must be finite"
);
numel_error!(
    NUMEL_ERROR_DIM_NON_INTEGER,
    "RM.NUMEL.DIM_NON_INTEGER",
    "NonIntegerDimension",
    "A floating selector is fractional.",
    "numel: dimension must be an integer"
);
numel_error!(
    NUMEL_ERROR_DIM_LT_ONE,
    "RM.NUMEL.DIM_LT_ONE",
    "NonPositiveDimension",
    "A selector is less than one.",
    "numel: dimension must be at least one"
);
numel_error!(
    NUMEL_ERROR_DIM_RANGE,
    "RM.NUMEL.DIM_RANGE",
    "DimensionOutOfRange",
    "A selector is outside the unsigned structural range.",
    "numel: dimension is outside the supported structural range"
);
numel_error!(
    NUMEL_ERROR_RESULT_NOT_EXACT,
    "RM.NUMEL.RESULT_NOT_EXACT",
    "ResultNotExactDouble",
    "The product over dimensions overflows or is not exactly representable as double.",
    "numel: result exceeds exact double range"
);
numel_error!(
    NUMEL_ERROR_INTERNAL,
    "RM.NUMEL.INTERNAL",
    "InternalError",
    "Shape metadata is invalid.",
    "numel: invalid shape metadata"
);
numel_error!(
    NUMEL_ERROR_TOO_MANY_OUTPUTS,
    "RM.NUMEL.TOO_MANY_OUTPUTS",
    "TooManyOutputs",
    "More than one output is requested.",
    "numel: too many output arguments"
);
const ERRORS: &[BuiltinErrorDescriptor] = &[
    NUMEL_ERROR_DIM_ARG_TYPE,
    NUMEL_ERROR_DIM_VECTOR_SHAPE,
    NUMEL_ERROR_DIM_SCALAR_LIST,
    NUMEL_ERROR_DIM_EMPTY,
    NUMEL_ERROR_DIM_NON_FINITE,
    NUMEL_ERROR_DIM_NON_INTEGER,
    NUMEL_ERROR_DIM_LT_ONE,
    NUMEL_ERROR_DIM_RANGE,
    NUMEL_ERROR_RESULT_NOT_EXACT,
    NUMEL_ERROR_INTERNAL,
    NUMEL_ERROR_TOO_MANY_OUTPUTS,
];
pub const NUMEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Only shape metadata is inspected.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Extension selectors are decoded exactly from native integer storage.",
    },
];
pub const NUMEL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "n = numel(integer_A, integer_dimensions)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::HostAndGpu,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "The documented form counts the complete shape; the RunMat extension multiplies selected extents. Both use checked structural arithmetic and an exact host double result.",
}];

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;
const EFFECTS: [runmat_types::EffectKind; 1] = [runmat_types::EffectKind::MayThrow];
pub const NUMEL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "numel" },
    category: "array/introspection",
    documentation: NUMEL_DOCUMENTATION,
    descriptor: &NUMEL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Introspection(
            ArrayIntrospectionInferenceRule::ShapeQuery(ShapeQuery::ElementCount),
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
    extensions: &NUMEL_EXTENSIONS,
    integer_capabilities: &NUMEL_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&NUMEL_CATALOG_ENTRY];
