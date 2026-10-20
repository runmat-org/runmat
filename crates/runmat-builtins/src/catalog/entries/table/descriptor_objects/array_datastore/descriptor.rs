use crate::*;

pub const ARRAY_DATASTORE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "arraydatastore-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "arrayDatastore with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ArrayDatastoreGpuInputExtension"),
    };

pub const ARRAY_DATASTORE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY_DATASTORE_GPU_INPUT_EXTENSION];

const ARRAY_DATASTORE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented in-memory array input accepts matrices whose values use any of the eight real integer classes.",
    }];
const ARRAY_DATASTORE_READ_SIZE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ReadSize",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public property is explicitly double and typed-integer values are rejected.",
    }];
const ARRAY_DATASTORE_ITERATION_DIMENSION_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IterationDimension",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public property is explicitly double and typed-integer values are rejected.",
    }];
pub const ARRAY_DATASTORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "arrds = arrayDatastore(integer_A, Name,Value...)",
        inputs: &ARRAY_DATASTORE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The datastore object retains A in authoritative same-class storage. Interactive resident input is a mode-gated RunMat extension that gathers before host datastore construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "arrds = arrayDatastore(A, \"ReadSize\", typed_integer)",
        inputs: &ARRAY_DATASTORE_READ_SIZE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "ReadSize is documented as a double-only value; all typed-integer scalar and tensor classes reject.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "arrds = arrayDatastore(A, \"IterationDimension\", typed_integer)",
        inputs: &ARRAY_DATASTORE_ITERATION_DIMENSION_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "IterationDimension is documented as a double-only value; all typed-integer scalar and tensor classes reject.",
    },
];

pub const ARRAY_DATASTORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "ds = arrayDatastore(A, Name, Value...)",
            inputs: &[
                BuiltinParamDescriptor { name: "A", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "In-memory data retained in Data, including exact native integer storage." },
                BuiltinParamDescriptor { name: "Name,Value", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic, default: None, description: "ReadSize (default 1), IterationDimension (default 1), OutputType (default 'cell'). Positive integer controls must be double; 'same' requires dimension 1." },
            ],
            outputs: &[BuiltinParamDescriptor { name: "descriptor", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Scalar host value descriptor; not a live iterator or UI handle." }],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};
