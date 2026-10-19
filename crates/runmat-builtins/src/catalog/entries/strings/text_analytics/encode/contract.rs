use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "counts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse word or n-gram count matrix, optionally wrapped in a scalar cell.",
}];
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "bag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "bagOfWords or bagOfNgrams model.",
    },
    BuiltinParamDescriptor {
        name: "documentsOrWords",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument or row word vector.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: DocumentsIn and ForceCellOutput.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "counts = encode(bag, documentsOrWords, Name, Value, ...)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ENCODE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENCODE.INVALID_INPUT",
    identifier: Some("RunMat:encode:InvalidInput"),
    when: "Inputs do not match a supported Text Analytics encode form.",
    message: "encode: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [ENCODE_ERROR_INVALID_INPUT];
pub const ENCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ENCODE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ENCODE_NUMERIC_FORCE_CELL_OUTPUT_EXTENSION,
    ENCODE_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION,
];
pub const ENCODE_NUMERIC_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "encode-numeric-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "encode with a numeric ForceCellOutput value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EncodeNumericForceCellOutputExtension"),
    };
pub const ENCODE_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "encode-resident-force-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "encode with a resident ForceCellOutput value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EncodeResidentForceCellOutputExtension"),
    };
const REJECTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "bag", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::Rejected, notes: "The model role requires a bagOfWords or bagOfNgrams object; integer values reject before provider access." },
    BuiltinIntegerInputCapability { name: "documentsOrWords", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::Rejected, notes: "The document role requires tokenizedDocument or text; integers are not converted to words." },
];
const FORCE_CELL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "ForceCellOutput", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "RunMat mode accepts exact scalar integer zero as false and every nonzero integer as true; MATLAB-compatible mode requires logical." }];
pub const ENCODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "counts = encode(bag, documentsOrWords)", inputs: &REJECTED_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "encode is object/text based. Its integer-valued counts intentionally cross the sparse-double output boundary, not integer sparse storage." },
    BuiltinIntegerCapabilityDescriptor { form: "counts = encode(___, 'ForceCellOutput', integer_value)", inputs: &FORCE_CELL_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "The compatibility-gated integer control selects sparse-double versus cell-of-sparse-double representation; it does not change count storage." },
];
