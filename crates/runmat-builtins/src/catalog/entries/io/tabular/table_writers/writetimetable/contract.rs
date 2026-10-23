use crate::*;
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "TT",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact timetable object.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonempty scalar text path on the selected filesystem provider.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Delimiter and WriteVariableNames name-value pairs.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bytes",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact double byte count; requesting this output requires RunMat extension mode.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "writetimetable(TT, filename, Name=Value)",
        inputs: &INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "bytes = writetimetable(TT, filename, Name=Value)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
];
pub const WRITETIMETABLE_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITETIMETABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:writetimetable:InvalidArgument"),
    when: "Input class, path, option, or output count is invalid.",
    message: "writetimetable: invalid argument",
};
pub const WRITETIMETABLE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITETIMETABLE.IO",
    identifier: Some("RunMat:writetimetable:IOError"),
    when: "The captured filesystem provider cannot stage, write, or publish the output.",
    message: "writetimetable: file write failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [WRITETIMETABLE_ERROR_ARGUMENT, WRITETIMETABLE_ERROR_IO];
pub const WRITETIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
