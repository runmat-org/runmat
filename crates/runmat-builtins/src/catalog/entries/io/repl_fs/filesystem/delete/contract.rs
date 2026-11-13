use crate::*;

pub const DELETE_INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filename/pattern string, string array, char matrix, cell array of path strings, or handle input.",
}];
pub const DELETE_INPUTS_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First filename/pattern/handle input.",
    },
    BuiltinParamDescriptor {
        name: "filenameN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional filename/pattern/handle inputs.",
    },
];
pub const DELETE_INPUTS_SYMLINK: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filename, filename collection, or wildcard pattern.",
    },
    BuiltinParamDescriptor {
        name: "ResolveSymbolicLinks",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the symbolic-link resolution option.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("false"),
        description: "Scalar logical or numeric zero/one selecting link-target deletion.",
    },
];
pub const DELETE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "delete(filename)",
        inputs: &DELETE_INPUTS_ONE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "delete(filename1, filename2, ...)",
        inputs: &DELETE_INPUTS_VARIADIC,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "delete(___, ResolveSymbolicLinks=tf)",
        inputs: &DELETE_INPUTS_SYMLINK,
        outputs: &[],
    },
];

pub const DELETE_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ResolveSymbolicLinks",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented numeric/logical flag accepts only exact scalar zero or one; MATLAB's numeric category includes every built-in integer class.",
    }];

pub const DELETE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "delete(filename, \"ResolveSymbolicLinks\", integer_tf)",
        inputs: &DELETE_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The flag is read exactly from authoritative integer storage before any filesystem access; deletion has no MATLAB-facing output.",
    }];
pub const DELETE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.INVALID_INPUT",
    identifier: Some("RunMat:delete:InvalidInput"),
    when: "Input arguments are missing or contain unsupported filename value types.",
    message: "delete: invalid input",
};
pub const DELETE_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.INVALID_HANDLE",
    identifier: Some("RunMat:delete:InvalidHandle"),
    when: "Handle deletion inputs are mixed with filename inputs or contain unsupported handle values.",
    message: "delete: invalid handle input",
};
pub const DELETE_ERROR_EMPTY_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.EMPTY_FILENAME",
    identifier: Some("RunMat:delete:EmptyFilename"),
    when: "A filename input is empty after trimming.",
    message: "delete: filename cannot be empty",
};
pub const DELETE_ERROR_INVALID_PATTERN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.INVALID_PATTERN",
    identifier: Some("RunMat:delete:InvalidPattern"),
    when: "Wildcard pattern parsing or structure validation fails.",
    message: "delete: invalid wildcard pattern",
};
pub const DELETE_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.FILE_NOT_FOUND",
    identifier: Some("RunMat:DELETE:FileNotFound"),
    when: "Target path does not exist or pattern matches no files.",
    message: "delete: file not found",
};
pub const DELETE_ERROR_IS_DIRECTORY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.IS_DIRECTORY",
    identifier: Some("RunMat:delete:Directories"),
    when: "Target path is a directory instead of a file.",
    message: "delete: cannot delete directories",
};
pub const DELETE_ERROR_OS_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.OS_ERROR",
    identifier: Some("RunMat:DELETE:PermissionDenied"),
    when: "Underlying filesystem operation fails while deleting files.",
    message: "delete: filesystem deletion failed",
};
pub const DELETE_ERROR_RUNTIME_CONTEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.RUNTIME_CONTEXT",
    identifier: Some("RunMat:parallel:RuntimeContextUnavailable"),
    when: "Pool deletion is requested without an active runtime context.",
    message: "delete: no active runtime context",
};
pub const DELETE_ERRORS: [BuiltinErrorDescriptor; 8] = [
    DELETE_ERROR_INVALID_INPUT,
    DELETE_ERROR_INVALID_HANDLE,
    DELETE_ERROR_EMPTY_FILENAME,
    DELETE_ERROR_INVALID_PATTERN,
    DELETE_ERROR_FILE_NOT_FOUND,
    DELETE_ERROR_IS_DIRECTORY,
    DELETE_ERROR_OS_ERROR,
    DELETE_ERROR_RUNTIME_CONTEXT,
];
pub const DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DELETE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DELETE_ERRORS,
};
