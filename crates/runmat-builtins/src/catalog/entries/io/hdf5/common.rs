use crate::*;

pub(super) const OUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dataset contents.",
}];
pub(super) const OUT_INFO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 metadata structure.",
}];
pub(super) const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Textual HDF5 metadata display.",
}];
pub(super) const OUT_NONE: [BuiltinParamDescriptor; 0] = [];

pub(super) const IN_FILE_DATASET: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
];
pub(super) const IN_FILE_DATASET_SELECTION: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based starting subscript for each HDF5 dimension.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of elements to read in each dimension.",
    },
    BuiltinParamDescriptor {
        name: "stride",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Stride for each HDF5 dimension.",
    },
];
pub(super) const IN_FILE_DATASET_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Data to write.",
    },
];
pub(super) const IN_FILE_DATASET_VALUE_SELECTION: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "dataset",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dataset path in the file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Data to write.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based starting subscript for each HDF5 dimension.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of elements to write in each dimension.",
    },
    BuiltinParamDescriptor {
        name: "stride",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Stride for each HDF5 dimension.",
    },
];
pub(super) const IN_FILE_LOCATION_ATTRIBUTE_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "location",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Group or dataset path.",
    },
    BuiltinParamDescriptor {
        name: "attribute",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute name.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Attribute value to write.",
    },
];
pub(super) const IN_FILE_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 file name.",
}];
pub(super) const IN_FILE_OPTIONAL_LOCATION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "HDF5 file name.",
    },
    BuiltinParamDescriptor {
        name: "location",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("/"),
        description: "Group or dataset path.",
    },
];

pub const ERR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.ARGUMENT",
    identifier: Some("RunMat:hdf5:InvalidArgument"),
    when: "Filename, object paths, selection vectors, or values are malformed.",
    message: "hdf5: invalid argument",
};
pub const ERR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.IO",
    identifier: Some("RunMat:hdf5:Io"),
    when: "The HDF5 file cannot be opened, read, written, or traversed.",
    message: "hdf5: file I/O failed",
};
pub const ERR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedType"),
    when: "The HDF5 type or MATLAB value is outside the supported dense numeric, logical, or string subset.",
    message: "hdf5: unsupported type",
};
pub const ERR_WASM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.WASM_UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedTarget"),
    when: "Native HDF5 library support is unavailable on wasm targets.",
    message: "hdf5: native HDF5 is unavailable on this target",
};
pub const ERR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.PROVIDER_UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedProvider"),
    when: "The captured filesystem provider cannot issue a native path for the HDF5 library.",
    message: "hdf5: active filesystem provider has no native HDF5 path route",
};
pub(super) const ERRORS: [BuiltinErrorDescriptor; 5] = [
    ERR_ARGUMENT,
    ERR_IO,
    ERR_UNSUPPORTED,
    ERR_WASM,
    ERR_PROVIDER,
];

pub(super) const INTEGER_SELECTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "start/count/stride",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Exact typed integer selection entries are checked as one-based positive platform indices; floating entries must be finite integral values.",
    }];

pub(super) const INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dense typed integer data retains its exact storage width and signedness during HDF5 serialization.",
    }];
