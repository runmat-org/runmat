use super::*;

const OUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dataset contents.",
}];
const OUT_INFO: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 metadata structure.",
}];
const OUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Textual HDF5 metadata display.",
}];
const OUT_NONE: [BuiltinParamDescriptor; 0] = [];

const IN_FILE_DATASET: [BuiltinParamDescriptor; 2] = [
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
const IN_FILE_DATASET_SELECTION: [BuiltinParamDescriptor; 5] = [
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
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based starting subscript for each HDF5 dimension.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
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
const IN_FILE_DATASET_VALUE: [BuiltinParamDescriptor; 3] = [
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
const IN_FILE_DATASET_VALUE_SELECTION: [BuiltinParamDescriptor; 6] = [
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
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based starting subscript for each HDF5 dimension.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
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
const IN_FILE_LOCATION_ATTRIBUTE_VALUE: [BuiltinParamDescriptor; 4] = [
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
const IN_FILE_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "HDF5 file name.",
}];
const IN_FILE_OPTIONAL_LOCATION: [BuiltinParamDescriptor; 2] = [
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

const H5READ_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "data = h5read(filename, dataset)",
        inputs: &IN_FILE_DATASET,
        outputs: &OUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = h5read(filename, dataset, start, count, stride)",
        inputs: &IN_FILE_DATASET_SELECTION,
        outputs: &OUT_DATA,
    },
];
const H5WRITE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h5write(filename, dataset, data)",
        inputs: &IN_FILE_DATASET_VALUE,
        outputs: &OUT_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "h5write(filename, dataset, data, start, count, stride)",
        inputs: &IN_FILE_DATASET_VALUE_SELECTION,
        outputs: &OUT_NONE,
    },
];
const H5WRITEATT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "h5writeatt(filename, location, attribute, data)",
    inputs: &IN_FILE_LOCATION_ATTRIBUTE_VALUE,
    outputs: &OUT_NONE,
}];
const H5INFO_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "info = h5info(filename)",
        inputs: &IN_FILE_ONLY,
        outputs: &OUT_INFO,
    },
    BuiltinSignatureDescriptor {
        label: "info = h5info(filename, location)",
        inputs: &IN_FILE_OPTIONAL_LOCATION,
        outputs: &OUT_INFO,
    },
];
const H5DISP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename)",
        inputs: &IN_FILE_ONLY,
        outputs: &OUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = h5disp(filename, location)",
        inputs: &IN_FILE_OPTIONAL_LOCATION,
        outputs: &OUT_TEXT,
    },
];

pub(super) const ERR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.ARGUMENT",
    identifier: Some("RunMat:hdf5:InvalidArgument"),
    when: "Filename, object paths, selection vectors, or values are malformed.",
    message: "hdf5: invalid argument",
};
pub(super) const ERR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.IO",
    identifier: Some("RunMat:hdf5:Io"),
    when: "The HDF5 file cannot be opened, read, written, or traversed.",
    message: "hdf5: file I/O failed",
};
pub(super) const ERR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedType"),
    when: "The HDF5 type or MATLAB value is outside the supported dense numeric, logical, or string subset.",
    message: "hdf5: unsupported type",
};
pub(super) const ERR_WASM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HDF5.WASM_UNSUPPORTED",
    identifier: Some("RunMat:hdf5:UnsupportedTarget"),
    when: "Native HDF5 library support is unavailable on wasm targets.",
    message: "hdf5: native HDF5 is unavailable on this target",
};
pub(super) const ERRORS: [BuiltinErrorDescriptor; 4] =
    [ERR_ARGUMENT, ERR_IO, ERR_UNSUPPORTED, ERR_WASM];

macro_rules! descriptor {
    ($name:ident, $sigs:expr, $outputs:expr) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$sigs,
            output_mode: $outputs,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

descriptor!(
    H5READ_DESCRIPTOR,
    H5READ_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5WRITE_DESCRIPTOR,
    H5WRITE_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5WRITEATT_DESCRIPTOR,
    H5WRITEATT_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5INFO_DESCRIPTOR,
    H5INFO_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    H5DISP_DESCRIPTOR,
    H5DISP_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    HDF5READ_DESCRIPTOR,
    H5READ_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    HDF5WRITE_DESCRIPTOR,
    H5WRITE_SIGNATURES,
    BuiltinOutputMode::Fixed
);
descriptor!(
    HDF5INFO_DESCRIPTOR,
    H5INFO_SIGNATURES,
    BuiltinOutputMode::Fixed
);
