use crate::*;

const IMWRITE_INPUTS_IMAGE_FILENAME: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grayscale, truecolor, or RGBA image data.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
];

const IMWRITE_INPUTS_INDEXED: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image data.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nx3 colormap.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
];

const IMWRITE_INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image data.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option value.",
    },
];

const IMWRITE_INPUTS_FORMAT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Image data.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Explicit supported output image format.",
    },
];

const IMWRITE_INPUTS_INDEXED_OPTIONS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image data.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nx3 colormap.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output image path.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option or explicit format followed by options.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option value.",
    },
];

const IMWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "imwrite(A, filename)",
        inputs: &IMWRITE_INPUTS_IMAGE_FILENAME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(A, filename, fmt)",
        inputs: &IMWRITE_INPUTS_FORMAT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(A, filename, name, value, ...)",
        inputs: &IMWRITE_INPUTS_OPTIONS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(X, map, filename)",
        inputs: &IMWRITE_INPUTS_INDEXED,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "imwrite(X, map, filename, ...)",
        inputs: &IMWRITE_INPUTS_INDEXED_OPTIONS,
        outputs: &[],
    },
];

pub const IMWRITE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_ARGUMENT",
    identifier: Some("RunMat:imwrite:InvalidArgument"),
    when: "Arguments do not match a supported imwrite form.",
    message: "imwrite: invalid argument",
};
pub const IMWRITE_ERROR_INVALID_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_FILENAME",
    identifier: Some("RunMat:imwrite:InvalidFilename"),
    when: "Filename is missing or empty.",
    message: "imwrite: invalid filename",
};
pub const IMWRITE_ERROR_INVALID_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_FORMAT",
    identifier: Some("RunMat:imwrite:InvalidFormat"),
    when: "Image format cannot be inferred or is unsupported.",
    message: "imwrite: invalid image format",
};
pub const IMWRITE_ERROR_INVALID_IMAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_IMAGE",
    identifier: Some("RunMat:imwrite:InvalidImage"),
    when: "Image data has unsupported type, shape, or values.",
    message: "imwrite: invalid image data",
};
pub const IMWRITE_ERROR_INVALID_COLORMAP: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_COLORMAP",
    identifier: Some("RunMat:imwrite:InvalidColormap"),
    when: "Indexed-image colormap is not an Nx3 numeric array.",
    message: "imwrite: invalid colormap",
};
pub const IMWRITE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.INVALID_OPTION",
    identifier: Some("RunMat:imwrite:InvalidOption"),
    when: "Name-value option is malformed or unsupported for the requested format.",
    message: "imwrite: invalid option",
};
pub const IMWRITE_ERROR_ENCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.ENCODE",
    identifier: Some("RunMat:imwrite:EncodeError"),
    when: "Image data cannot be encoded.",
    message: "imwrite: encode error",
};
pub const IMWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMWRITE.IO",
    identifier: Some("RunMat:imwrite:Io"),
    when: "Image file cannot be read for append or written.",
    message: "imwrite: file I/O error",
};

const IMWRITE_ERRORS: [BuiltinErrorDescriptor; 8] = [
    IMWRITE_ERROR_INVALID_ARGUMENT,
    IMWRITE_ERROR_INVALID_FILENAME,
    IMWRITE_ERROR_INVALID_FORMAT,
    IMWRITE_ERROR_INVALID_IMAGE,
    IMWRITE_ERROR_INVALID_COLORMAP,
    IMWRITE_ERROR_INVALID_OPTION,
    IMWRITE_ERROR_ENCODE,
    IMWRITE_ERROR_IO,
];

pub const IMWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMWRITE_ERRORS,
};

pub const IMWRITE_SINGLE_GIF_TIFF_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "imwrite-single-gif-tiff",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "imwrite accepts direct single image data for GIF and TIFF as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ImwriteSingleGifTiffExtension"),
    };
pub const IMWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [IMWRITE_SINGLE_GIF_TIFF_EXTENSION];
