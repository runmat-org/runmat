use crate::*;

const SOURCE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Local path, file URL, or HTTP(S) URL to load.",
}];
const SOURCE_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Local path, file URL, or HTTP(S) URL to load.",
    },
    BuiltinParamDescriptor {
        name: "fmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Explicit supported image format hint.",
    },
];
const IMAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decoded grayscale or truecolor image array.",
};
const MAP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "map",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty double colormap for the current direct-color decoding forms.",
};
const ALPHA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "alpha",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decoded alpha plane when present, otherwise empty double.",
};
const OUTPUT_I: [BuiltinParamDescriptor; 1] = [IMAGE];
const OUTPUT_IMAP: [BuiltinParamDescriptor; 2] = [IMAGE, MAP];
const OUTPUT_IMAP_ALPHA: [BuiltinParamDescriptor; 3] = [IMAGE, MAP, ALPHA];
const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "I = imread(filename)",
        inputs: &SOURCE,
        outputs: &OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "I = imread(filename, fmt)",
        inputs: &SOURCE_FORMAT,
        outputs: &OUTPUT_I,
    },
    BuiltinSignatureDescriptor {
        label: "[I, map] = imread(filename)",
        inputs: &SOURCE,
        outputs: &OUTPUT_IMAP,
    },
    BuiltinSignatureDescriptor {
        label: "[I, map] = imread(filename, fmt)",
        inputs: &SOURCE_FORMAT,
        outputs: &OUTPUT_IMAP,
    },
    BuiltinSignatureDescriptor {
        label: "[I, map, alpha] = imread(filename)",
        inputs: &SOURCE,
        outputs: &OUTPUT_IMAP_ALPHA,
    },
    BuiltinSignatureDescriptor {
        label: "[I, map, alpha] = imread(filename, fmt)",
        inputs: &SOURCE_FORMAT,
        outputs: &OUTPUT_IMAP_ALPHA,
    },
];

macro_rules! diagnostic {
    ($name:ident, $code:literal, $id:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($id),
            when: $when,
            message: $message,
        };
    };
}
diagnostic!(
    IMREAD_ERROR_INVALID_ARGUMENT,
    "RM.IMREAD.INVALID_ARGUMENT",
    "RunMat:imread:InvalidArgument",
    "Input argument types are invalid.",
    "imread: invalid argument"
);
diagnostic!(
    IMREAD_ERROR_INVALID_FILENAME,
    "RM.IMREAD.INVALID_FILENAME",
    "RunMat:imread:InvalidFilename",
    "Filename is empty.",
    "imread: invalid filename"
);
diagnostic!(
    IMREAD_ERROR_INVALID_FORMAT,
    "RM.IMREAD.INVALID_FORMAT",
    "RunMat:imread:InvalidFormat",
    "Format hint is empty.",
    "imread: invalid format hint"
);
diagnostic!(
    IMREAD_ERROR_UNSUPPORTED_FORMAT,
    "RM.IMREAD.UNSUPPORTED_FORMAT",
    "RunMat:imread:UnsupportedFormat",
    "Format hint is unsupported.",
    "imread: unsupported image format"
);
diagnostic!(
    IMREAD_ERROR_TOO_MANY_INPUTS,
    "RM.IMREAD.TOO_MANY_INPUTS",
    "RunMat:imread:TooManyInputs",
    "More than two inputs are supplied.",
    "imread: too many input arguments"
);
diagnostic!(
    IMREAD_ERROR_TOO_MANY_OUTPUTS,
    "RM.IMREAD.TOO_MANY_OUTPUTS",
    "RunMat:imread:TooManyOutputs",
    "More than three outputs are requested.",
    "imread: too many output arguments"
);
diagnostic!(
    IMREAD_ERROR_UNSUPPORTED_SCHEME,
    "RM.IMREAD.UNSUPPORTED_SCHEME",
    "RunMat:imread:UnsupportedScheme",
    "URL scheme is unsupported.",
    "imread: unsupported URL scheme"
);
diagnostic!(
    IMREAD_ERROR_FILE_READ,
    "RM.IMREAD.FILE_READ",
    "RunMat:imread:FileReadError",
    "Local file cannot be read.",
    "imread: file read error"
);
diagnostic!(
    IMREAD_ERROR_INVALID_FILE_URL,
    "RM.IMREAD.INVALID_FILE_URL",
    "RunMat:imread:InvalidFileUrl",
    "File URL path or host is invalid.",
    "imread: invalid file URL"
);
diagnostic!(
    IMREAD_ERROR_TIMEOUT,
    "RM.IMREAD.TIMEOUT",
    "RunMat:imread:Timeout",
    "HTTP request times out.",
    "imread: request timed out"
);
diagnostic!(
    IMREAD_ERROR_NETWORK,
    "RM.IMREAD.NETWORK",
    "RunMat:imread:NetworkError",
    "HTTP request fails.",
    "imread: network error"
);
diagnostic!(
    IMREAD_ERROR_HTTP_STATUS,
    "RM.IMREAD.HTTP_STATUS",
    "RunMat:imread:HttpStatus",
    "HTTP response is not successful.",
    "imread: HTTP status error"
);
diagnostic!(
    IMREAD_ERROR_INVALID_HEADER,
    "RM.IMREAD.INVALID_HEADER",
    "RunMat:imread:InvalidHeader",
    "HTTP request header is invalid.",
    "imread: invalid request header"
);
diagnostic!(
    IMREAD_ERROR_DECODE,
    "RM.IMREAD.DECODE",
    "RunMat:imread:DecodeError",
    "Image bytes cannot be decoded.",
    "imread: decode error"
);
diagnostic!(
    IMREAD_ERROR_SHAPE,
    "RM.IMREAD.SHAPE",
    "RunMat:imread:ShapeError",
    "Decoded image tensor cannot be materialized.",
    "imread: shape materialization error"
);

const ERRORS: [BuiltinErrorDescriptor; 15] = [
    IMREAD_ERROR_INVALID_ARGUMENT,
    IMREAD_ERROR_INVALID_FILENAME,
    IMREAD_ERROR_INVALID_FORMAT,
    IMREAD_ERROR_UNSUPPORTED_FORMAT,
    IMREAD_ERROR_TOO_MANY_INPUTS,
    IMREAD_ERROR_TOO_MANY_OUTPUTS,
    IMREAD_ERROR_UNSUPPORTED_SCHEME,
    IMREAD_ERROR_FILE_READ,
    IMREAD_ERROR_INVALID_FILE_URL,
    IMREAD_ERROR_TIMEOUT,
    IMREAD_ERROR_NETWORK,
    IMREAD_ERROR_HTTP_STATUS,
    IMREAD_ERROR_INVALID_HEADER,
    IMREAD_ERROR_DECODE,
    IMREAD_ERROR_SHAPE,
];
pub const IMREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
