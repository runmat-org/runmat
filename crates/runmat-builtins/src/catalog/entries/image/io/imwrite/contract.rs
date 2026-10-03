use crate::*;

/// The public spelling boundary for writer name-value options. Advanced
/// recognized names remain distinct from unknown names so they can retain
/// the stable unsupported-option error rather than being parsed as a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteOption {
    Alpha,
    Quality,
    WriteMode,
    DelayTime,
    LoopCount,
    UnsupportedAdvanced,
}

impl ImwriteOption {
    pub fn from_name(name: &str) -> Option<Self> {
        let normalized: String = name
            .chars()
            .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
            .flat_map(char::to_lowercase)
            .collect();
        match normalized.as_str() {
            "alpha" => Some(Self::Alpha),
            "quality" => Some(Self::Quality),
            "writemode" => Some(Self::WriteMode),
            "delaytime" => Some(Self::DelayTime),
            "loopcount" => Some(Self::LoopCount),
            "compression" | "bitdepth" | "mode" | "disposalmethod" | "backgroundcolor"
            | "comment" | "transparentcolor" => Some(Self::UnsupportedAdvanced),
            _ => None,
        }
    }

    pub fn validate_numeric(self, number: f64) -> Result<(), &'static str> {
        match self {
            Self::Quality if !number.is_finite() || !(0.0..=100.0).contains(&number) => {
                Err("Quality must be a scalar from 0 to 100")
            }
            Self::DelayTime if !number.is_finite() || number < 0.0 || number >= u64::MAX as f64 => {
                Err("DelayTime must be a finite non-negative duration representable by the GIF encoder")
            }
            Self::LoopCount if number.is_nan() || number < 0.0 => {
                Err("LoopCount must be an integer between 0 and 65535, or Inf")
            }
            Self::LoopCount if number.is_finite() && {
                let rounded = number.round();
                (rounded - number).abs() > 1e-6 || rounded > u16::MAX as f64
            } => Err("LoopCount must be an integer between 0 and 65535, or Inf"),
            Self::LoopCount if number.is_infinite() && number.is_sign_negative() => {
                Err("LoopCount must be an integer between 0 and 65535, or Inf")
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteWriteMode {
    Overwrite,
    Append,
}

impl ImwriteWriteMode {
    pub fn from_name(mode: &str) -> Option<Self> {
        let mode = mode.trim();
        if mode.eq_ignore_ascii_case("overwrite") {
            Some(Self::Overwrite)
        } else if mode.eq_ignore_ascii_case("append") {
            Some(Self::Append)
        } else {
            None
        }
    }
}

/// Supported encoder identity, decoded once from public format spellings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteFormat {
    Png,
    Jpeg,
    Bmp,
    Gif,
    Tiff,
}

impl ImwriteFormat {
    pub fn from_hint(hint: &str) -> Option<Self> {
        let label = hint.trim().trim_start_matches('.');
        if label.eq_ignore_ascii_case("png") {
            Some(Self::Png)
        } else if ["jpg", "jpeg", "jpe"]
            .iter()
            .any(|name| label.eq_ignore_ascii_case(name))
        {
            Some(Self::Jpeg)
        } else if label.eq_ignore_ascii_case("bmp") {
            Some(Self::Bmp)
        } else if label.eq_ignore_ascii_case("gif") {
            Some(Self::Gif)
        } else if ["tif", "tiff"]
            .iter()
            .any(|name| label.eq_ignore_ascii_case(name))
        {
            Some(Self::Tiff)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteAdmissionFailure {
    InvalidImage(&'static str),
    InvalidOption(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImwriteEncoderFacts {
    pub format: ImwriteFormat,
    pub channels: usize,
    pub sixteen_bit: bool,
    pub indexed_source: bool,
    pub alpha_applied: bool,
    pub append: bool,
}

impl ImwriteEncoderFacts {
    pub fn validate(self) -> Result<(), ImwriteAdmissionFailure> {
        use ImwriteAdmissionFailure::{InvalidImage, InvalidOption};
        use ImwriteFormat::*;
        if self.channels == 4 && !self.alpha_applied && self.format != Png {
            return Err(InvalidImage(if self.format == Tiff {
                "CMYK TIFF encoding is not supported by this RunMat encoder"
            } else {
                "direct four-channel input is supported for PNG only"
            }));
        }
        if self.append && self.format != Gif {
            return Err(InvalidOption(
                "WriteMode 'append' is supported for GIF files only",
            ));
        }
        match self.format {
            Gif if (self.channels == 3 && !self.indexed_source) || self.sixteen_bit => Err(
                InvalidImage("GIF direct input must be grayscale/indexed 8-bit data"),
            ),
            Jpeg if self.channels == 4 => {
                Err(InvalidOption("JPEG does not support alpha channels"))
            }
            Jpeg if self.sixteen_bit => Err(InvalidImage(
                "16-bit JPEG encoding is not supported by this RunMat encoder",
            )),
            Bmp if self.channels == 4 => Err(InvalidOption("BMP alpha output is not supported")),
            Bmp if self.sixteen_bit => {
                Err(InvalidImage("BMP does not support this 16-bit image input"))
            }
            Tiff if self.channels == 4 => Err(InvalidImage(
                "CMYK TIFF encoding is not supported by this RunMat encoder",
            )),
            _ => Ok(()),
        }
    }
}

const ALL_INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

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

const IMWRITE_DOCUMENTED_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];
const IMWRITE_REJECTED_INTEGER_CLASSES: [BuiltinIntegerClass; 6] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const IMWRITE_DOCUMENTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "A_or_X", classes: &IMWRITE_DOCUMENTED_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Uint8 and uint16 direct or indexed images retain their native sample/index interpretation through the host encoder sink." }];
const IMWRITE_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "A_or_X", classes: &IMWRITE_REJECTED_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Signed integer, uint32, and uint64 image arrays are outside the documented imwrite image-data surface and reject before file effects." }];
const IMWRITE_ALPHA_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];
const IMWRITE_ALPHA_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "Alpha", classes: &IMWRITE_ALPHA_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Documented uint8 and uint16 Alpha matrices are read from authoritative storage and scaled by their own full class range into the encoded channel." }];
const IMWRITE_CONTROL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "integer-valued format control", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "RowsPerStrip is documented for every integer class. Other integer-valued controls are format-specific; implemented controls read exact scalar storage before validation." }];
pub const IMWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(integer_A_or_X, ...)", inputs: &IMWRITE_DOCUMENTED_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The documented host/file sink retains native uint8 or uint16 samples and indexed values through encoding. Resident image input is gathered non-destructively through its exact owner, and all validation/encoding completes before the write begins. Encoder limitations such as 16-bit JPEG and TIFF CMYK reject explicitly without changing the class contract or creating a file." },
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(unsupported_integer_A_or_X, ...)", inputs: &IMWRITE_REJECTED_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Unsupported image classes reject from authoritative host or resident metadata without a floating compatibility conversion or file effect." },
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(A, ..., Alpha=integer_alpha)", inputs: &IMWRITE_ALPHA_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Documented uint8 and uint16 Alpha arrays preserve exact native samples until deterministic class-range scaling into the image bit depth; shape and format validation complete before file effects." },
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(A, ..., integer_control)", inputs: &IMWRITE_CONTROL_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Typed scalar controls are read exactly and range-checked before encoding. Documented but not-yet-supported format controls, including RowsPerStrip and explicit BitDepth, reject with the stable invalid-option error before any file effect rather than silently coercing or ignoring the value." },
];
