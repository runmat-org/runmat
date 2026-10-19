use crate::*;

pub const PRINT_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "print-status-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Request a logical success status from PNG file export",
    error_identifier: Some("RunMat:compatibility:PrintStatusOutputExtension"),
};
pub const PRINT_INTEGER_FIGURE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "print-integer-figure-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Use a typed-integer alias for RunMat's numeric figure registry",
    error_identifier: Some("RunMat:compatibility:PrintIntegerFigureHandleExtension"),
};
pub const PRINT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PRINT_STATUS_OUTPUT_EXTENSION,
    PRINT_INTEGER_FIGURE_EXTENSION,
];

const INTEGER_HANDLE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "fig numeric handle alias",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "MATLAB documents a Figure object. RunMat mode permits an exactly representable typed-integer alias for its numeric graphics registry.",
}];
pub const PRINT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "print(integer_fig, filename, ___)",
        inputs: &INTEGER_HANDLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The alias is gated before gather and must map exactly into RunMat's f64-backed graphics handle registry. Resolution remains a text option.",
    }];

const NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];
const STATUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat-only logical completion status.",
}];
const FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "PNG destination; .png is appended when absent.",
}];
const FILE_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "PNG destination; .png is appended when absent.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional -dpng and -rN tokens in either order with the filename.",
    },
];
const FIG_FILE_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fig",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Figure handle or RunMat-only typed-integer handle alias.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "PNG destination; .png is appended when absent.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional -dpng and -rN tokens.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "print(filename)",
        inputs: &FILE,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "print(filename, '-dpng', '-rN')",
        inputs: &FILE_OPTIONS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "print(fig, filename, ___)",
        inputs: &FIG_FILE_OPTIONS,
        outputs: &NO_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "ok = print(filename, ___)",
        inputs: &FILE_OPTIONS,
        outputs: &STATUS_OUTPUT,
    },
];

pub const PRINT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.INVALID_INPUT",
    identifier: Some("RunMat:print:InvalidInput"),
    when: "The filename, figure handle, or argument ordering is invalid.",
    message: "print: invalid input arguments",
};
pub const PRINT_ERROR_UNSUPPORTED_DEVICE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.UNSUPPORTED_DEVICE",
    identifier: Some("RunMat:print:UnsupportedDevice"),
    when: "The requested device is not supplied by the active exporter.",
    message: "print: unsupported output device",
};
pub const PRINT_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.INVALID_OPTION",
    identifier: Some("RunMat:print:InvalidOption"),
    when:
        "An option token or resolution is invalid, a device is duplicated, or PNG dimensions exceed supported limits. Repeated resolution tokens use the last value.",
    message: "print: invalid option",
};
pub const PRINT_ERROR_RENDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.RENDER",
    identifier: Some("RunMat:print:RenderFailed"),
    when: "Figure capture or PNG rendering fails.",
    message: "print: figure export failed",
};
pub const PRINT_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.IO",
    identifier: Some("RunMat:print:IoFailure"),
    when: "Captured filesystem authority, staging, writing, or publication fails.",
    message: "print: file I/O failed",
};
pub const PRINT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.INTERNAL",
    identifier: None,
    when: "An underlying materialization or runtime control-flow operation fails with its original cause.",
    message: "print: internal error",
};
pub const PRINT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRINT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:print:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "print: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    PRINT_ERROR_INVALID_INPUT,
    PRINT_ERROR_UNSUPPORTED_DEVICE,
    PRINT_ERROR_INVALID_OPTION,
    PRINT_ERROR_RENDER,
    PRINT_ERROR_IO,
    PRINT_ERROR_INTERNAL,
    PRINT_ERROR_TOO_MANY_OUTPUTS,
];
pub const PRINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

/// The only admitted file device in the current renderer. Other MATLAB device
/// spellings are recognized as unsupported rather than silently made PNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintDevice {
    Png,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintControl {
    Device(PrintDevice),
    Resolution(u32),
}

#[derive(Debug)]
pub enum PrintControlError {
    InvalidOption,
    UnsupportedDevice,
    ResolutionParse(std::num::ParseIntError),
    UnsupportedResolution(u64),
}

pub const PRINT_DEFAULT_DPI: u32 = 150;
pub const PRINT_DEFAULT_WIDTH: u32 = 800;
pub const PRINT_DEFAULT_HEIGHT: u32 = 600;
pub const PRINT_MAX_EXPORT_DIMENSION: u32 = 8192;

impl PrintControl {
    pub fn decode(token: &str) -> Result<Self, PrintControlError> {
        let Some(option) = token.strip_prefix('-') else {
            return Err(PrintControlError::InvalidOption);
        };
        let lower = option.to_ascii_lowercase();
        if let Some(device) = lower.strip_prefix('d') {
            return match device {
                "png" => Ok(Self::Device(PrintDevice::Png)),
                "" => Err(PrintControlError::InvalidOption),
                _ => Err(PrintControlError::UnsupportedDevice),
            };
        }
        if let Some(resolution) = lower.strip_prefix('r') {
            let dpi = resolution
                .parse::<u32>()
                .map_err(PrintControlError::ResolutionParse)?;
            let dpi = if dpi == 0 { PRINT_DEFAULT_DPI } else { dpi };
            let width = scaled_dimension(PRINT_DEFAULT_WIDTH, dpi)?;
            let height = scaled_dimension(PRINT_DEFAULT_HEIGHT, dpi)?;
            if width == 0 || height == 0 {
                return Err(PrintControlError::UnsupportedResolution(0));
            }
            return Ok(Self::Resolution(dpi));
        }
        Err(PrintControlError::InvalidOption)
    }
}

pub fn print_dimensions(dpi: u32) -> Result<(u32, u32), PrintControlError> {
    Ok((
        scaled_dimension(PRINT_DEFAULT_WIDTH, dpi)?,
        scaled_dimension(PRINT_DEFAULT_HEIGHT, dpi)?,
    ))
}

fn scaled_dimension(base: u32, dpi: u32) -> Result<u32, PrintControlError> {
    let scaled = (u64::from(base) * u64::from(dpi)).div_ceil(u64::from(PRINT_DEFAULT_DPI));
    if scaled == 0 || scaled > u64::from(PRINT_MAX_EXPORT_DIMENSION) {
        return Err(PrintControlError::UnsupportedResolution(scaled));
    }
    Ok(scaled as u32)
}
