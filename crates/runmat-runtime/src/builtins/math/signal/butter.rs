//! Butterworth IIR filter design builtin.

use std::f64::consts::PI;

use num_complex::Complex64;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::butter_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "butter";
const EPS: f64 = 1.0e-12;
const REAL_TOL: f64 = 1.0e-8;
const MAX_ORDER: usize = 1024;

const BUTTER_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator coefficient vector.",
}];

const BUTTER_OUTPUT_BA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
];

const BUTTER_OUTPUT_ZPK: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter zeros.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter poles.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter gain.",
    },
];

const BUTTER_OUTPUT_SS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "State matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output matrix.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Feedthrough term.",
    },
];

const BUTTER_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized digital cutoff frequency or analog cutoff frequency.",
    },
];

const BUTTER_INPUTS_TYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized digital cutoff frequency or analog cutoff frequency.",
    },
    BuiltinParamDescriptor {
        name: "ftype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"low\" for scalar Wn, \"bandpass\" for two-element Wn"),
        description: "Filter type: low, high, bandpass, or stop.",
    },
];

const BUTTER_INPUTS_ANALOG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog cutoff frequency.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog filter-design flag.",
    },
];

const BUTTER_INPUTS_TYPE_ANALOG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog cutoff frequency.",
    },
    BuiltinParamDescriptor {
        name: "ftype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"low\" for scalar Wn, \"bandpass\" for two-element Wn"),
        description: "Filter type: low, high, bandpass, or stop.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Analog filter-design flag.",
    },
];

const BUTTER_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "b = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[b, a] = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_BA,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[z, p, k] = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_ZPK,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn)",
        inputs: &BUTTER_INPUTS_CORE,
        outputs: &BUTTER_OUTPUT_SS,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn, ftype)",
        inputs: &BUTTER_INPUTS_TYPE,
        outputs: &BUTTER_OUTPUT_SS,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn, 's')",
        inputs: &BUTTER_INPUTS_ANALOG,
        outputs: &BUTTER_OUTPUT_SS,
    },
    BuiltinSignatureDescriptor {
        label: "[A, B, C, D] = butter(n, Wn, ftype, 's')",
        inputs: &BUTTER_INPUTS_TYPE_ANALOG,
        outputs: &BUTTER_OUTPUT_SS,
    },
];

const BUTTER_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.ARG_COUNT",
    identifier: Some("RunMat:butter:ArgCount"),
    when: "The argument count is outside butter's supported forms.",
    message: "butter: expected two to four input arguments",
};

const BUTTER_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INVALID_ORDER",
    identifier: Some("RunMat:butter:InvalidOrder"),
    when: "Filter order is missing, non-finite, non-integer, less than one, or too large.",
    message: "butter: order must be a positive integer",
};

const BUTTER_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INVALID_FREQUENCY",
    identifier: Some("RunMat:butter:InvalidFrequency"),
    when: "Cutoff frequencies are malformed or outside the valid range.",
    message: "butter: invalid cutoff frequency",
};

const BUTTER_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INVALID_OPTION",
    identifier: Some("RunMat:butter:InvalidOption"),
    when: "Filter type or analog flag is malformed or unknown.",
    message: "butter: invalid option",
};

const BUTTER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:butter:TooManyOutputs"),
    when: "More than four outputs are requested.",
    message: "butter: too many output arguments",
};

const BUTTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTER.INTERNAL",
    identifier: Some("RunMat:butter:Internal"),
    when: "Filter design or output materialization fails internally.",
    message: "butter: internal error",
};

const BUTTER_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BUTTER_ERROR_ARG_COUNT,
    BUTTER_ERROR_INVALID_ORDER,
    BUTTER_ERROR_INVALID_FREQUENCY,
    BUTTER_ERROR_INVALID_OPTION,
    BUTTER_ERROR_TOO_MANY_OUTPUTS,
    BUTTER_ERROR_INTERNAL,
];

pub const BUTTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUTTER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUTTER_ERRORS,
};

fn butter_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    butter_error_with_message(error.message, error)
}

fn butter_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    butter_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn butter_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn butter_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "butter",
    category = "math/signal",
    summary = "Design Butterworth IIR filters.",
    keywords = "butter,butterworth,IIR,lowpass,highpass,bandpass,bandstop,signal processing",
    type_resolver(butter_type),
    descriptor(crate::builtins::math::signal::butter::BUTTER_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::butter"
)]
async fn butter_builtin(n: Value, wn: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();
    let args = ButterArgs::parse(n, wn, &rest).await?;
    let design = design_butterworth(&args)?;

    if let Some(out_count) = out_count {
        return match out_count {
            0 => Ok(Value::OutputList(Vec::new())),
            1 => Ok(Value::OutputList(vec![design.numerator_value()?])),
            2 => Ok(Value::OutputList(vec![
                design.numerator_value()?,
                design.denominator_value()?,
            ])),
            3 => {
                let (z, p, k) = design.zpk_values()?;
                Ok(Value::OutputList(vec![z, p, k]))
            }
            4 => {
                let (a, b, c, d) = design.state_space_values()?;
                Ok(Value::OutputList(vec![a, b, c, d]))
            }
            _ => Err(butter_error(&BUTTER_ERROR_TOO_MANY_OUTPUTS)),
        };
    }

    design.numerator_value()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Debug, Clone)]
struct ButterArgs {
    order: usize,
    cutoff: Vec<f64>,
    kind: FilterKind,
    analog: bool,
}

impl ButterArgs {
    async fn parse(n: Value, wn: Value, rest: &[Value]) -> BuiltinResult<Self> {
        if rest.len() > 2 {
            return Err(butter_error(&BUTTER_ERROR_ARG_COUNT));
        }

        let order = parse_order(gather_value(&n).await?)?;
        let cutoff = parse_cutoff(gather_value(&wn).await?)?;
        let mut analog = false;
        let mut kind = None;

        for value in rest {
            let token = parse_option_token(gather_value(value).await?)?;
            match classify_option_token(&token)? {
                OptionToken::Analog => {
                    if analog {
                        return Err(butter_error_with_detail(
                            &BUTTER_ERROR_INVALID_OPTION,
                            "analog flag was provided more than once",
                        ));
                    }
                    analog = true;
                }
                OptionToken::Kind(parsed_kind) => {
                    if kind.is_some() {
                        return Err(butter_error_with_detail(
                            &BUTTER_ERROR_INVALID_OPTION,
                            "filter type was provided more than once",
                        ));
                    }
                    kind = Some(parsed_kind);
                }
            }
        }

        let kind = match kind {
            Some(kind) => kind,
            None if cutoff.len() == 1 => FilterKind::Lowpass,
            None if cutoff.len() == 2 => FilterKind::Bandpass,
            None => {
                return Err(butter_error_with_detail(
                    &BUTTER_ERROR_INVALID_FREQUENCY,
                    "Wn must be a scalar or a two-element vector",
                ));
            }
        };

        validate_cutoff(&cutoff, kind, analog)?;

        Ok(Self {
            order,
            cutoff,
            kind,
            analog,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum OptionToken {
    Analog,
    Kind(FilterKind),
}

#[derive(Debug, Clone)]
struct Zpk {
    zeros: Vec<Complex64>,
    poles: Vec<Complex64>,
    gain: Complex64,
}

#[derive(Debug, Clone)]
struct ButterDesign {
    zeros: Vec<Complex64>,
    poles: Vec<Complex64>,
    gain: Complex64,
    numerator: Vec<Complex64>,
    denominator: Vec<Complex64>,
}

impl ButterDesign {
    fn numerator_value(&self) -> BuiltinResult<Value> {
        row_coefficients_to_value(&self.numerator)
    }

    fn denominator_value(&self) -> BuiltinResult<Value> {
        row_coefficients_to_value(&self.denominator)
    }

    fn zpk_values(&self) -> BuiltinResult<(Value, Value, Value)> {
        Ok((
            column_complex_to_value(&self.zeros)?,
            column_complex_to_value(&self.poles)?,
            scalar_complex_to_value(self.gain)?,
        ))
    }

    fn state_space_values(&self) -> BuiltinResult<(Value, Value, Value, Value)> {
        let b = real_coefficients(&self.numerator)?;
        let a = real_coefficients(&self.denominator)?;
        let (a_mat, b_mat, c_mat, d_val) = transfer_function_to_state_space(&b, &a)?;
        Ok((
            Value::Tensor(a_mat),
            Value::Tensor(b_mat),
            Value::Tensor(c_mat),
            Value::Num(d_val),
        ))
    }
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gpu_helpers::gather_value_async(value).await.map_err(|err| {
        butter_error_with_source(
            &BUTTER_ERROR_INTERNAL,
            "failed to gather GPU argument",
            map_control_flow_with_builtin(err, BUILTIN_NAME),
        )
    })
}

fn parse_order(value: Value) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(value) => value,
        Value::Int(value) => value.to_f64(),
        Value::Bool(value) => {
            if value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INVALID_ORDER,
                format!("received {other:?}"),
            ));
        }
    };

    if !raw.is_finite() {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            "order must be finite",
        ));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > EPS || rounded < 1.0 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            "order must be an integer greater than or equal to 1",
        ));
    }
    if rounded > MAX_ORDER as f64 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            format!("order exceeds RunMat's resource limit of {MAX_ORDER}"),
        ));
    }
    Ok(rounded as usize)
}

fn parse_cutoff(value: Value) -> BuiltinResult<Vec<f64>> {
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(value) => vec![if value { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => {
            ensure_vector_shape("Wn", &tensor.shape)?;
            tensor.data
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|detail| {
                butter_error_with_detail(&BUTTER_ERROR_INVALID_FREQUENCY, detail)
            })?;
            ensure_vector_shape("Wn", &tensor.shape)?;
            tensor.data
        }
        other => {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INVALID_FREQUENCY,
                format!("received {other:?}"),
            ));
        }
    };

    if values.is_empty() || values.len() > 2 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "Wn must be a scalar or a two-element vector",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "cutoff frequencies must be finite",
        ));
    }
    Ok(values)
}

fn parse_option_token(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text),
        Value::CharArray(chars) => Ok(chars.data.into_iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_OPTION,
            format!("expected a string option, received {other:?}"),
        )),
    }
}

fn classify_option_token(token: &str) -> BuiltinResult<OptionToken> {
    match token.trim().to_ascii_lowercase().as_str() {
        "s" => Ok(OptionToken::Analog),
        "low" | "lowpass" => Ok(OptionToken::Kind(FilterKind::Lowpass)),
        "high" | "highpass" => Ok(OptionToken::Kind(FilterKind::Highpass)),
        "bandpass" | "pass" => Ok(OptionToken::Kind(FilterKind::Bandpass)),
        "stop" | "bandstop" | "bandreject" | "notch" => Ok(OptionToken::Kind(FilterKind::Bandstop)),
        other => Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_OPTION,
            format!("unrecognized option '{other}'"),
        )),
    }
}

fn validate_cutoff(cutoff: &[f64], kind: FilterKind, analog: bool) -> BuiltinResult<()> {
    let expected_len = match kind {
        FilterKind::Lowpass | FilterKind::Highpass => 1,
        FilterKind::Bandpass | FilterKind::Bandstop => 2,
    };
    if cutoff.len() != expected_len {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            match kind {
                FilterKind::Lowpass | FilterKind::Highpass => {
                    "lowpass and highpass designs require scalar Wn"
                }
                FilterKind::Bandpass | FilterKind::Bandstop => {
                    "bandpass and bandstop designs require two-element Wn"
                }
            },
        ));
    }

    if cutoff.iter().any(|value| *value <= 0.0) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "cutoff frequencies must be greater than zero",
        ));
    }
    if !analog && cutoff.iter().any(|value| *value >= 1.0) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "digital cutoff frequencies must be less than 1.0, where 1.0 is Nyquist",
        ));
    }
    if cutoff.len() == 2 && cutoff[0] >= cutoff[1] {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "two-element Wn must be strictly increasing",
        ));
    }
    Ok(())
}

fn ensure_vector_shape(label: &str, shape: &[usize]) -> BuiltinResult<()> {
    if shape.is_empty() {
        return Ok(());
    }
    let non_singletons = shape.iter().filter(|&&dim| dim != 1).count();
    if non_singletons > 1 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            format!("{label} must be a vector"),
        ));
    }
    Ok(())
}

fn design_butterworth(args: &ButterArgs) -> BuiltinResult<ButterDesign> {
    let analog = analog_zpk(args)?;
    let zpk = if args.analog {
        analog
    } else {
        bilinear_zpk(&analog)?
    };
    let numerator = polynomial_from_roots(&zpk.zeros)
        .into_iter()
        .map(|coeff| coeff * zpk.gain)
        .collect::<Vec<_>>();
    let denominator = polynomial_from_roots(&zpk.poles);
    let (numerator, denominator) = normalize_transfer_function(numerator, denominator)?;

    Ok(ButterDesign {
        zeros: zpk.zeros,
        poles: zpk.poles,
        gain: zpk.gain,
        numerator,
        denominator,
    })
}

fn analog_zpk(args: &ButterArgs) -> BuiltinResult<Zpk> {
    let cutoff = if args.analog {
        args.cutoff.clone()
    } else {
        args.cutoff
            .iter()
            .map(|value| (PI * value / 2.0).tan())
            .collect::<Vec<_>>()
    };
    let prototype_poles = butterworth_prototype(args.order);

    let zpk = match args.kind {
        FilterKind::Lowpass => lowpass_zpk(&prototype_poles, cutoff[0], args.order),
        FilterKind::Highpass => highpass_zpk(&prototype_poles, cutoff[0], args.order),
        FilterKind::Bandpass => bandpass_zpk(&prototype_poles, cutoff[0], cutoff[1], args.order),
        FilterKind::Bandstop => bandstop_zpk(&prototype_poles, cutoff[0], cutoff[1], args.order),
    };
    validate_zpk(&zpk)?;
    Ok(zpk)
}

fn butterworth_prototype(order: usize) -> Vec<Complex64> {
    (0..order)
        .map(|idx| {
            let angle = PI * (2.0 * idx as f64 + order as f64 + 1.0) / (2.0 * order as f64);
            Complex64::from_polar(1.0, angle)
        })
        .collect()
}

fn lowpass_zpk(prototype_poles: &[Complex64], cutoff: f64, order: usize) -> Zpk {
    Zpk {
        zeros: Vec::new(),
        poles: prototype_poles.iter().map(|pole| *pole * cutoff).collect(),
        gain: Complex64::new(cutoff.powi(order as i32), 0.0),
    }
}

fn highpass_zpk(prototype_poles: &[Complex64], cutoff: f64, order: usize) -> Zpk {
    Zpk {
        zeros: vec![Complex64::new(0.0, 0.0); order],
        poles: prototype_poles
            .iter()
            .map(|pole| Complex64::new(cutoff, 0.0) / *pole)
            .collect(),
        gain: Complex64::new(1.0, 0.0),
    }
}

fn bandpass_zpk(prototype_poles: &[Complex64], low: f64, high: f64, order: usize) -> Zpk {
    let bandwidth = high - low;
    let center_sq = low * high;
    let mut poles = Vec::with_capacity(order * 2);
    for pole in prototype_poles {
        let scaled = *pole * bandwidth;
        let discriminant = scaled * scaled - Complex64::new(4.0 * center_sq, 0.0);
        let root = discriminant.sqrt();
        poles.push((scaled + root) * 0.5);
        poles.push((scaled - root) * 0.5);
    }
    Zpk {
        zeros: vec![Complex64::new(0.0, 0.0); order],
        poles,
        gain: Complex64::new(bandwidth.powi(order as i32), 0.0),
    }
}

fn bandstop_zpk(prototype_poles: &[Complex64], low: f64, high: f64, order: usize) -> Zpk {
    let bandwidth = high - low;
    let center = (low * high).sqrt();
    let center_sq = low * high;
    let mut poles = Vec::with_capacity(order * 2);
    for pole in prototype_poles {
        let scaled = Complex64::new(bandwidth, 0.0) / *pole;
        let discriminant = scaled * scaled - Complex64::new(4.0 * center_sq, 0.0);
        let root = discriminant.sqrt();
        poles.push((scaled + root) * 0.5);
        poles.push((scaled - root) * 0.5);
    }
    let mut zeros = Vec::with_capacity(order * 2);
    for _ in 0..order {
        zeros.push(Complex64::new(0.0, center));
        zeros.push(Complex64::new(0.0, -center));
    }
    Zpk {
        zeros,
        poles,
        gain: Complex64::new(1.0, 0.0),
    }
}

fn validate_zpk(zpk: &Zpk) -> BuiltinResult<()> {
    if !complex_is_finite(zpk.gain)
        || zpk.zeros.iter().any(|value| !complex_is_finite(*value))
        || zpk.poles.iter().any(|value| !complex_is_finite(*value))
    {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "filter design produced non-finite roots or gain",
        ));
    }
    Ok(())
}

fn bilinear_zpk(analog: &Zpk) -> BuiltinResult<Zpk> {
    let one = Complex64::new(1.0, 0.0);
    let mut zeros = Vec::with_capacity(analog.poles.len());
    for zero in &analog.zeros {
        let denominator = one - *zero;
        if denominator.norm() <= EPS {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INTERNAL,
                "analog zero cannot be mapped through bilinear transform",
            ));
        }
        zeros.push((one + *zero) / denominator);
    }

    let mut poles = Vec::with_capacity(analog.poles.len());
    for pole in &analog.poles {
        let denominator = one - *pole;
        if denominator.norm() <= EPS {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INTERNAL,
                "analog pole cannot be mapped through bilinear transform",
            ));
        }
        poles.push((one + *pole) / denominator);
    }

    let degree = poles.len().saturating_sub(zeros.len());
    zeros.extend(std::iter::repeat_n(Complex64::new(-1.0, 0.0), degree));

    let zero_factor = analog
        .zeros
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, zero| acc * (one - *zero));
    let pole_factor = analog
        .poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, pole| acc * (one - *pole));
    if pole_factor.norm() <= EPS {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "bilinear gain normalization failed",
        ));
    }
    let gain = analog.gain * zero_factor / pole_factor;

    let zpk = Zpk { zeros, poles, gain };
    validate_zpk(&zpk)?;
    Ok(zpk)
}

fn polynomial_from_roots(roots: &[Complex64]) -> Vec<Complex64> {
    let mut coeffs = vec![Complex64::new(1.0, 0.0)];
    for root in roots {
        let mut next = vec![Complex64::new(0.0, 0.0); coeffs.len() + 1];
        for (idx, coeff) in coeffs.iter().enumerate() {
            next[idx] += *coeff;
            next[idx + 1] -= *coeff * *root;
        }
        coeffs = next;
    }
    coeffs
}

fn normalize_transfer_function(
    mut numerator: Vec<Complex64>,
    mut denominator: Vec<Complex64>,
) -> BuiltinResult<(Vec<Complex64>, Vec<Complex64>)> {
    ensure_finite_coefficients(&numerator, "numerator")?;
    ensure_finite_coefficients(&denominator, "denominator")?;
    let Some(leading) = denominator.first().copied() else {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "empty denominator",
        ));
    };
    if leading.norm() <= EPS {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "zero leading denominator coefficient",
        ));
    }
    for coeff in &mut numerator {
        *coeff /= leading;
    }
    for coeff in &mut denominator {
        *coeff /= leading;
    }
    if numerator.len() < denominator.len() {
        let mut padded = vec![Complex64::new(0.0, 0.0); denominator.len() - numerator.len()];
        padded.extend(numerator);
        numerator = padded;
    }
    scrub_coefficients(&mut numerator);
    scrub_coefficients(&mut denominator);
    ensure_finite_coefficients(&numerator, "numerator")?;
    ensure_finite_coefficients(&denominator, "denominator")?;
    Ok((numerator, denominator))
}

fn ensure_finite_coefficients(coeffs: &[Complex64], label: &str) -> BuiltinResult<()> {
    if coeffs.iter().any(|value| !complex_is_finite(*value)) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            format!("filter design produced non-finite {label} coefficients"),
        ));
    }
    Ok(())
}

fn scrub_coefficients(coeffs: &mut [Complex64]) {
    for coeff in coeffs {
        if coeff.re.abs() <= REAL_TOL {
            coeff.re = 0.0;
        }
        if coeff.im.abs() <= REAL_TOL {
            coeff.im = 0.0;
        }
    }
}

fn row_coefficients_to_value(coeffs: &[Complex64]) -> BuiltinResult<Value> {
    complex_slice_to_value(coeffs, vec![1, coeffs.len()])
}

fn column_complex_to_value(values: &[Complex64]) -> BuiltinResult<Value> {
    complex_slice_to_value(values, vec![values.len(), 1])
}

fn scalar_complex_to_value(value: Complex64) -> BuiltinResult<Value> {
    if value.im.abs() <= REAL_TOL * (1.0 + value.re.abs()) {
        Ok(Value::Num(value.re))
    } else {
        let tensor = ComplexTensor::new(vec![(value.re, value.im)], vec![1, 1])
            .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn complex_slice_to_value(values: &[Complex64], shape: Vec<usize>) -> BuiltinResult<Value> {
    if values
        .iter()
        .all(|value| value.im.abs() <= REAL_TOL * (1.0 + value.re.abs()))
    {
        let tensor = Tensor::new(values.iter().map(|value| value.re).collect(), shape)
            .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;
        Ok(Value::Tensor(tensor))
    } else {
        let tensor = ComplexTensor::new(
            values.iter().map(|value| (value.re, value.im)).collect(),
            shape,
        )
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn real_coefficients(values: &[Complex64]) -> BuiltinResult<Vec<f64>> {
    values
        .iter()
        .map(|value| {
            if value.im.abs() <= REAL_TOL * (1.0 + value.re.abs()) {
                Ok(value.re)
            } else {
                Err(butter_error_with_detail(
                    &BUTTER_ERROR_INTERNAL,
                    "state-space output requires real coefficients",
                ))
            }
        })
        .collect()
}

fn transfer_function_to_state_space(
    numerator: &[f64],
    denominator: &[f64],
) -> BuiltinResult<(Tensor, Tensor, Tensor, f64)> {
    if denominator.len() < 2 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "state-space conversion requires a positive-order denominator",
        ));
    }

    let mut a = denominator.to_vec();
    let mut b = numerator.to_vec();
    let leading = a[0];
    if leading.abs() <= EPS {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "state-space conversion received a zero leading denominator",
        ));
    }
    for coeff in &mut a {
        *coeff /= leading;
    }
    for coeff in &mut b {
        *coeff /= leading;
    }
    if b.len() < a.len() {
        let mut padded = vec![0.0; a.len() - b.len()];
        padded.extend(b);
        b = padded;
    } else if b.len() > a.len() {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "state-space conversion requires a proper transfer function",
        ));
    }

    let order = a.len() - 1;
    let mut a_data = vec![0.0; order * order];
    for col in 0..order {
        a_data[col * order] = -a[col + 1];
    }
    for row in 1..order {
        a_data[row + (row - 1) * order] = 1.0;
    }
    let a_tensor = Tensor::new(a_data, vec![order, order])
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;

    let mut b_data = vec![0.0; order];
    b_data[0] = 1.0;
    let b_tensor = Tensor::new(b_data, vec![order, 1])
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;

    let d_value = b[0];
    let c_data = (0..order)
        .map(|idx| b[idx + 1] - a[idx + 1] * d_value)
        .collect::<Vec<_>>();
    let c_tensor = Tensor::new(c_data, vec![1, order])
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;

    Ok((a_tensor, b_tensor, c_tensor, d_value))
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, IntValue};

    fn output_values(order: usize, cutoff: Value, rest: Vec<Value>, outputs: usize) -> Vec<Value> {
        let _guard = crate::output_count::push_output_count(Some(outputs));
        match block_on(butter_builtin(
            Value::Int(IntValue::U64(order as u64)),
            cutoff,
            rest,
        ))
        .expect("butter")
        {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn tensor_data(value: &Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data.clone(),
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn butter_rejects_non_finite_transfer_coefficients() {
        let err = normalize_transfer_function(
            vec![Complex64::new(f64::INFINITY, 0.0)],
            vec![Complex64::new(1.0, 0.0)],
        )
        .expect_err("non-finite numerator should fail");
        assert_eq!(err.identifier(), BUTTER_ERROR_INTERNAL.identifier);
        assert!(err.message().contains("non-finite numerator coefficients"));
    }

    fn complex_column(value: &Value) -> Vec<Complex64> {
        match value {
            Value::Tensor(tensor) => tensor
                .data
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect(),
            Value::ComplexTensor(tensor) => tensor
                .data
                .iter()
                .map(|(re, im)| Complex64::new(*re, *im))
                .collect(),
            other => panic!("expected complex-compatible vector, got {other:?}"),
        }
    }

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= tol,
            "expected {lhs} to be within {tol} of {rhs}; diff={diff}"
        );
    }

    fn assert_slice_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= tol,
                "mismatch at {idx}: {left} vs {right}; diff={diff}"
            );
        }
    }

    fn dc_gain(b: &[f64], a: &[f64]) -> f64 {
        b.iter().sum::<f64>() / a.iter().sum::<f64>()
    }

    fn nyquist_gain(b: &[f64], a: &[f64]) -> f64 {
        let numerator = b
            .iter()
            .enumerate()
            .map(|(idx, coeff)| if idx % 2 == 0 { *coeff } else { -*coeff })
            .sum::<f64>();
        let denominator = a
            .iter()
            .enumerate()
            .map(|(idx, coeff)| if idx % 2 == 0 { *coeff } else { -*coeff })
            .sum::<f64>();
        numerator / denominator
    }

    #[test]
    fn butter_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("butter builtin");
        let descriptor = builtin.descriptor.expect("butter descriptor");
        let labels = descriptor
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"b = butter(n, Wn)"));
        assert!(labels.contains(&"b = butter(n, Wn, 's')"));
        assert!(labels.contains(&"[b, a] = butter(n, Wn, ftype)"));
        assert!(labels.contains(&"[b, a] = butter(n, Wn, 's')"));
        assert!(labels.contains(&"[z, p, k] = butter(n, Wn, ftype, 's')"));
        assert!(labels.contains(&"[z, p, k] = butter(n, Wn, 's')"));
        assert!(labels.contains(&"[A, B, C, D] = butter(n, Wn)"));
        assert!(labels.contains(&"[A, B, C, D] = butter(n, Wn, 's')"));
        assert!(descriptor
            .errors
            .iter()
            .any(|error| error.code == "RM.BUTTER.INVALID_FREQUENCY"));
    }

    #[test]
    fn butter_second_order_coefficients_match_reference_values() {
        let low = output_values(2, Value::Num(0.25), Vec::new(), 2);
        assert_slice_close(
            &tensor_data(&low[0]),
            &[0.0976310729378175, 0.195262145875635, 0.0976310729378175],
            1e-12,
        );
        assert_slice_close(
            &tensor_data(&low[1]),
            &[1.0, -0.9428090415820632, 0.33333333333333326],
            1e-12,
        );

        let high = output_values(
            2,
            Value::Num(0.25),
            vec![Value::String("high".to_string())],
            2,
        );
        assert_slice_close(
            &tensor_data(&high[0]),
            &[0.5690355937288492, -1.1380711874576983, 0.5690355937288492],
            1e-12,
        );
        assert_slice_close(
            &tensor_data(&high[1]),
            &[1.0, -0.9428090415820632, 0.33333333333333326],
            1e-12,
        );

        let wn = Value::Tensor(Tensor::new(vec![0.2, 0.4], vec![1, 2]).unwrap());
        let bandpass = output_values(
            2,
            wn.clone(),
            vec![Value::String("bandpass".to_string())],
            2,
        );
        assert_slice_close(
            &tensor_data(&bandpass[0]),
            &[
                0.06745527388907191,
                0.0,
                -0.13491054777814382,
                0.0,
                0.06745527388907191,
            ],
            1e-12,
        );
        assert_slice_close(
            &tensor_data(&bandpass[1]),
            &[
                1.0,
                -1.942_468_776_547_884,
                2.119_202_397_144_283,
                -1.216651635515531,
                0.4128015980961886,
            ],
            1e-12,
        );

        let bandstop = output_values(2, wn, vec![Value::String("stop".to_string())], 2);
        assert_slice_close(
            &tensor_data(&bandstop[0]),
            &[
                0.6389455251590221,
                -1.579_560_206_031_707,
                2.254_112_944_922_426,
                -1.579_560_206_031_707,
                0.6389455251590221,
            ],
            1e-12,
        );
        assert_slice_close(
            &tensor_data(&bandstop[1]),
            &[
                1.0,
                -1.942_468_776_547_884,
                2.119_202_397_144_283,
                -1.216651635515531,
                0.4128015980961886,
            ],
            1e-12,
        );
    }

    #[test]
    fn butter_first_order_lowpass_matches_closed_form() {
        let values = output_values(1, Value::Num(0.5), Vec::new(), 2);
        assert_slice_close(&tensor_data(&values[0]), &[0.5, 0.5], 1e-12);
        assert_slice_close(&tensor_data(&values[1]), &[1.0, 0.0], 1e-12);
    }

    #[test]
    fn butter_first_order_highpass_matches_closed_form() {
        let values = output_values(
            1,
            Value::Num(0.5),
            vec![Value::String("high".to_string())],
            2,
        );
        assert_slice_close(&tensor_data(&values[0]), &[0.5, -0.5], 1e-12);
        assert_slice_close(&tensor_data(&values[1]), &[1.0, 0.0], 1e-12);
    }

    #[test]
    fn butter_fourth_order_lowpass_has_unity_dc_gain() {
        let values = output_values(
            4,
            Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap()),
            vec![Value::String("low".to_string())],
            2,
        );
        let b = tensor_data(&values[0]);
        let a = tensor_data(&values[1]);
        assert_eq!(b.len(), 5);
        assert_eq!(a.len(), 5);
        assert_close(dc_gain(&b, &a), 1.0, 1e-8);
        assert!(nyquist_gain(&b, &a).abs() < 1e-10);
    }

    #[test]
    fn butter_bandpass_and_bandstop_have_expected_edge_zeros() {
        let wn = Value::Tensor(Tensor::new(vec![0.2, 0.4], vec![1, 2]).unwrap());
        let bandpass = output_values(
            3,
            wn.clone(),
            vec![Value::String("bandpass".to_string())],
            2,
        );
        let b_pass = tensor_data(&bandpass[0]);
        let a_pass = tensor_data(&bandpass[1]);
        assert_eq!(b_pass.len(), 7);
        assert_eq!(a_pass.len(), 7);
        assert!(dc_gain(&b_pass, &a_pass).abs() < 1e-9);
        assert!(nyquist_gain(&b_pass, &a_pass).abs() < 1e-9);

        let bandstop = output_values(3, wn, vec![Value::String("stop".to_string())], 2);
        let b_stop = tensor_data(&bandstop[0]);
        let a_stop = tensor_data(&bandstop[1]);
        assert_eq!(b_stop.len(), 7);
        assert_eq!(a_stop.len(), 7);
        assert_close(dc_gain(&b_stop, &a_stop), 1.0, 1e-8);
        assert_close(nyquist_gain(&b_stop, &a_stop), 1.0, 1e-8);
    }

    #[test]
    fn butter_zpk_outputs_stable_digital_poles() {
        let values = output_values(4, Value::Num(0.35), Vec::new(), 3);
        let zeros = complex_column(&values[0]);
        let poles = complex_column(&values[1]);
        assert_eq!(zeros.len(), 4);
        assert_eq!(poles.len(), 4);
        assert!(zeros.iter().all(|zero| (zero.re + 1.0).abs() < 1e-8));
        assert!(poles.iter().all(|pole| pole.norm() < 1.0));
        match &values[2] {
            Value::Num(value) => assert!(*value > 0.0),
            other => panic!("expected scalar gain, got {other:?}"),
        }
    }

    #[test]
    fn butter_state_space_outputs_canonical_dimensions() {
        let values = output_values(2, Value::Num(0.25), Vec::new(), 4);
        match &values[0] {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 2]),
            other => panic!("expected state matrix, got {other:?}"),
        }
        match &values[1] {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.data, vec![1.0, 0.0]);
            }
            other => panic!("expected input matrix, got {other:?}"),
        }
        match &values[2] {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![1, 2]),
            other => panic!("expected output matrix, got {other:?}"),
        }
        assert!(matches!(values[3], Value::Num(_)));
    }

    #[test]
    fn butter_analog_lowpass_and_highpass_are_supported() {
        let low = output_values(1, Value::Num(2.0), vec![Value::String("s".to_string())], 2);
        assert_slice_close(&tensor_data(&low[0]), &[0.0, 2.0], 1e-12);
        assert_slice_close(&tensor_data(&low[1]), &[1.0, 2.0], 1e-12);

        let high = output_values(
            1,
            Value::Num(2.0),
            vec![
                Value::String("high".to_string()),
                Value::String("s".to_string()),
            ],
            2,
        );
        assert_slice_close(&tensor_data(&high[0]), &[1.0, 0.0], 1e-12);
        assert_slice_close(&tensor_data(&high[1]), &[1.0, 2.0], 1e-12);
    }

    #[test]
    fn butter_rejects_invalid_frequency_and_outputs() {
        let err = block_on(butter_builtin(Value::Num(2.0), Value::Num(1.0), Vec::new()))
            .expect_err("digital Wn at Nyquist should be rejected");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:butter:InvalidFrequency")
        );

        let _guard = crate::output_count::push_output_count(Some(5));
        let err = block_on(butter_builtin(Value::Num(2.0), Value::Num(0.4), Vec::new()))
            .expect_err("too many outputs should be rejected");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:butter:TooManyOutputs")
        );
    }
}
