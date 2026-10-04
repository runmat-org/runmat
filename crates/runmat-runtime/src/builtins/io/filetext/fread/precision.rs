use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum InputType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

impl InputType {
    pub(super) fn byte_len(&self) -> usize {
        match self {
            InputType::UInt8 | InputType::Int8 => 1,
            InputType::UInt16 | InputType::Int16 => 2,
            InputType::UInt32 | InputType::Int32 | InputType::Float32 => 4,
            InputType::UInt64 | InputType::Int64 | InputType::Float64 => 8,
        }
    }

    pub(super) fn numeric_dtype(self) -> NumericDType {
        match self {
            Self::UInt8 => NumericDType::U8,
            Self::Int8 => NumericDType::I8,
            Self::UInt16 => NumericDType::U16,
            Self::Int16 => NumericDType::I16,
            Self::UInt32 => NumericDType::U32,
            Self::Int32 => NumericDType::I32,
            Self::UInt64 => NumericDType::U64,
            Self::Int64 => NumericDType::I64,
            Self::Float32 => NumericDType::F32,
            Self::Float64 => NumericDType::F64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum OutputKind {
    Numeric(NumericDType),
    Char,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PrecisionSpec {
    pub(super) input: InputType,
    pub(super) output: OutputKind,
    pub(super) repeat: usize,
}

impl PrecisionSpec {
    fn default() -> Self {
        Self {
            input: InputType::UInt8,
            output: OutputKind::Numeric(NumericDType::F64),
            repeat: 1,
        }
    }
}

pub(super) fn parse_precision(arg: Option<&Value>) -> Result<PrecisionSpec, String> {
    match arg {
        None => Ok(PrecisionSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
        }
    }
}

pub(super) fn parse_precision_string(raw: &str) -> Result<PrecisionSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('*') {
        parse_star_precision(rest.trim())
    } else if let Some((lhs, rhs)) = lower.split_once("=>") {
        let (repeat, input) = parse_repeated_input_label(lhs.trim())?;
        let output = parse_output_label(rhs.trim())?;
        Ok(PrecisionSpec {
            input,
            output,
            repeat,
        })
    } else {
        let (repeat, input) = parse_repeated_input_label(lower.trim())?;
        let wants_char =
            lower == "char" || (matches!(input, InputType::UInt8) && lower.contains("char"));
        let output = if wants_char {
            OutputKind::Char
        } else {
            OutputKind::Numeric(NumericDType::F64)
        };
        Ok(PrecisionSpec {
            input,
            output,
            repeat,
        })
    }
}

pub(super) fn parse_star_precision(label: &str) -> Result<PrecisionSpec, String> {
    if label == "char" {
        return Ok(PrecisionSpec {
            input: InputType::UInt8,
            output: OutputKind::Char,
            repeat: 1,
        });
    }
    let input = parse_input_label(label)?;
    Ok(PrecisionSpec {
        input,
        output: OutputKind::Numeric(input.numeric_dtype()),
        repeat: 1,
    })
}

pub(super) fn parse_repeated_input_label(label: &str) -> Result<(usize, InputType), String> {
    if let Some((repeat, source)) = label.split_once('*') {
        if repeat.trim().chars().all(|ch| ch.is_ascii_digit()) {
            let repeat = repeat
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|&repeat| repeat > 0)
                .ok_or_else(|| "precision repeat count must be a positive integer".to_string())?;
            return Ok((repeat, parse_input_label(source.trim())?));
        }
    }
    Ok((1, parse_input_label(label)?))
}

pub(super) fn parse_input_label(label: &str) -> Result<InputType, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(InputType::Float64),
        "single" | "float" | "float32" | "real*4" => Ok(InputType::Float32),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => Ok(InputType::Int8),
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" | "char" | "byte" => {
            Ok(InputType::UInt8)
        }
        "int16" | "short" | "integer*2" => Ok(InputType::Int16),
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => Ok(InputType::UInt16),
        "int32" | "int" | "integer*4" | "long" => Ok(InputType::Int32),
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(InputType::UInt32)
        }
        "int64" | "integer*8" | "longlong" | "long long" => Ok(InputType::Int64),
        "uint64" | "unsignedlonglong" | "unsigned long long" => Ok(InputType::UInt64),
        other => Err(format!("unsupported precision '{other}'")),
    }
}

pub(super) fn parse_output_label(label: &str) -> Result<OutputKind, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(OutputKind::Numeric(NumericDType::F64)),
        "single" | "float" | "float32" | "real*4" => Ok(OutputKind::Numeric(NumericDType::F32)),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => {
            Ok(OutputKind::Numeric(NumericDType::I8))
        }
        "int16" | "short" | "integer*2" => Ok(OutputKind::Numeric(NumericDType::I16)),
        "int32" | "int" | "integer*4" | "long" => Ok(OutputKind::Numeric(NumericDType::I32)),
        "int64" | "integer*8" | "longlong" | "long long" => {
            Ok(OutputKind::Numeric(NumericDType::I64))
        }
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" => {
            Ok(OutputKind::Numeric(NumericDType::U8))
        }
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => {
            Ok(OutputKind::Numeric(NumericDType::U16))
        }
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(OutputKind::Numeric(NumericDType::U32))
        }
        "uint64" | "unsignedlonglong" | "unsigned long long" => {
            Ok(OutputKind::Numeric(NumericDType::U64))
        }
        "char" => Ok(OutputKind::Char),
        other => Err(format!("output class '{other}' is not implemented yet")),
    }
}

pub(super) fn parse_skip(arg: Option<&Value>) -> Result<usize, String> {
    match arg {
        None => Ok(0),
        Some(Value::Int(int)) => int_to_skip(int),
        Some(Value::Tensor(t)) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int_to_skip(&int);
            }
            parse_skip_scalar(tensor::tensor_value_f64(t, 0))
        }
        Some(value) => {
            let scalar = value_to_scalar(value, "skip value must be numeric")?;
            parse_skip_scalar(scalar)
        }
    }
}

pub(super) fn parse_skip_scalar(scalar: f64) -> Result<usize, String> {
    if !scalar.is_finite() {
        return Err("skip value must be finite".to_string());
    }
    if scalar < 0.0 {
        return Err("skip value must be non-negative".to_string());
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > f64::EPSILON {
        return Err("skip value must be an integer".to_string());
    }
    if rounded >= i64::MAX as f64 {
        return Err("skip value is too large".to_string());
    }
    Ok(rounded as usize)
}

pub(super) fn int_to_skip(value: &IntValue) -> Result<usize, String> {
    let skip = int_to_usize(value, "skip value must be non-negative")?;
    if skip > i64::MAX as usize {
        return Err("skip value is too large".to_string());
    }
    Ok(skip)
}

pub(super) fn int_to_usize(value: &IntValue, err: &str) -> Result<usize, String> {
    value.try_to_usize().ok_or_else(|| err.to_string())
}

#[derive(Clone, Copy, Debug)]
pub(super) enum MachineFormat {
    Native,
    LittleEndian,
    BigEndian,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Endianness {
    Little,
    Big,
}

impl MachineFormat {
    pub(super) fn to_endianness(self) -> Endianness {
        match self {
            MachineFormat::Native => {
                if cfg!(target_endian = "little") {
                    Endianness::Little
                } else {
                    Endianness::Big
                }
            }
            MachineFormat::LittleEndian => Endianness::Little,
            MachineFormat::BigEndian => Endianness::Big,
        }
    }
}

pub(super) fn parse_machine_format(
    arg: Option<&Value>,
    default_label: &str,
) -> Result<MachineFormat, String> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
        }
        None => machine_format_from_label(default_label),
    }
}

pub(super) fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("machine format must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let collapsed: String = lower
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect();
    if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
        return Ok(MachineFormat::Native);
    }
    if matches!(
        collapsed.as_str(),
        "l" | "le" | "littleendian" | "pc" | "intel"
    ) {
        return Ok(MachineFormat::LittleEndian);
    }
    if matches!(
        collapsed.as_str(),
        "b" | "be" | "bigendian" | "mac" | "motorola"
    ) {
        return Ok(MachineFormat::BigEndian);
    }
    if lower.starts_with("ieee-le") {
        return Ok(MachineFormat::LittleEndian);
    }
    if lower.starts_with("ieee-be") {
        return Ok(MachineFormat::BigEndian);
    }
    Err(format!("unsupported machine format '{trimmed}'"))
}

pub(super) fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

pub(super) fn value_to_scalar(value: &Value, err: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int) => Ok(int.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(err.to_string()),
    }
}
