use super::*;
use crate::builtins::common::tensor;
use runmat_builtins::{parse_fread_precision, FreadOutputKind, FreadPrecisionSpec};

pub(super) use runmat_builtins::FreadInputSource as InputSource;
pub(super) use runmat_builtins::FreadInputType as InputType;

#[derive(Clone, Copy, Debug)]
pub(super) enum OutputKind {
    Numeric(NumericDType),
    Char,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PrecisionSpec {
    pub(super) input: InputSource,
    pub(super) output: OutputKind,
    pub(super) repeat: usize,
}

impl Default for PrecisionSpec {
    fn default() -> Self {
        Self::from_contract(FreadPrecisionSpec::default())
    }
}

impl PrecisionSpec {
    fn from_contract(contract: FreadPrecisionSpec) -> Self {
        let output = match contract.output {
            FreadOutputKind::Char => OutputKind::Char,
            FreadOutputKind::Numeric(class) => OutputKind::Numeric(NumericDType::from(class)),
        };
        Self {
            input: contract.input,
            output,
            repeat: contract.repeat,
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
    Ok(PrecisionSpec::from_contract(parse_fread_precision(raw)?))
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
    if rounded >= i64::MAX as f64 || rounded >= usize::MAX as f64 + 1.0 {
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
    extract_scalar_string(value).ok_or_else(|| err.to_string())
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
