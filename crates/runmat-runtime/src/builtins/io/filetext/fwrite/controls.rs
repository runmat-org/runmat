use super::*;

#[derive(Clone, Copy)]
pub(super) enum ControlRole {
    FileId,
    Skip,
}

pub(super) fn preflight_control(value: &Value, role: ControlRole) -> BuiltinResult<()> {
    let integer_extension = match role {
        ControlRole::FileId => &FWRITE_INTEGER_ID_EXTENSION,
        ControlRole::Skip => &FWRITE_INTEGER_SKIP_EXTENSION,
    };
    match value {
        Value::Int(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FWRITE_LOGICAL_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FWRITE_SINGLE_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FWRITE_RESIDENT_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FWRITE_LOGICAL_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    integer_extension,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FWRITE_SINGLE_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(super) fn parse_fid(value: &Value) -> Result<i32, String> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(int) => {
            return int
                .try_to_i32()
                .ok_or_else(|| "fwrite: file identifier is out of range".to_string());
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int
                    .try_to_i32()
                    .ok_or_else(|| "fwrite: file identifier is out of range".to_string());
            }
            tensor::tensor_value_f64(t, 0)
        }
        _ => return Err("fwrite: file identifier must be numeric".to_string()),
    };
    if !scalar.is_finite() {
        return Err("fwrite: file identifier must be finite".to_string());
    }
    if scalar.fract().abs() > f64::EPSILON {
        return Err("fwrite: file identifier must be an integer".to_string());
    }
    if scalar < i32::MIN as f64 || scalar > i32::MAX as f64 {
        return Err("fwrite: file identifier is out of range".to_string());
    }
    Ok(scalar as i32)
}

pub(super) type FwriteArgs<'a> = (Option<&'a Value>, Option<&'a Value>, Option<&'a Value>);

pub(super) fn classify_arguments<'a>(
    args: crate::call::FunctionArgumentView<'a>,
) -> Result<FwriteArgs<'a>, String> {
    match args.len() {
        0 => Ok((None, None, None)),
        1 => {
            if is_string_like(&args[0]) {
                Ok((
                    Some(args.get(0).expect("validated fwrite control arity")),
                    None,
                    None,
                ))
            } else {
                Err(
                    "fwrite: precision argument must be a string scalar or character vector"
                        .to_string(),
                )
            }
        }
        2 => {
            if !is_string_like(&args[0]) {
                return Err(
                    "fwrite: precision argument must be a string scalar or character vector"
                        .to_string(),
                );
            }
            if is_numeric_like(&args[1]) {
                Ok((
                    Some(args.get(0).expect("validated fwrite control arity")),
                    Some(args.get(1).expect("validated fwrite control arity")),
                    None,
                ))
            } else if is_string_like(&args[1]) {
                Ok((
                    Some(args.get(0).expect("validated fwrite control arity")),
                    None,
                    Some(args.get(1).expect("validated fwrite control arity")),
                ))
            } else {
                Err("fwrite: invalid argument combination (expected numeric skip or machine format string)".to_string())
            }
        }
        3 => {
            if !is_string_like(&args[0]) || !is_numeric_like(&args[1]) || !is_string_like(&args[2])
            {
                return Err("fwrite: expected arguments (precision, skip, machinefmt)".to_string());
            }
            Ok((
                Some(args.get(0).expect("validated fwrite control arity")),
                Some(args.get(1).expect("validated fwrite control arity")),
                Some(args.get(2).expect("validated fwrite control arity")),
            ))
        }
        _ => Err("fwrite: too many input arguments".to_string()),
    }
}

pub(super) fn is_string_like(value: &Value) -> bool {
    match value {
        Value::String(runmat_value::StringElement::Text(_)) => true,
        Value::CharArray(ca) => ca.rows == 1,
        Value::StringArray(sa) => sa.data().len() == 1,
        _ => false,
    }
}

pub(super) fn is_numeric_like(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(t) => tensor::is_scalar_tensor(t),
        Value::LogicalArray(la) => la.data.len() == 1,
        Value::GpuTensor(handle) => handle.shape().iter().product::<usize>() == 1,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct WriteSpec {
    pub(super) input: InputType,
}

impl WriteSpec {
    fn default() -> Self {
        Self {
            input: InputType::UInt8,
        }
    }
}

pub(super) fn parse_precision(arg: Option<&Value>) -> BuiltinResult<WriteSpec> {
    match arg {
        None => Ok(WriteSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "fwrite: precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
                .map_err(|detail| fwrite_error_with_detail(&FWRITE_ERROR_INVALID_OPTION, detail))
        }
    }
}

pub(super) fn parse_precision_string(raw: &str) -> Result<WriteSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("fwrite: precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some((lhs, rhs)) = lower.split_once("=>") {
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        let input = parse_input_label(lhs)?;
        let output = parse_input_label(rhs)?;
        if input != output {
            return Err(
                "fwrite: differing input/output precisions are not implemented yet".to_string(),
            );
        }
        Ok(WriteSpec { input })
    } else {
        parse_input_label(lower.trim()).map(|input| WriteSpec { input })
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
            let scalar = numeric_scalar(value, "fwrite: skip must be numeric")?;
            parse_skip_scalar(scalar)
        }
    }
}

pub(super) fn parse_skip_scalar(scalar: f64) -> Result<usize, String> {
    if !scalar.is_finite() {
        return Err("fwrite: skip value must be finite".to_string());
    }
    if scalar < 0.0 {
        return Err("fwrite: skip value must be non-negative".to_string());
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > f64::EPSILON {
        return Err("fwrite: skip value must be an integer".to_string());
    }
    if rounded >= i64::MAX as f64 {
        return Err("fwrite: skip value is too large".to_string());
    }
    Ok(rounded as usize)
}

pub(super) fn int_to_skip(value: &IntValue) -> Result<usize, String> {
    let Some(skip) = value.try_to_usize() else {
        return Err("fwrite: skip value must be non-negative".to_string());
    };
    if skip > i64::MAX as usize {
        return Err("fwrite: skip value is too large".to_string());
    }
    Ok(skip)
}

#[derive(Clone, Copy, Debug)]
pub(super) enum MachineFormat {
    Native,
    LittleEndian,
    BigEndian,
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

#[derive(Clone, Copy, Debug)]
pub(super) enum Endianness {
    Little,
    Big,
}

pub(super) fn parse_machine_format(
    arg: Option<&Value>,
    default_label: &str,
) -> BuiltinResult<MachineFormat> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "fwrite: machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
                .map_err(|detail| fwrite_error_with_detail(&FWRITE_ERROR_INVALID_OPTION, detail))
        }
        None => machine_format_from_label(default_label)
            .map_err(|detail| fwrite_error_with_detail(&FWRITE_ERROR_INVALID_OPTION, detail)),
    }
}

pub(super) fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("fwrite: machine format must not be empty".to_string());
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
    Err(format!("fwrite: unsupported machine format '{trimmed}'"))
}

pub(super) fn scalar_string(value: &Value, err: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                fwrite_error_with_detail(&FWRITE_ERROR_INVALID_OPTION, err)
            })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        _ => Err(fwrite_error_with_detail(&FWRITE_ERROR_INVALID_OPTION, err)),
    }
}

pub(super) fn numeric_scalar(value: &Value, err: &str) -> Result<f64, String> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub(super) const fn byte_len(self) -> usize {
        match self {
            Self::UInt8 | Self::Int8 => 1,
            Self::UInt16 | Self::Int16 => 2,
            Self::UInt32 | Self::Int32 | Self::Float32 => 4,
            Self::UInt64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

pub(super) fn parse_input_label(label: &str) -> Result<InputType, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(InputType::Float64),
        "single" | "float32" | "real*4" => Ok(InputType::Float32),
        "int8" | "schar" | "integer*1" => Ok(InputType::Int8),
        "uint8" | "uchar" | "unsignedchar" | "char" | "byte" => Ok(InputType::UInt8),
        "int16" | "short" | "integer*2" => Ok(InputType::Int16),
        "uint16" | "ushort" | "unsignedshort" => Ok(InputType::UInt16),
        "int32" | "integer*4" | "long" => Ok(InputType::Int32),
        "uint32" | "unsignedint" | "unsignedlong" => Ok(InputType::UInt32),
        "int64" | "integer*8" | "longlong" => Ok(InputType::Int64),
        "uint64" | "unsignedlonglong" => Ok(InputType::UInt64),
        other => Err(format!("fwrite: unsupported precision '{other}'")),
    }
}
