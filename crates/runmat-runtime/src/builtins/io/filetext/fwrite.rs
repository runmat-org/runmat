//! MATLAB-compatible `fwrite` builtin for RunMat.
use std::io::{Seek, SeekFrom, Write};

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::gather_if_needed;
use runmat_filesystem::File;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fwrite",
    op_kind: GpuOpKind::Custom("file-io-write"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only binary file I/O; GPU arguments are gathered to the CPU prior to writing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is never fused; metadata recorded for completeness.",
};

#[runtime_builtin(
    name = "fwrite",
    category = "io/filetext",
    summary = "Write binary data to a file identifier.",
    keywords = "fwrite,file,io,binary,precision",
    accel = "cpu",
    builtin_path = "crate::builtins::io::filetext::fwrite"
)]
fn fwrite_builtin(fid: Value, data: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&fid, &data, &rest)?;
    Ok(Value::Num(eval.count as f64))
}

/// Result of an `fwrite` evaluation.
#[derive(Debug, Clone)]
pub struct FwriteEval {
    count: usize,
}

impl FwriteEval {
    fn new(count: usize) -> Self {
        Self { count }
    }

    /// Number of elements successfully written.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Evaluate the `fwrite` builtin without invoking the runtime dispatcher.
pub fn evaluate(
    fid_value: &Value,
    data_value: &Value,
    rest: &[Value],
) -> Result<FwriteEval, String> {
    let fid_host = gather_value(fid_value)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err("fwrite: file identifier must be non-negative".to_string());
    }
    if fid < 3 {
        return Err("fwrite: standard input/output identifiers are not supported yet".to_string());
    }

    let info = registry::info_for(fid).ok_or_else(|| {
        "fwrite: Invalid file identifier. Use fopen to generate a valid file ID.".to_string()
    })?;
    let handle = registry::take_handle(fid).ok_or_else(|| {
        "fwrite: Invalid file identifier. Use fopen to generate a valid file ID.".to_string()
    })?;

    let mut file = handle
        .lock()
        .map_err(|_| "fwrite: failed to lock file handle (poisoned mutex)".to_string())?;

    let data_host = gather_value(data_value)?;
    let rest_host = gather_args(rest)?;
    let (precision_arg, skip_arg, machine_arg) = classify_arguments(&rest_host)?;

    let precision_spec = parse_precision(precision_arg)?;
    let skip_bytes = parse_skip(skip_arg)?;
    let machine_format = parse_machine_format(machine_arg, &info.machinefmt)?;

    let elements = flatten_elements(&data_host)?;
    let count = write_elements(
        &mut file,
        &elements,
        precision_spec,
        skip_bytes,
        machine_format,
    )?;
    Ok(FwriteEval::new(count))
}

fn gather_value(value: &Value) -> Result<Value, String> {
    gather_if_needed(value).map_err(|e| format!("fwrite: {e}"))
}

fn gather_args(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed(value).map_err(|e| format!("fwrite: {e}"))?);
    }
    Ok(gathered)
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(int) => int.to_f64(),
        _ => return Err("fwrite: file identifier must be numeric".to_string()),
    };
    if !scalar.is_finite() {
        return Err("fwrite: file identifier must be finite".to_string());
    }
    if scalar.fract().abs() > f64::EPSILON {
        return Err("fwrite: file identifier must be an integer".to_string());
    }
    Ok(scalar as i32)
}

type FwriteArgs<'a> = (Option<&'a Value>, Option<&'a Value>, Option<&'a Value>);

fn classify_arguments(args: &[Value]) -> Result<FwriteArgs<'_>, String> {
    match args.len() {
        0 => Ok((None, None, None)),
        1 => {
            if is_string_like(&args[0]) {
                Ok((Some(&args[0]), None, None))
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
                Ok((Some(&args[0]), Some(&args[1]), None))
            } else if is_string_like(&args[1]) {
                Ok((Some(&args[0]), None, Some(&args[1])))
            } else {
                Err("fwrite: invalid argument combination (expected numeric skip or machine format string)".to_string())
            }
        }
        3 => {
            if !is_string_like(&args[0]) || !is_numeric_like(&args[1]) || !is_string_like(&args[2])
            {
                return Err("fwrite: expected arguments (precision, skip, machinefmt)".to_string());
            }
            Ok((Some(&args[0]), Some(&args[1]), Some(&args[2])))
        }
        _ => Err("fwrite: too many input arguments".to_string()),
    }
}

fn is_string_like(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::CharArray(ca) => ca.rows == 1,
        Value::StringArray(sa) => sa.data.len() == 1,
        _ => false,
    }
}

fn is_numeric_like(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(la) => la.data.len() == 1,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
struct WriteSpec {
    input: InputType,
}

impl WriteSpec {
    fn default() -> Self {
        Self {
            input: InputType::UInt8,
        }
    }
}

fn parse_precision(arg: Option<&Value>) -> Result<WriteSpec, String> {
    match arg {
        None => Ok(WriteSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "fwrite: precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
        }
    }
}

fn parse_precision_string(raw: &str) -> Result<WriteSpec, String> {
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

fn parse_skip(arg: Option<&Value>) -> Result<usize, String> {
    match arg {
        None => Ok(0),
        Some(value) => {
            let scalar = numeric_scalar(value, "fwrite: skip must be numeric")?;
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
            if rounded > i64::MAX as f64 {
                return Err("fwrite: skip value is too large".to_string());
            }
            Ok(rounded as usize)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum MachineFormat {
    Native,
    LittleEndian,
    BigEndian,
}

impl MachineFormat {
    fn to_endianness(self) -> Endianness {
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
enum Endianness {
    Little,
    Big,
}

fn parse_machine_format(arg: Option<&Value>, default_label: &str) -> Result<MachineFormat, String> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "fwrite: machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
        }
        None => machine_format_from_label(default_label),
    }
}

fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
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

fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

fn numeric_scalar(value: &Value, err: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int) => Ok(int.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(err.to_string()),
    }
}

fn flatten_elements(value: &Value) -> Result<Vec<f64>, String> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(int) => Ok(vec![int.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|bit| if *bit != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::CharArray(ca) => Ok(flatten_char_array(ca)),
        Value::String(text) => Ok(text.chars().map(|ch| ch as u32 as f64).collect()),
        Value::StringArray(sa) => Ok(flatten_string_array(sa)),
        Value::GpuTensor(_) => Err("fwrite: expected host tensor data after gathering".to_string()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("fwrite: complex values are not supported yet".to_string())
        }
        _ => Err(format!("fwrite: unsupported data type {:?}", value)),
    }
}

fn flatten_char_array(ca: &CharArray) -> Vec<f64> {
    let mut values = Vec::with_capacity(ca.rows.saturating_mul(ca.cols));
    for c in 0..ca.cols {
        for r in 0..ca.rows {
            let idx = r * ca.cols + c;
            values.push(ca.data[idx] as u32 as f64);
        }
    }
    values
}

fn flatten_string_array(sa: &runmat_builtins::StringArray) -> Vec<f64> {
    if sa.data.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::new();
    for (idx, text) in sa.data.iter().enumerate() {
        if idx > 0 {
            values.push('\n' as u32 as f64);
        }
        values.extend(text.chars().map(|ch| ch as u32 as f64));
    }
    values
}

fn write_elements(
    file: &mut File,
    values: &[f64],
    spec: WriteSpec,
    skip: usize,
    machine: MachineFormat,
) -> Result<usize, String> {
    let endianness = machine.to_endianness();
    let skip_offset = skip as i64;
    for &value in values {
        match spec.input {
            InputType::UInt8 => {
                let byte = to_u8(value);
                write_bytes(file, &[byte])?;
            }
            InputType::Int8 => {
                let byte = to_i8(value) as u8;
                write_bytes(file, &[byte])?;
            }
            InputType::UInt16 => {
                let bytes = encode_u16(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Int16 => {
                let bytes = encode_i16(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::UInt32 => {
                let bytes = encode_u32(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Int32 => {
                let bytes = encode_i32(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::UInt64 => {
                let bytes = encode_u64(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Int64 => {
                let bytes = encode_i64(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Float32 => {
                let bytes = encode_f32(value, endianness);
                write_bytes(file, &bytes)?;
            }
            InputType::Float64 => {
                let bytes = encode_f64(value, endianness);
                write_bytes(file, &bytes)?;
            }
        }

        if skip > 0 {
            file.seek(SeekFrom::Current(skip_offset))
                .map_err(|err| format!("fwrite: failed to seek while applying skip ({err})"))?;
        }
    }
    Ok(values.len())
}

fn write_bytes(file: &mut File, bytes: &[u8]) -> Result<(), String> {
    file.write_all(bytes)
        .map_err(|err| format!("fwrite: failed to write to file ({err})"))
}

fn to_u8(value: f64) -> u8 {
    if !value.is_finite() {
        return if value.is_sign_negative() { 0 } else { u8::MAX };
    }
    let mut rounded = value.round();
    if rounded.is_nan() {
        return 0;
    }
    if rounded < 0.0 {
        rounded = 0.0;
    }
    if rounded > u8::MAX as f64 {
        rounded = u8::MAX as f64;
    }
    rounded as u8
}

fn to_i8(value: f64) -> i8 {
    saturating_round(value, i8::MIN as f64, i8::MAX as f64) as i8
}

fn encode_u16(value: f64, endianness: Endianness) -> [u8; 2] {
    let rounded = saturating_round(value, 0.0, u16::MAX as f64) as u16;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_i16(value: f64, endianness: Endianness) -> [u8; 2] {
    let rounded = saturating_round(value, i16::MIN as f64, i16::MAX as f64) as i16;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_u32(value: f64, endianness: Endianness) -> [u8; 4] {
    let rounded = saturating_round(value, 0.0, u32::MAX as f64) as u32;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_i32(value: f64, endianness: Endianness) -> [u8; 4] {
    let rounded = saturating_round(value, i32::MIN as f64, i32::MAX as f64) as i32;
    match endianness {
        Endianness::Little => rounded.to_le_bytes(),
        Endianness::Big => rounded.to_be_bytes(),
    }
}

fn encode_u64(value: f64, endianness: Endianness) -> [u8; 8] {
    let rounded = saturating_round(value, 0.0, u64::MAX as f64);
    let as_u64 = if rounded.is_finite() {
        rounded as u64
    } else if rounded.is_sign_negative() {
        0
    } else {
        u64::MAX
    };
    match endianness {
        Endianness::Little => as_u64.to_le_bytes(),
        Endianness::Big => as_u64.to_be_bytes(),
    }
}

fn encode_i64(value: f64, endianness: Endianness) -> [u8; 8] {
    let rounded = saturating_round(value, i64::MIN as f64, i64::MAX as f64);
    let as_i64 = if rounded.is_finite() {
        rounded as i64
    } else if rounded.is_sign_negative() {
        i64::MIN
    } else {
        i64::MAX
    };
    match endianness {
        Endianness::Little => as_i64.to_le_bytes(),
        Endianness::Big => as_i64.to_be_bytes(),
    }
}

fn encode_f32(value: f64, endianness: Endianness) -> [u8; 4] {
    let as_f32 = value as f32;
    let bits = as_f32.to_bits();
    match endianness {
        Endianness::Little => bits.to_le_bytes(),
        Endianness::Big => bits.to_be_bytes(),
    }
}

fn encode_f64(value: f64, endianness: Endianness) -> [u8; 8] {
    let bits = value.to_bits();
    match endianness {
        Endianness::Little => bits.to_le_bytes(),
        Endianness::Big => bits.to_be_bytes(),
    }
}

fn saturating_round(value: f64, min: f64, max: f64) -> f64 {
    if !value.is_finite() {
        return if value.is_sign_negative() { min } else { max };
    }
    let mut rounded = value.round();
    if rounded.is_nan() {
        return 0.0;
    }
    if rounded < min {
        rounded = min;
    }
    if rounded > max {
        rounded = max;
    }
    rounded
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputType {
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

fn parse_input_label(label: &str) -> Result<InputType, String> {
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::registry;
    use crate::builtins::io::filetext::{fclose, fopen};
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::Tensor;
    use runmat_filesystem::{self as fs, File};
    use runmat_time::system_time_now;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_default_uint8_bytes() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_uint8");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0, 2.0, 255.0], vec![3, 1]).unwrap();
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &Vec::new()).expect("fwrite");
        assert_eq!(eval.count(), 3);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![1u8, 2, 255]);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_double_precision_writes_native_endian() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_double");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.5, -2.25], vec![2, 1]).unwrap();
        let args = vec![Value::from("double")];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 2);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        let expected: Vec<u8> = if cfg!(target_endian = "little") {
            [1.5f64.to_le_bytes(), (-2.25f64).to_le_bytes()].concat()
        } else {
            [1.5f64.to_be_bytes(), (-2.25f64).to_be_bytes()].concat()
        };
        assert_eq!(bytes, expected);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_big_endian_uint16() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_be");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![258.0, 772.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("uint16")];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 2);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![0x01, 0x02, 0x03, 0x04]);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_skip_inserts_padding() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_skip");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
        let args = vec![Value::from("uint8"), Value::Num(1.0)];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
        assert_eq!(eval.count(), 3);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![10u8, 0, 20, 0, 30]);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_gpu_tensor_gathers_before_write() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_gpu");

        test_support::with_test_provider(|provider| {
            registry::reset_for_tests();
            let open = fopen::evaluate(&[
                Value::from(path.to_string_lossy().to_string()),
                Value::from("w+b"),
            ])
            .expect("fopen");
            let fid = open.as_open().unwrap().fid as i32;

            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::from("uint16")];
            let eval = evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args)
                .expect("fwrite");
            assert_eq!(eval.count(), 4);

            fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        });

        let mut file = File::open(&path).expect("open");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("read");
        assert_eq!(bytes.len(), 8);
        let mut decoded = Vec::new();
        for chunk in bytes.chunks_exact(2) {
            let value = if cfg!(target_endian = "little") {
                u16::from_le_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], chunk[1]])
            };
            decoded.push(value);
        }
        assert_eq!(decoded, vec![1u16, 2, 3, 4]);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_invalid_precision_errors() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_invalid_precision");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let args = vec![Value::from("bogus-class")];
        let err = evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err();
        assert!(err.contains("unsupported precision"));
        let _ = fclose::evaluate(&[Value::Num(fid as f64)]);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_negative_skip_errors() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_negative_skip");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![10.0], vec![1, 1]).unwrap();
        let args = vec![Value::from("uint8"), Value::Num(-1.0)];
        let err = evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err();
        assert!(err.contains("skip value must be non-negative"));
        let _ = fclose::evaluate(&[Value::Num(fid as f64)]);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fwrite_wgpu_tensor_roundtrip() {
        registry::reset_for_tests();
        let path = unique_path("fwrite_wgpu_roundtrip");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let provider = provider::register_wgpu_provider(provider::WgpuProviderOptions::default())
            .expect("wgpu provider");

        let tensor = Tensor::new(vec![0.5, -1.25, 3.75], vec![3, 1]).unwrap();
        let expected = tensor.data.clone();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload to gpu");
        let args = vec![Value::from("double")];
        let eval =
            evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args).expect("fwrite");
        assert_eq!(eval.count(), 3);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let mut file = File::open(&path).expect("open");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("read");
        assert_eq!(bytes.len(), 24);
        for (chunk, expected_value) in bytes.chunks_exact(8).zip(expected.iter()) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            let value = if cfg!(target_endian = "little") {
                f64::from_le_bytes(buf)
            } else {
                f64::from_be_bytes(buf)
            };
            assert!(
                (value - expected_value).abs() < 1e-12,
                "mismatch: {} vs {}",
                value,
                expected_value
            );
        }
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fwrite_invalid_identifier_errors() {
        registry::reset_for_tests();
        let err = evaluate(&Value::Num(-1.0), &Value::Num(1.0), &Vec::new()).unwrap_err();
        assert!(err.contains("file identifier must be non-negative"));
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!(
            "runmat_{prefix}_{}_{}.tmp",
            now.as_secs(),
            now.subsec_nanos()
        );
        std::env::temp_dir().join(filename)
    }
}
