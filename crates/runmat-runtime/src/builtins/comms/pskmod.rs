//! MATLAB-compatible `pskmod` builtin for M-PSK integer and bit modulation.

use std::f64::consts::TAU;

use runmat_accelerate_api::{
    GpuTensorHandle, ProviderBitModulationRequest, ProviderModulationRequest,
};
use runmat_builtins::{ComplexTensor, LogicalArray, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "pskmod";
const INTEGER_TOL: f64 = 1e-9;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::comms::pskmod")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("modulation"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64, ScalarType::I32, ScalarType::Bool],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("pskmod")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side constellation modulation for integer and bit gpuArray inputs and returns complex-interleaved GPU storage; providers without the hook fall back to host-compatible validation and re-upload.",
};

fn pskmod_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn pskmod_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num) | Some(Type::Int) | Some(Type::Bool) => Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "pskmod",
    category = "comms/modulation",
    summary = "Map integer or bit symbols onto a PSK complex-baseband constellation.",
    keywords = "pskmod,psk,modulation,communications,gray,binary,phaseoffset,gpu",
    type_resolver(pskmod_type),
    builtin_path = "crate::builtins::comms::pskmod"
)]
async fn pskmod_builtin(x: Value, m: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let order = parse_modulation_order(&m)?;
    let options = ParsedOptions::parse(&rest, order)?;
    match x {
        Value::GpuTensor(handle) => pskmod_gpu(handle, order, options).await,
        other => {
            let symbols = SymbolInput::from_value(other, order, options.input_type)?;
            modulate_symbols(symbols, order, &options)
        }
    }
}

async fn pskmod_gpu(
    handle: GpuTensorHandle,
    order: usize,
    options: ParsedOptions,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| pskmod_error("pskmod: no acceleration provider registered"))?;
    let constellation = constellation_table(order, &options)?;
    if runmat_accelerate_api::handle_is_logical(&handle)
        && !matches!(options.input_type, InputType::Bit)
    {
        return Err(pskmod_error("pskmod: logical X requires InputType='bit'"));
    }
    match options.input_type {
        InputType::Integer => {
            let request = ProviderModulationRequest {
                input: &handle,
                constellation: &constellation,
            };
            if let Ok(out) = provider.modulate_constellation(request).await {
                return Ok(gpu_helpers::complex_gpu_value(out));
            }
        }
        InputType::Bit => {
            let input_rows = handle.shape.first().copied().unwrap_or(0);
            if input_rows > 0 {
                let request = ProviderBitModulationRequest {
                    input: &handle,
                    input_rows,
                    bits_per_symbol: bits_per_symbol(order)?,
                    constellation: &constellation,
                };
                if let Ok(out) = provider.modulate_bits_constellation(request).await {
                    return Ok(gpu_helpers::complex_gpu_value(out));
                }
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let symbols = SymbolInput::from_tensor(tensor, order, options.input_type)?;
    let result = modulate_symbols(symbols, order, &options)?;
    let Value::ComplexTensor(tensor) = result else {
        return Err(pskmod_error("pskmod: expected complex modulation output"));
    };
    let out = gpu_helpers::upload_complex_tensor(provider, &tensor)
        .map_err(|err| pskmod_error(format!("pskmod: {err}")))?;
    Ok(gpu_helpers::complex_gpu_value(out))
}

#[derive(Clone, Debug)]
struct ParsedOptions {
    phase_offset: f64,
    mapping: SymbolMapping,
    input_type: InputType,
    output_dtype: OutputDType,
}

#[derive(Clone, Debug)]
enum SymbolMapping {
    Binary,
    Gray,
    Custom(Vec<usize>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputType {
    Integer,
    Bit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputDType {
    Double,
    Single,
}

impl ParsedOptions {
    fn parse(args: &[Value], order: usize) -> BuiltinResult<Self> {
        let mut phase_offset = 0.0;
        let mut mapping = SymbolMapping::Gray;
        let mut input_type = InputType::Integer;
        let mut output_dtype = OutputDType::Double;
        let mut idx = 0usize;

        if let Some(first) = args.first() {
            if !is_name_value_key(first) {
                if is_scalar_numeric(first) {
                    phase_offset = scalar_number(first, "phaseoffset")?;
                } else {
                    mapping = parse_symbol_mapping(first, order)?;
                }
                idx = 1;
            }
        }
        if let Some(second) = args.get(idx) {
            if !is_name_value_key(second) {
                mapping = parse_symbol_mapping(second, order)?;
                idx += 1;
            }
        }

        while idx < args.len() {
            let key = value_as_string(&args[idx])
                .ok_or_else(|| pskmod_error("pskmod: expected name-value option name"))?;
            let Some(value) = args.get(idx + 1) else {
                return Err(pskmod_error(format!(
                    "pskmod: expected value after option '{key}'"
                )));
            };
            match normalize_key(&key).as_str() {
                "inputtype" => {
                    let text = value_as_string(value)
                        .ok_or_else(|| pskmod_error("pskmod: InputType must be a string"))?;
                    input_type = match text.trim().to_ascii_lowercase().as_str() {
                        "integer" => InputType::Integer,
                        "bit" => InputType::Bit,
                        other => {
                            return Err(pskmod_error(format!(
                                "pskmod: unsupported InputType '{other}'"
                            )));
                        }
                    };
                }
                "outputdatatype" => {
                    let dtype = value_as_string(value).ok_or_else(|| {
                        pskmod_error("pskmod: OutputDataType must be 'double' or 'single'")
                    })?;
                    output_dtype = match dtype.trim().to_ascii_lowercase().as_str() {
                        "double" => OutputDType::Double,
                        "single" => OutputDType::Single,
                        _ => {
                            return Err(pskmod_error(
                                "pskmod: only OutputDataType 'double' and 'single' are accepted",
                            ));
                        }
                    };
                }
                "plotconstellation" => {
                    if value_as_bool(value, "PlotConstellation")? {
                        return Err(pskmod_error(
                            "pskmod: PlotConstellation is not implemented in RunMat yet",
                        ));
                    }
                }
                other => {
                    return Err(pskmod_error(format!(
                        "pskmod: unrecognised option '{other}'"
                    )));
                }
            }
            idx += 2;
        }

        if matches!(input_type, InputType::Bit) && !order.is_power_of_two() {
            return Err(pskmod_error(
                "pskmod: InputType='bit' requires M to be a power of two",
            ));
        }

        Ok(Self {
            phase_offset,
            mapping,
            input_type,
            output_dtype,
        })
    }
}

#[derive(Debug)]
struct SymbolInput {
    data: Vec<usize>,
    shape: Vec<usize>,
}

impl SymbolInput {
    fn from_value(value: Value, order: usize, input_type: InputType) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Self::from_tensor(tensor, order, input_type),
            Value::LogicalArray(logical) => Self::from_logical(logical, order, input_type),
            Value::Num(n) => {
                let tensor = Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| pskmod_error(format!("pskmod: {e}")))?;
                Self::from_tensor(tensor, order, input_type)
            }
            Value::Int(i) => {
                let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                    .map_err(|e| pskmod_error(format!("pskmod: {e}")))?;
                Self::from_tensor(tensor, order, input_type)
            }
            Value::Bool(b) => {
                let logical = LogicalArray {
                    data: vec![u8::from(b)],
                    shape: vec![1, 1],
                };
                Self::from_logical(logical, order, input_type)
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err(pskmod_error("pskmod: X must contain real symbols or bits"))
            }
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                Err(pskmod_error("pskmod: X must be numeric or logical"))
            }
            other => Err(pskmod_error(format!(
                "pskmod: unsupported input type {other:?}"
            ))),
        }
    }

    fn from_tensor(tensor: Tensor, order: usize, input_type: InputType) -> BuiltinResult<Self> {
        match input_type {
            InputType::Integer => {
                let data = tensor
                    .data
                    .iter()
                    .map(|&value| number_to_symbol_with_name(value, "X"))
                    .collect::<BuiltinResult<Vec<_>>>()?;
                Ok(Self {
                    data,
                    shape: tensor.shape,
                })
            }
            InputType::Bit => {
                let bits = tensor
                    .data
                    .iter()
                    .map(|&value| number_to_bit(value, "X"))
                    .collect::<BuiltinResult<Vec<_>>>()?;
                bits_to_symbols(bits, tensor.shape, order)
            }
        }
    }

    fn from_logical(
        logical: LogicalArray,
        order: usize,
        input_type: InputType,
    ) -> BuiltinResult<Self> {
        match input_type {
            InputType::Integer => Err(pskmod_error("pskmod: logical X requires InputType='bit'")),
            InputType::Bit => bits_to_symbols(logical.data, logical.shape, order),
        }
    }
}

fn bits_to_symbols(
    bits: Vec<u8>,
    mut shape: Vec<usize>,
    order: usize,
) -> BuiltinResult<SymbolInput> {
    let bits_per_symbol = bits_per_symbol(order)?;
    let rows = shape.first().copied().unwrap_or(bits.len());
    if rows == 0 {
        if shape.is_empty() {
            shape.push(0);
            shape.push(1);
        } else {
            shape[0] = 0;
        }
        return Ok(SymbolInput {
            data: Vec::new(),
            shape,
        });
    }
    if !bits.len().is_multiple_of(rows) {
        return Err(pskmod_error("pskmod: bit input shape is inconsistent"));
    }
    if rows % bits_per_symbol != 0 {
        return Err(pskmod_error(format!(
            "pskmod: number of bit rows must be a multiple of log2(M) ({bits_per_symbol})"
        )));
    }
    let out_rows = rows / bits_per_symbol;
    let channels = if rows == 0 { 0 } else { bits.len() / rows };
    let mut data = Vec::with_capacity(out_rows * channels);
    for channel in 0..channels {
        let channel_offset = channel * rows;
        for group in 0..out_rows {
            let mut symbol = 0usize;
            for bit_idx in 0..bits_per_symbol {
                symbol = (symbol << 1)
                    | usize::from(bits[channel_offset + group * bits_per_symbol + bit_idx]);
            }
            data.push(symbol);
        }
    }
    if shape.is_empty() {
        shape.push(out_rows);
        shape.push(1);
    } else {
        shape[0] = out_rows;
    }
    Ok(SymbolInput { data, shape })
}

fn modulate_symbols(
    symbols: SymbolInput,
    order: usize,
    options: &ParsedOptions,
) -> BuiltinResult<Value> {
    validate_symbol_range(&symbols.data, order)?;
    let constellation = constellation_table(order, options)?;
    let mut out = Vec::with_capacity(symbols.data.len());
    for symbol in symbols.data {
        let point = symbol * 2;
        out.push((constellation[point], constellation[point + 1]));
    }

    let tensor =
        ComplexTensor::new(out, symbols.shape).map_err(|e| pskmod_error(format!("pskmod: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn constellation_table(order: usize, options: &ParsedOptions) -> BuiltinResult<Vec<f64>> {
    let inverse_mapping = match &options.mapping {
        SymbolMapping::Binary => None,
        SymbolMapping::Gray => Some(invert_mapping(&gray_symbol_order(order)?, order)?),
        SymbolMapping::Custom(mapping) => Some(invert_mapping(mapping, order)?),
    };
    let mut table = Vec::with_capacity(order * 2);
    for symbol in 0..order {
        let point_index = inverse_mapping
            .as_ref()
            .map(|inverse| inverse[symbol])
            .unwrap_or(symbol);
        let theta = TAU * point_index as f64 / order as f64 + options.phase_offset;
        let point = match options.output_dtype {
            OutputDType::Double => (theta.cos(), theta.sin()),
            OutputDType::Single => ((theta.cos() as f32) as f64, (theta.sin() as f32) as f64),
        };
        table.push(point.0);
        table.push(point.1);
    }
    Ok(table)
}

fn gray_symbol_order(order: usize) -> BuiltinResult<Vec<usize>> {
    let gray_space = order
        .checked_next_power_of_two()
        .ok_or_else(|| pskmod_error("pskmod: modulation order is too large"))?;
    let bits = bits_per_symbol(gray_space)?;
    let count = 1usize
        .checked_shl(bits as u32)
        .ok_or_else(|| pskmod_error("pskmod: modulation order is too large"))?;
    let mut mapping = Vec::with_capacity(order);
    for index in 0..count {
        let symbol = gray_encode(index);
        if symbol < order {
            mapping.push(symbol);
            if mapping.len() == order {
                return Ok(mapping);
            }
        }
    }
    Err(pskmod_error(
        "pskmod: unable to construct Gray symbol order",
    ))
}

fn gray_encode(value: usize) -> usize {
    value ^ (value >> 1)
}

fn bits_per_symbol(order: usize) -> BuiltinResult<usize> {
    if order < 2 || !order.is_power_of_two() {
        return Err(pskmod_error(
            "pskmod: M must be a power of two for bit grouping",
        ));
    }
    Ok(order.trailing_zeros() as usize)
}

fn parse_modulation_order(value: &Value) -> BuiltinResult<usize> {
    let number = scalar_number(value, "M")?;
    let order = number_to_symbol_with_name(number, "M")?;
    if order < 2 {
        return Err(pskmod_error(
            "pskmod: M must be a positive integer greater than 1",
        ));
    }
    Ok(order)
}

fn parse_symbol_mapping(value: &Value, order: usize) -> BuiltinResult<SymbolMapping> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "gray" => Ok(SymbolMapping::Gray),
            "bin" | "binary" => Ok(SymbolMapping::Binary),
            other => Err(pskmod_error(format!(
                "pskmod: unsupported symbol order '{other}'"
            ))),
        };
    }

    let mapping = vector_to_symbols(value, "symOrder")?;
    if mapping.len() != order {
        return Err(pskmod_error(format!(
            "pskmod: custom symbol order must contain exactly {order} elements"
        )));
    }
    validate_custom_mapping(&mapping, order)?;
    Ok(SymbolMapping::Custom(mapping))
}

fn validate_symbol_range(symbols: &[usize], order: usize) -> BuiltinResult<()> {
    for &symbol in symbols {
        if symbol >= order {
            return Err(pskmod_error(format!(
                "pskmod: input symbols must be in the range [0, {}]",
                order - 1
            )));
        }
    }
    Ok(())
}

fn validate_custom_mapping(mapping: &[usize], order: usize) -> BuiltinResult<()> {
    let mut seen = vec![false; order];
    for &symbol in mapping {
        if symbol >= order {
            return Err(pskmod_error(format!(
                "pskmod: custom symbol order values must be in the range [0, {}]",
                order - 1
            )));
        }
        if seen[symbol] {
            return Err(pskmod_error(
                "pskmod: custom symbol order values must be unique",
            ));
        }
        seen[symbol] = true;
    }
    Ok(())
}

fn invert_mapping(mapping: &[usize], order: usize) -> BuiltinResult<Vec<usize>> {
    let mut inverse = vec![0usize; order];
    for (point_index, &symbol) in mapping.iter().enumerate() {
        inverse[symbol] = point_index;
    }
    Ok(inverse)
}

fn vector_to_symbols(value: &Value, name: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|&number| number_to_symbol_with_name(number, name))
            .collect(),
        Value::LogicalArray(logical) => Ok(logical.data.iter().map(|&v| usize::from(v)).collect()),
        Value::Num(n) => Ok(vec![number_to_symbol_with_name(*n, name)?]),
        Value::Int(i) => Ok(vec![number_to_symbol_with_name(i.to_f64(), name)?]),
        Value::Bool(b) => Ok(vec![usize::from(*b)]),
        other => Err(pskmod_error(format!(
            "pskmod: {name} must be a numeric vector, got {other:?}"
        ))),
    }
}

fn scalar_number(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(pskmod_error(format!("pskmod: {name} must be a scalar"))),
    }
}

fn is_scalar_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor.data.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

fn number_to_symbol_with_name(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(pskmod_error(format!(
            "pskmod: {name} values must be finite integers"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > INTEGER_TOL || rounded < 0.0 {
        return Err(pskmod_error(format!(
            "pskmod: {name} values must be nonnegative integers"
        )));
    }
    if rounded > usize::MAX as f64 {
        return Err(pskmod_error(format!(
            "pskmod: {name} value is too large for this platform"
        )));
    }
    Ok(rounded as usize)
}

fn number_to_bit(value: f64, name: &str) -> BuiltinResult<u8> {
    let symbol = number_to_symbol_with_name(value, name)?;
    match symbol {
        0 | 1 => Ok(symbol as u8),
        _ => Err(pskmod_error(format!(
            "pskmod: {name} bit inputs must contain only 0 or 1"
        ))),
    }
}

fn value_as_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    let n = scalar_number(value, name)?;
    if (n - 0.0).abs() <= INTEGER_TOL {
        Ok(false)
    } else if (n - 1.0).abs() <= INTEGER_TOL {
        Ok(true)
    } else {
        Err(pskmod_error(format!(
            "pskmod: {name} must be logical or numeric 0/1"
        )))
    }
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(|c| *c != '_' && *c != '-' && !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_name_value_key(value: &Value) -> bool {
    let Some(text) = value_as_string(value) else {
        return false;
    };
    matches!(
        normalize_key(&text).as_str(),
        "inputtype" | "outputdatatype" | "plotconstellation"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn pskmod(x: Value, order: usize, rest: Vec<Value>) -> ComplexTensor {
        match block_on(super::pskmod_builtin(x, Value::Num(order as f64), rest)).expect("pskmod") {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected ComplexTensor, got {other:?}"),
        }
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).expect("tensor"))
    }

    fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, ((ar, ai), (er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (ar - er).abs() < 1e-12 && (ai - ei).abs() < 1e-12,
                "at {idx}: expected ({er}, {ei}), got ({ar}, {ai})"
            );
        }
    }

    #[test]
    fn pskmod_bpsk_gray_matches_binary() {
        let out = pskmod(tensor(vec![0.0, 1.0], vec![1, 2]), 2, vec![]);
        assert_eq!(out.shape, vec![1, 2]);
        assert_complex_close(&out.data, &[(1.0, 0.0), (-1.0, 0.0)]);
    }

    #[test]
    fn pskmod_qpsk_gray_mapping() {
        let out = pskmod(tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]), 4, vec![]);
        assert_complex_close(
            &out.data,
            &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
        );
    }

    #[test]
    fn pskmod_8psk_gray_mapping() {
        let out = pskmod(
            tensor((0..8).map(|v| v as f64).collect(), vec![1, 8]),
            8,
            vec![],
        );
        let expected_indices = [0usize, 1, 3, 2, 7, 6, 4, 5];
        let expected = expected_indices
            .iter()
            .map(|idx| {
                let theta = TAU * *idx as f64 / 8.0;
                (theta.cos(), theta.sin())
            })
            .collect::<Vec<_>>();
        assert_complex_close(&out.data, &expected);
    }

    #[test]
    fn pskmod_binary_order_is_counterclockwise() {
        let out = pskmod(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
            4,
            vec![Value::Num(0.0), Value::from("bin")],
        );
        assert_complex_close(
            &out.data,
            &[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
        );
    }

    #[test]
    fn pskmod_phase_offset_rotates_constellation() {
        let out = pskmod(
            tensor(vec![0.0, 1.0], vec![1, 2]),
            2,
            vec![Value::Num(std::f64::consts::FRAC_PI_2), Value::from("bin")],
        );
        assert_complex_close(&out.data, &[(0.0, 1.0), (0.0, -1.0)]);
    }

    #[test]
    fn pskmod_custom_mapping_uses_constellation_labels() {
        let mapping = tensor(vec![0.0, 3.0, 1.0, 2.0], vec![1, 4]);
        let out = pskmod(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
            4,
            vec![Value::Num(0.0), mapping],
        );
        assert_complex_close(
            &out.data,
            &[(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0)],
        );
    }

    #[test]
    fn pskmod_bit_input_groups_rows_by_channel() {
        let bits = tensor(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
        let out = pskmod(bits, 4, vec![Value::from("InputType"), Value::from("bit")]);
        assert_eq!(out.shape, vec![2, 2]);
        assert_complex_close(
            &out.data,
            &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
        );
    }

    #[test]
    fn pskmod_output_datatype_single_rounds_samples() {
        let out = pskmod(
            tensor(vec![1.0], vec![1, 1]),
            8,
            vec![Value::from("OutputDataType"), Value::from("single")],
        );
        let theta = TAU / 8.0;
        assert_eq!(out.data[0].0, (theta.cos() as f32) as f64);
        assert_eq!(out.data[0].1, (theta.sin() as f32) as f64);
    }

    #[test]
    fn pskmod_rejects_out_of_range_symbols() {
        let err = block_on(super::pskmod_builtin(
            tensor(vec![0.0, 8.0], vec![1, 2]),
            Value::Num(8.0),
            vec![],
        ))
        .expect_err("expected out-of-range error");
        assert!(err.to_string().contains("range [0, 7]"));
    }

    #[test]
    fn pskmod_rejects_default_logical_input() {
        let err = block_on(super::pskmod_builtin(
            Value::LogicalArray(LogicalArray {
                data: vec![0, 1],
                shape: vec![1, 2],
            }),
            Value::Num(2.0),
            vec![],
        ))
        .expect_err("expected logical input error");
        assert!(err.to_string().contains("InputType='bit'"));
    }

    #[test]
    fn pskmod_accepts_logical_bit_input() {
        let out = pskmod(
            Value::LogicalArray(LogicalArray {
                data: vec![0, 0, 1, 1],
                shape: vec![2, 2],
            }),
            4,
            vec![Value::from("InputType"), Value::from("bit")],
        );
        assert_eq!(out.shape, vec![1, 2]);
        assert_complex_close(&out.data, &[(1.0, 0.0), (-1.0, 0.0)]);
    }

    #[test]
    fn pskmod_inprocess_gpu_bit_input_returns_complex_resident_output() {
        use runmat_accelerate_api::{GpuTensorStorage, HostTensorView};

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
            let expected = pskmod(
                Value::Tensor(input.clone()),
                4,
                vec![Value::from("InputType"), Value::from("bit")],
            );
            let input_handle = provider
                .upload(&HostTensorView {
                    data: &input.data,
                    shape: &input.shape,
                })
                .expect("upload");
            let result = block_on(super::pskmod_builtin(
                Value::GpuTensor(input_handle.clone()),
                Value::Num(4.0),
                vec![Value::from("InputType"), Value::from("bit")],
            ))
            .expect("gpu pskmod bit");
            let Value::GpuTensor(output_handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(output_handle.clone()),
            ))
            .expect("gather output");
            let Value::ComplexTensor(actual) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(actual.shape, expected.shape);
            assert_complex_close(&actual.data, &expected.data);
            provider.free(&input_handle).ok();
            provider.free(&output_handle).ok();
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn pskmod_wgpu_input_returns_complex_resident_output() {
        use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView};

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap();
                let expected = pskmod(Value::Tensor(input.clone()), 4, vec![]);
                let view = HostTensorView {
                    data: &input.data,
                    shape: &input.shape,
                };
                let input_handle = provider.upload(&view).expect("upload");
                let result = block_on(super::pskmod_builtin(
                    Value::GpuTensor(input_handle.clone()),
                    Value::Num(4.0),
                    vec![],
                ))
                .expect("gpu pskmod");
                let Value::GpuTensor(output_handle) = result else {
                    panic!("expected resident gpu output");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_storage(&output_handle),
                    GpuTensorStorage::ComplexInterleaved
                );
                let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                    &Value::GpuTensor(output_handle.clone()),
                ))
                .expect("gather output");
                let Value::ComplexTensor(actual) = gathered else {
                    panic!("expected gathered complex tensor");
                };
                assert_eq!(actual.shape, expected.shape);
                assert_complex_close(&actual.data, &expected.data);
                provider.free(&input_handle).ok();
                provider.free(&output_handle).ok();
            }
            Err(err) => {
                tracing::warn!("Skipping pskmod_wgpu_input_returns_complex_resident_output: {err}");
            }
        }
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn pskmod_wgpu_bit_input_returns_complex_resident_output() {
        use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView};

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                let input = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
                let expected = pskmod(
                    Value::Tensor(input.clone()),
                    4,
                    vec![Value::from("InputType"), Value::from("bit")],
                );
                let view = HostTensorView {
                    data: &input.data,
                    shape: &input.shape,
                };
                let input_handle = provider.upload(&view).expect("upload");
                let result = block_on(super::pskmod_builtin(
                    Value::GpuTensor(input_handle.clone()),
                    Value::Num(4.0),
                    vec![Value::from("InputType"), Value::from("bit")],
                ))
                .expect("gpu pskmod bit");
                let Value::GpuTensor(output_handle) = result else {
                    panic!("expected resident gpu output");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_storage(&output_handle),
                    GpuTensorStorage::ComplexInterleaved
                );
                let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                    &Value::GpuTensor(output_handle.clone()),
                ))
                .expect("gather output");
                let Value::ComplexTensor(actual) = gathered else {
                    panic!("expected gathered complex tensor");
                };
                assert_eq!(actual.shape, expected.shape);
                assert_complex_close(&actual.data, &expected.data);
                provider.free(&input_handle).ok();
                provider.free(&output_handle).ok();
            }
            Err(err) => {
                tracing::warn!(
                    "Skipping pskmod_wgpu_bit_input_returns_complex_resident_output: {err}"
                );
            }
        }
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    #[test]
    fn pskmod_rejects_plot_constellation_true() {
        let err = block_on(super::pskmod_builtin(
            Value::Num(0.0),
            Value::Num(2.0),
            vec![Value::from("PlotConstellation"), Value::Bool(true)],
        ))
        .expect_err("expected plot error");
        assert!(err.to_string().contains("PlotConstellation"));
    }
}
