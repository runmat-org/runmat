//! MATLAB-compatible `pskmod` builtin for M-PSK integer and bit modulation.

use runmat_value::NumericDType;
use std::f64::consts::TAU;

use runmat_accelerate_api::{
    GpuTensorHandle, IntegerElementType, ProviderBitModulationRequest, ProviderModulationRequest,
    ProviderPrecision,
};
use runmat_builtins::{
    PSKMOD_INTEGER_CUSTOM_ORDER_EXTENSION, PSKMOD_INTEGER_ORDER_EXTENSION,
    PSKMOD_INTEGER_PHASE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntValue, IntegerStorage, LogicalArray, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "pskmod";

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

#[runtime_builtin(
    name = "pskmod",
    binding_variant = "default",
    builtin_path = "crate::builtins::comms::pskmod"
)]
async fn pskmod_builtin(
    x: Value,
    m: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let order = parse_modulation_order(&m)?;
    if is_typed_integer_value(&m) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PSKMOD_INTEGER_ORDER_EXTENSION,
            NAME,
        )?;
    }
    let options = ParsedOptions::parse(rest.view(), order)?;
    if options.integer_phase_extension {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PSKMOD_INTEGER_PHASE_EXTENSION,
            NAME,
        )?;
    }
    if options.integer_custom_order_extension {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PSKMOD_INTEGER_CUSTOM_ORDER_EXTENSION,
            NAME,
        )?;
    }
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
        .ok_or_else(|| pskmod_error("pskmod: no owning acceleration provider registered"))?;
    let constellation = constellation_table(order, &options)?;
    if let Some(class) = runmat_accelerate_api::handle_integer_type(&handle) {
        validate_gpu_symbol_class(class)?;
    }
    if runmat_accelerate_api::handle_is_logical(&handle)
        && !matches!(options.input_type, InputType::Bit)
    {
        return Err(pskmod_error("pskmod: logical X requires InputType='bit'"));
    }
    // These hooks produce provider-native complex storage. Other requested
    // classes retain the existing typed host modulation/upload route.
    let target_precision = match options.output_dtype {
        OutputDType::Double => ProviderPrecision::F64,
        OutputDType::Single => ProviderPrecision::F32,
    };
    if provider.precision() == target_precision {
        let mut output_shape = handle.shape().to_vec();
        let result = match options.input_type {
            InputType::Integer => Some(
                provider
                    .modulate_constellation(ProviderModulationRequest {
                        input: &handle,
                        constellation: &constellation,
                    })
                    .await,
            ),
            InputType::Bit => {
                let input_rows = handle.shape().first().copied().unwrap_or(0);
                let bits = bits_per_symbol(order)?;
                if input_rows > 0 {
                    if !input_rows.is_multiple_of(bits) {
                        return Err(pskmod_error(
                            "pskmod: bit rows must be a multiple of bits per symbol",
                        ));
                    }
                    output_shape[0] = input_rows / bits;
                    Some(
                        provider
                            .modulate_bits_constellation(ProviderBitModulationRequest {
                                input: &handle,
                                input_rows,
                                bits_per_symbol: bits,
                                constellation: &constellation,
                            })
                            .await,
                    )
                } else {
                    None
                }
            }
        };
        if let Some(result) = result {
            match result {
                Ok(out) => {
                    if !valid_gpu_output(
                        provider,
                        &handle,
                        &out,
                        options.output_dtype,
                        &output_shape,
                    ) {
                        return Err(gpu_helpers::reject_gpu_output(
                            out,
                            pskmod_error("pskmod: provider returned an invalid modulation result"),
                        ));
                    }
                    return gpu_helpers::complex_gpu_value(out);
                }
                Err(error) if error.is_unsupported() => {}
                Err(error) => {
                    return Err(crate::runtime_error::with_provider_execution_source(
                        pskmod_error(format!("pskmod: provider modulation failed: {error}")),
                        error,
                    ));
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
    let out = gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[&handle])?;
    gpu_helpers::complex_gpu_value(out)
}

fn valid_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    handle: &GpuTensorHandle,
    dtype: OutputDType,
    shape: &[usize],
) -> bool {
    let element_type = match dtype {
        OutputDType::Double => runmat_accelerate_api::NumericElementType::F64,
        OutputDType::Single => runmat_accelerate_api::NumericElementType::F32,
    };
    !gpu_helpers::same_gpu_handle(input, handle)
        && handle.device_id() == provider.device_id()
        && handle.shape() == shape
        && runmat_accelerate_api::provider_for_handle(handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && gpu_helpers::numeric_descriptor_matches(
            handle,
            element_type,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        )
        && !runmat_accelerate_api::handle_is_logical(handle)
}

fn validate_gpu_symbol_class(class: IntegerElementType) -> BuiltinResult<()> {
    if matches!(
        class,
        IntegerElementType::I8
            | IntegerElementType::I16
            | IntegerElementType::I32
            | IntegerElementType::U8
            | IntegerElementType::U16
            | IntegerElementType::U32
    ) {
        Ok(())
    } else {
        Err(pskmod_error(
            "pskmod: integer X must have class int8, int16, int32, uint8, uint16, or uint32",
        ))
    }
}

#[derive(Clone, Debug)]
struct ParsedOptions {
    phase_offset: f64,
    mapping: SymbolMapping,
    input_type: InputType,
    output_dtype: OutputDType,
    integer_phase_extension: bool,
    integer_custom_order_extension: bool,
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
    fn parse(args: crate::call::FunctionArgumentView<'_>, order: usize) -> BuiltinResult<Self> {
        let mut phase_offset = 0.0;
        let mut mapping = SymbolMapping::Gray;
        let mut input_type = InputType::Integer;
        let mut output_dtype = OutputDType::Double;
        let mut integer_phase_extension = false;
        let mut integer_custom_order_extension = false;
        let mut idx = 0usize;

        if let Some(first) = args.first() {
            if !is_name_value_key(first)? {
                if is_scalar_numeric(first) {
                    phase_offset = scalar_number(first, "phaseoffset")?;
                    integer_phase_extension = is_typed_integer_value(first);
                } else {
                    mapping = parse_symbol_mapping(first, order)?;
                    integer_custom_order_extension = matches!(&mapping, SymbolMapping::Custom(_))
                        && is_typed_integer_value(first);
                }
                idx = 1;
            }
        }
        if let Some(second) = args.get(idx) {
            if !is_name_value_key(second)? {
                mapping = parse_symbol_mapping(second, order)?;
                integer_custom_order_extension =
                    matches!(&mapping, SymbolMapping::Custom(_)) && is_typed_integer_value(second);
                idx += 1;
            }
        }

        while idx < args.len() {
            let key = value_as_string(&args[idx])?
                .ok_or_else(|| pskmod_error("pskmod: expected name-value option name"))?;
            let Some(value) = args.get(idx + 1) else {
                return Err(pskmod_error(format!(
                    "pskmod: expected value after option '{key}'"
                )));
            };
            match normalize_key(&key).as_str() {
                "inputtype" => {
                    let text = value_as_string(value)?
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
                    let dtype = value_as_string(value)?.ok_or_else(|| {
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
            integer_phase_extension,
            integer_custom_order_extension,
        })
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
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
                validate_scalar_symbol_class(&i)?;
                let symbol = integer_to_symbol_with_name(&i, "X")?;
                match input_type {
                    InputType::Integer => Ok(Self {
                        data: vec![symbol],
                        shape: vec![1, 1],
                    }),
                    InputType::Bit => {
                        bits_to_symbols(vec![symbol_to_bit(symbol, "X")?], vec![1, 1], order)
                    }
                }
            }
            Value::Bool(b) => {
                let logical = LogicalArray {
                    data: vec![u8::from(b)].into(),
                    shape: vec![1, 1].into(),
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
        let shape = tensor.shape().to_vec();
        if let Some(storage) = tensor.integer_storage() {
            validate_symbol_storage_class(storage)?;
            let symbols = integer_storage_to_symbols(storage, "X")?;
            return match input_type {
                InputType::Integer => Ok(Self {
                    data: symbols,
                    shape,
                }),
                InputType::Bit => bits_to_symbols(symbols_to_bits(symbols, "X")?, shape, order),
            };
        }
        match input_type {
            InputType::Integer => {
                let data = tensor_utils::tensor_values_f64_cow(&tensor)
                    .iter()
                    .map(|&value| number_to_symbol_with_name(value, "X"))
                    .collect::<BuiltinResult<Vec<_>>>()?;
                Ok(Self { data, shape })
            }
            InputType::Bit => {
                let bits = tensor_utils::tensor_values_f64_cow(&tensor)
                    .iter()
                    .map(|&value| number_to_bit(value, "X"))
                    .collect::<BuiltinResult<Vec<_>>>()?;
                bits_to_symbols(bits, shape, order)
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
            InputType::Bit => {
                bits_to_symbols(logical.data.into_vec(), logical.shape.to_vec(), order)
            }
        }
    }
}

fn validate_scalar_symbol_class(value: &IntValue) -> BuiltinResult<()> {
    if matches!(
        value,
        IntValue::I8(_)
            | IntValue::I16(_)
            | IntValue::I32(_)
            | IntValue::U8(_)
            | IntValue::U16(_)
            | IntValue::U32(_)
    ) {
        Ok(())
    } else {
        Err(pskmod_error(
            "pskmod: integer X must have class int8, int16, int32, uint8, uint16, or uint32",
        ))
    }
}

fn validate_symbol_storage_class(storage: &IntegerStorage) -> BuiltinResult<()> {
    if matches!(
        storage,
        IntegerStorage::I8(_)
            | IntegerStorage::I16(_)
            | IntegerStorage::I32(_)
            | IntegerStorage::U8(_)
            | IntegerStorage::U16(_)
            | IntegerStorage::U32(_)
    ) {
        Ok(())
    } else {
        Err(pskmod_error(
            "pskmod: integer X must have class int8, int16, int32, uint8, uint16, or uint32",
        ))
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

    let dtype = match options.output_dtype {
        OutputDType::Double => NumericDType::F64,
        OutputDType::Single => NumericDType::F32,
    };
    let tensor = ComplexTensor::from_f64_values_with_dtype(out, symbols.shape, dtype)
        .map_err(|e| pskmod_error(format!("pskmod: {e}")))?;
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
    if let Value::Int(value) = value {
        let order = integer_to_symbol_with_name(value, "M")?;
        if order < 2 || order > isize::MAX as usize {
            return Err(pskmod_error(
                "pskmod: M must be a positive integer greater than 1",
            ));
        }
        return Ok(order);
    }
    if let Value::Tensor(tensor) = value {
        if tensor_utils::is_scalar_tensor(tensor) {
            if let Some(storage) = tensor.integer_storage() {
                let integer = storage
                    .value_at(0)
                    .ok_or_else(|| pskmod_error("pskmod: M must be a scalar"))?;
                let order = integer_to_symbol_with_name(&integer, "M")?;
                if order < 2 || order > isize::MAX as usize {
                    return Err(pskmod_error(
                        "pskmod: M must be a positive integer greater than 1",
                    ));
                }
                return Ok(order);
            }
        }
    }
    let number = scalar_number(value, "M")?;
    let order = number_to_symbol_with_name(number, "M")?;
    if order < 2 || order > isize::MAX as usize {
        return Err(pskmod_error(
            "pskmod: M must be a positive integer greater than 1",
        ));
    }
    Ok(order)
}

fn parse_symbol_mapping(value: &Value, order: usize) -> BuiltinResult<SymbolMapping> {
    if let Some(text) = value_as_string(value)? {
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
        Value::Tensor(tensor) => {
            let shape = tensor.shape();
            if shape.len() > 2 || (shape.len() == 2 && shape[0] > 1 && shape[1] > 1) {
                return Err(pskmod_error(format!("pskmod: {name} must be a vector")));
            }
            if let Some(storage) = tensor.integer_storage() {
                integer_storage_to_symbols(storage, name)
            } else {
                tensor_utils::tensor_values_f64_cow(tensor)
                    .iter()
                    .map(|&number| number_to_symbol_with_name(number, name))
                    .collect()
            }
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() > 2
                || (logical.shape.len() == 2 && logical.shape[0] > 1 && logical.shape[1] > 1)
            {
                return Err(pskmod_error(format!("pskmod: {name} must be a vector")));
            }
            Ok(logical.data.iter().map(|&v| usize::from(v)).collect())
        }
        Value::Num(n) => Ok(vec![number_to_symbol_with_name(*n, name)?]),
        Value::Int(i) => Ok(vec![integer_to_symbol_with_name(i, name)?]),
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
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(pskmod_error(format!("pskmod: {name} must be a scalar"))),
    }
}

fn is_scalar_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor_utils::is_scalar_tensor(tensor),
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
    if value.fract() != 0.0 || value < 0.0 {
        return Err(pskmod_error(format!(
            "pskmod: {name} values must be nonnegative integers"
        )));
    }
    if value > usize::MAX.saturating_sub(1) as f64 {
        return Err(pskmod_error(format!(
            "pskmod: {name} value is too large for this platform"
        )));
    }
    let parsed = value as usize;
    if parsed as f64 != value || parsed == usize::MAX {
        return Err(pskmod_error(format!(
            "pskmod: {name} value is too large for this platform"
        )));
    }
    Ok(parsed)
}

fn integer_storage_to_symbols(storage: &IntegerStorage, name: &str) -> BuiltinResult<Vec<usize>> {
    storage
        .exact_values()
        .iter()
        .map(|value| integer_to_symbol_with_name(value, name))
        .collect()
}

fn integer_to_symbol_with_name(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        pskmod_error(format!(
            "pskmod: {name} value is too large for this platform"
        ))
    })
}

fn number_to_bit(value: f64, name: &str) -> BuiltinResult<u8> {
    let symbol = number_to_symbol_with_name(value, name)?;
    symbol_to_bit(symbol, name)
}

fn symbol_to_bit(symbol: usize, name: &str) -> BuiltinResult<u8> {
    match symbol {
        0 | 1 => Ok(symbol as u8),
        _ => Err(pskmod_error(format!(
            "pskmod: {name} bit inputs must contain only 0 or 1"
        ))),
    }
}

fn symbols_to_bits(symbols: Vec<usize>, name: &str) -> BuiltinResult<Vec<u8>> {
    symbols
        .into_iter()
        .map(|symbol| symbol_to_bit(symbol, name))
        .collect()
}

fn value_as_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    let n = scalar_number(value, name)?;
    if n == 0.0 {
        Ok(false)
    } else if n == 1.0 {
        Ok(true)
    } else {
        Err(pskmod_error(format!(
            "pskmod: {name} must be logical or numeric 0/1"
        )))
    }
}

fn value_as_string(value: &Value) -> BuiltinResult<Option<runmat_value::Utf8Text>> {
    Ok(match value {
        Value::String(s) => s.text_ref().cloned(),
        Value::StringArray(sa) if sa.data().len() == 1 => sa.data()[0].text_ref().cloned(),
        Value::CharArray(ca) if ca.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(ca.data())
                .map_err(|source| {
                    let mut error = pskmod_error(source.to_string());
                    error.source = Some(Box::new(source));
                    error
                })?
                .into(),
        ),
        _ => None,
    })
}

fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(|c| *c != '_' && *c != '-' && !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_name_value_key(value: &Value) -> BuiltinResult<bool> {
    let Some(text) = value_as_string(value)? else {
        return Ok(false);
    };
    Ok(matches!(
        normalize_key(&text).as_str(),
        "inputtype" | "outputdatatype" | "plotconstellation"
    ))
}

#[cfg(test)]
mod tests;
