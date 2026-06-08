//! MATLAB-compatible `qammod` builtin for integer-input rectangular QAM.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, LogicalArray, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "qammod";
const INTEGER_TOL: f64 = 1e-9;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::comms::qammod")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("modulation"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64, ScalarType::I32, ScalarType::Bool],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("qammod")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Accepts gpuArray inputs by gathering through the active provider and returning a host ComplexTensor; complex GPU handles are not yet represented by the runtime.",
};

fn qammod_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn qammod_type(args: &[Type], _ctx: &ResolveContext) -> Type {
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
    name = "qammod",
    category = "comms/modulation",
    summary = "Map integer symbols onto a QAM complex-baseband constellation.",
    keywords = "qammod,qam,modulation,communications,gray,binary,gpu",
    type_resolver(qammod_type),
    builtin_path = "crate::builtins::comms::qammod"
)]
async fn qammod_builtin(x: Value, m: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let order = parse_modulation_order(&m)?;
    let options = ParsedOptions::parse(&rest, order)?;
    match x {
        Value::GpuTensor(handle) => qammod_gpu(handle, order, options).await,
        other => {
            let symbols = SymbolInput::from_value(other)?;
            modulate_symbols(symbols, order, &options)
        }
    }
}

async fn qammod_gpu(
    handle: GpuTensorHandle,
    order: usize,
    options: ParsedOptions,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let symbols = SymbolInput::from_tensor(tensor)?;
    modulate_symbols(symbols, order, &options)
}

#[derive(Clone, Debug)]
struct ParsedOptions {
    mapping: SymbolMapping,
    unit_average_power: bool,
    output_dtype: OutputDType,
}

#[derive(Clone, Debug)]
enum SymbolMapping {
    Binary,
    Gray,
    Custom(Vec<usize>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputDType {
    Double,
    Single,
}

impl ParsedOptions {
    fn parse(args: &[Value], order: usize) -> BuiltinResult<Self> {
        let mut mapping = SymbolMapping::Gray;
        let mut unit_average_power = false;
        let mut output_dtype = OutputDType::Double;
        let mut idx = 0;

        if let Some(first) = args.first() {
            if !is_name_value_key(first) {
                mapping = parse_symbol_mapping(first, order)?;
                idx = 1;
            }
        }

        while idx < args.len() {
            let key = value_as_string(&args[idx])
                .ok_or_else(|| qammod_error("qammod: expected name-value option name"))?;
            let Some(value) = args.get(idx + 1) else {
                return Err(qammod_error(format!(
                    "qammod: expected value after option '{key}'"
                )));
            };
            match normalize_key(&key).as_str() {
                "unitaveragepower" => {
                    unit_average_power = value_as_bool(value, "UnitAveragePower")?;
                }
                "plotconstellation" => {
                    if value_as_bool(value, "PlotConstellation")? {
                        return Err(qammod_error(
                            "qammod: PlotConstellation is not implemented in RunMat yet",
                        ));
                    }
                }
                "outputdatatype" => {
                    let dtype = value_as_string(value).ok_or_else(|| {
                        qammod_error("qammod: OutputDataType must be 'double' or 'single'")
                    })?;
                    output_dtype = match dtype.trim().to_ascii_lowercase().as_str() {
                        "double" => OutputDType::Double,
                        "single" => OutputDType::Single,
                        _ => {
                            return Err(qammod_error(
                                "qammod: only OutputDataType 'double' and 'single' are accepted",
                            ));
                        }
                    };
                }
                "inputtype" => {
                    let input_type = value_as_string(value)
                        .ok_or_else(|| qammod_error("qammod: InputType must be a string"))?;
                    match input_type.trim().to_ascii_lowercase().as_str() {
                        "integer" => {}
                        "bit" => {
                            return Err(qammod_error(
                                "qammod: InputType='bit' is not implemented in RunMat yet",
                            ));
                        }
                        other => {
                            return Err(qammod_error(format!(
                                "qammod: unsupported InputType '{other}'"
                            )));
                        }
                    }
                }
                other => {
                    return Err(qammod_error(format!(
                        "qammod: unrecognised option '{other}'"
                    )));
                }
            }
            idx += 2;
        }

        Ok(Self {
            mapping,
            unit_average_power,
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
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::LogicalArray(logical) => Self::from_logical(logical),
            Value::Num(n) => Ok(Self {
                data: vec![number_to_symbol(n)?],
                shape: vec![1, 1],
            }),
            Value::Int(i) => Ok(Self {
                data: vec![number_to_symbol(i.to_f64())?],
                shape: vec![1, 1],
            }),
            Value::Bool(b) => Ok(Self {
                data: vec![usize::from(b)],
                shape: vec![1, 1],
            }),
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err(qammod_error("qammod: X must contain real integer symbols"))
            }
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                Err(qammod_error("qammod: X must be numeric or logical"))
            }
            other => Err(qammod_error(format!(
                "qammod: unsupported input type {other:?}"
            ))),
        }
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let data = tensor
            .data
            .iter()
            .map(|&value| number_to_symbol(value))
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            data,
            shape: tensor.shape,
        })
    }

    fn from_logical(logical: LogicalArray) -> BuiltinResult<Self> {
        Ok(Self {
            data: logical.data.into_iter().map(usize::from).collect(),
            shape: logical.shape,
        })
    }
}

fn modulate_symbols(
    symbols: SymbolInput,
    order: usize,
    options: &ParsedOptions,
) -> BuiltinResult<Value> {
    validate_symbol_range(&symbols.data, order)?;
    let layout = ConstellationLayout::for_order(order)?;
    let scale = if options.unit_average_power {
        1.0 / layout.average_power().sqrt()
    } else {
        1.0
    };
    let custom_inverse = match &options.mapping {
        SymbolMapping::Custom(mapping) => Some(invert_mapping(mapping, order)?),
        _ => None,
    };

    let mut out = Vec::with_capacity(symbols.data.len());
    for symbol in symbols.data {
        let point_index = match &options.mapping {
            SymbolMapping::Binary => symbol,
            SymbolMapping::Gray => layout.gray_point_index(symbol),
            SymbolMapping::Custom(_) => custom_inverse
                .as_ref()
                .and_then(|inverse| inverse.get(symbol).copied())
                .ok_or_else(|| qammod_error("qammod: invalid custom mapping"))?,
        };
        let (i_amp, q_amp) = layout.point_for_index(point_index);
        let point = (i_amp * scale, q_amp * scale);
        out.push(match options.output_dtype {
            OutputDType::Double => point,
            OutputDType::Single => ((point.0 as f32) as f64, (point.1 as f32) as f64),
        });
    }

    let tensor =
        ComplexTensor::new(out, symbols.shape).map_err(|e| qammod_error(format!("qammod: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

#[derive(Clone, Debug)]
struct ConstellationLayout {
    rows: usize,
    cols: usize,
}

impl ConstellationLayout {
    fn for_order(order: usize) -> BuiltinResult<Self> {
        let bits = order.trailing_zeros() as usize;
        let row_bits = bits / 2;
        let col_bits = bits - row_bits;
        Ok(Self {
            rows: 1usize
                .checked_shl(row_bits as u32)
                .ok_or_else(|| qammod_error("qammod: modulation order is too large"))?,
            cols: 1usize
                .checked_shl(col_bits as u32)
                .ok_or_else(|| qammod_error("qammod: modulation order is too large"))?,
        })
    }

    fn gray_point_index(&self, symbol: usize) -> usize {
        let row = symbol % self.rows;
        let col = symbol / self.rows;
        gray_encode(col) * self.rows + gray_encode(row)
    }

    fn point_for_index(&self, point_index: usize) -> (f64, f64) {
        let row = point_index % self.rows;
        let col = point_index / self.rows;
        let i_amp = 2.0 * col as f64 - (self.cols as f64 - 1.0);
        let q_amp = (self.rows as f64 - 1.0) - 2.0 * row as f64;
        (i_amp, q_amp)
    }

    fn average_power(&self) -> f64 {
        average_axis_power(self.rows) + average_axis_power(self.cols)
    }
}

fn average_axis_power(points: usize) -> f64 {
    if points <= 1 {
        0.0
    } else {
        ((points * points - 1) as f64) / 3.0
    }
}

fn gray_encode(value: usize) -> usize {
    value ^ (value >> 1)
}

fn parse_modulation_order(value: &Value) -> BuiltinResult<usize> {
    let number = scalar_number(value, "M")?;
    let order = number_to_symbol_with_name(number, "M")?;
    if order < 2 || !order.is_power_of_two() {
        return Err(qammod_error(
            "qammod: M must be a positive power-of-two integer greater than 1",
        ));
    }
    Ok(order)
}

fn parse_symbol_mapping(value: &Value, order: usize) -> BuiltinResult<SymbolMapping> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "gray" => Ok(SymbolMapping::Gray),
            "bin" | "binary" => Ok(SymbolMapping::Binary),
            other => Err(qammod_error(format!(
                "qammod: unsupported symbol order '{other}'"
            ))),
        };
    }

    let mapping = vector_to_symbols(value, "symOrder")?;
    if mapping.len() != order {
        return Err(qammod_error(format!(
            "qammod: custom symbol order must contain exactly {order} elements"
        )));
    }
    validate_custom_mapping(&mapping, order)?;
    Ok(SymbolMapping::Custom(mapping))
}

fn validate_symbol_range(symbols: &[usize], order: usize) -> BuiltinResult<()> {
    for &symbol in symbols {
        if symbol >= order {
            return Err(qammod_error(format!(
                "qammod: input symbols must be in the range [0, {}]",
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
            return Err(qammod_error(format!(
                "qammod: custom symbol order values must be in the range [0, {}]",
                order - 1
            )));
        }
        if seen[symbol] {
            return Err(qammod_error(
                "qammod: custom symbol order values must be unique",
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
        other => Err(qammod_error(format!(
            "qammod: {name} must be a numeric vector, got {other:?}"
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
        _ => Err(qammod_error(format!("qammod: {name} must be a scalar"))),
    }
}

fn number_to_symbol(value: f64) -> BuiltinResult<usize> {
    number_to_symbol_with_name(value, "X")
}

fn number_to_symbol_with_name(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(qammod_error(format!(
            "qammod: {name} values must be finite integers"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > INTEGER_TOL || rounded < 0.0 {
        return Err(qammod_error(format!(
            "qammod: {name} values must be nonnegative integers"
        )));
    }
    if rounded > usize::MAX as f64 {
        return Err(qammod_error(format!(
            "qammod: {name} value is too large for this platform"
        )));
    }
    Ok(rounded as usize)
}

fn value_as_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    let n = scalar_number(value, name)?;
    if (n - 0.0).abs() <= INTEGER_TOL {
        Ok(false)
    } else if (n - 1.0).abs() <= INTEGER_TOL {
        Ok(true)
    } else {
        Err(qammod_error(format!(
            "qammod: {name} must be logical or numeric 0/1"
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
        "unitaveragepower" | "plotconstellation" | "outputdatatype" | "inputtype"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn qammod(x: Value, order: usize, rest: Vec<Value>) -> ComplexTensor {
        match block_on(super::qammod_builtin(x, Value::Num(order as f64), rest)).expect("qammod") {
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
    fn qammod_4_gray_mapping() {
        let out = qammod(tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]), 4, vec![]);
        assert_eq!(out.shape, vec![1, 4]);
        assert_complex_close(
            &out.data,
            &[(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)],
        );
    }

    #[test]
    fn qammod_16_gray_mapping() {
        let input: Vec<f64> = (0..16).map(|v| v as f64).collect();
        let out = qammod(tensor(input, vec![1, 16]), 16, vec![]);
        assert_complex_close(
            &out.data,
            &[
                (-3.0, 3.0),
                (-3.0, 1.0),
                (-3.0, -3.0),
                (-3.0, -1.0),
                (-1.0, 3.0),
                (-1.0, 1.0),
                (-1.0, -3.0),
                (-1.0, -1.0),
                (3.0, 3.0),
                (3.0, 1.0),
                (3.0, -3.0),
                (3.0, -1.0),
                (1.0, 3.0),
                (1.0, 1.0),
                (1.0, -3.0),
                (1.0, -1.0),
            ],
        );
    }

    #[test]
    fn qammod_64_gray_has_expected_edges() {
        let out = qammod(
            tensor(vec![0.0, 1.0, 2.0, 3.0, 63.0], vec![1, 5]),
            64,
            vec![],
        );
        assert_complex_close(
            &out.data,
            &[
                (-7.0, 7.0),
                (-7.0, 5.0),
                (-7.0, 1.0),
                (-7.0, 3.0),
                (1.0, -1.0),
            ],
        );
    }

    #[test]
    fn qammod_binary_mapping_is_sequential_columnwise() {
        let out = qammod(
            tensor(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![1, 5]),
            16,
            vec![Value::from("bin")],
        );
        assert_complex_close(
            &out.data,
            &[
                (-3.0, 3.0),
                (-3.0, 1.0),
                (-3.0, -1.0),
                (-3.0, -3.0),
                (-1.0, 3.0),
            ],
        );
    }

    #[test]
    fn qammod_custom_mapping_uses_columnwise_symbol_order() {
        let mapping = tensor(vec![0.0, 3.0, 1.0, 2.0], vec![1, 4]);
        let out = qammod(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
            4,
            vec![mapping],
        );
        assert_complex_close(
            &out.data,
            &[(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)],
        );
    }

    #[test]
    fn qammod_unit_average_power_normalizes_constellation() {
        let input: Vec<f64> = (0..16).map(|v| v as f64).collect();
        let out = qammod(
            tensor(input, vec![1, 16]),
            16,
            vec![Value::from("UnitAveragePower"), Value::Bool(true)],
        );
        let avg_power = out
            .data
            .iter()
            .map(|(re, im)| re * re + im * im)
            .sum::<f64>()
            / out.data.len() as f64;
        assert!((avg_power - 1.0).abs() < 1e-12, "{avg_power}");
    }

    #[test]
    fn qammod_output_datatype_single_rounds_samples() {
        let out = qammod(
            tensor(vec![0.0], vec![1, 1]),
            16,
            vec![
                Value::from("UnitAveragePower"),
                Value::Bool(true),
                Value::from("OutputDataType"),
                Value::from("single"),
            ],
        );
        let expected = (-3.0 / 10.0_f64.sqrt()) as f32 as f64;
        assert_eq!(out.data[0].0, expected);
        assert_eq!(out.data[0].1, -expected);
    }

    #[test]
    fn qammod_rejects_out_of_range_symbols() {
        let err = block_on(super::qammod_builtin(
            tensor(vec![0.0, 16.0], vec![1, 2]),
            Value::Num(16.0),
            vec![],
        ))
        .expect_err("expected out-of-range error");
        assert!(err.to_string().contains("range [0, 15]"));
    }
}
