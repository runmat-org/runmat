//! MATLAB-compatible legacy `histc` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "histc";
const MAX_DIMENSION_ARGUMENT: usize = 32;

const OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bincounts",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy histogram bin counts.",
}];

const OUTPUT_N_IND: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bincounts",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Legacy histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "ind",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based bin index for each input element, or zero when outside all bins.",
    },
];

const INPUT_X_EDGES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input values to sort into legacy bins.",
    },
    BuiltinParamDescriptor {
        name: "binranges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Monotonically nondecreasing bin endpoints.",
    },
];

const INPUT_X_EDGES_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input values to sort into legacy bins.",
    },
    BuiltinParamDescriptor {
        name: "binranges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Monotonically nondecreasing bin endpoints.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension along which to operate.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "bincounts = histc(x, binranges)",
        inputs: &INPUT_X_EDGES,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "bincounts = histc(x, binranges, dim)",
        inputs: &INPUT_X_EDGES_DIM,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[bincounts, ind] = histc(x, binranges)",
        inputs: &INPUT_X_EDGES,
        outputs: &OUTPUT_N_IND,
    },
    BuiltinSignatureDescriptor {
        label: "[bincounts, ind] = histc(x, binranges, dim)",
        inputs: &INPUT_X_EDGES_DIM,
        outputs: &OUTPUT_N_IND,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTC.INVALID_ARGUMENT",
    identifier: Some("RunMat:histc:InvalidArgument"),
    when: "Arguments are malformed, inconsistent, or unsupported.",
    message: "histc: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTC.INTERNAL",
    identifier: Some("RunMat:histc:Internal"),
    when: "Internal tensor conversion, GPU gather, or allocation fails.",
    message: "histc: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const HISTC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn histc_type(args: &[Type], ctx: &runmat_builtins::ResolveContext) -> Type {
    let Some(edge_len) = args.get(1).and_then(edge_count_from_type) else {
        return Type::tensor();
    };
    let Some(input_shape) = args
        .first()
        .and_then(type_shape)
        .map(|shape| shape.to_vec())
    else {
        return Type::Tensor {
            shape: Some(vec![Some(1), Some(edge_len)]),
        };
    };
    if is_type_vector(&input_shape) && ctx.numeric_dims().is_empty() {
        return match args.get(1).and_then(type_shape) {
            Some(edge_shape) => Type::Tensor {
                shape: Some(edge_shape.to_vec()),
            },
            None => Type::Tensor {
                shape: Some(vec![Some(1), Some(edge_len)]),
            },
        };
    }
    let dim = ctx
        .numeric_dims()
        .first()
        .and_then(|dim| dim.and_then(|d| d.checked_sub(1)))
        .unwrap_or_else(|| first_nonsingleton_type_dim(&input_shape));
    let mut out_shape = input_shape;
    if out_shape.len() <= dim {
        out_shape.resize(dim + 1, Some(1));
    }
    out_shape[dim] = Some(edge_len);
    Type::Tensor {
        shape: Some(out_shape),
    }
}

#[runtime_builtin(
    name = "histc",
    category = "stats/hist",
    summary = "Count values in legacy histogram bins.",
    keywords = "histc,histogram,legacy,bin counts,bin index",
    accel = "reduction",
    sink = true,
    type_resolver(histc_type),
    descriptor(crate::builtins::stats::hist::histc::HISTC_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::hist::histc"
)]
async fn histc_builtin(x: Value, binranges: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let dim = match rest.as_slice() {
        [] => None,
        [value] => Some(
            tensor::dimension_from_value_async(value, NAME, false)
                .await
                .map_err(invalid_argument)?
                .ok_or_else(|| invalid_argument("histc: dim must be numeric"))
                .and_then(validate_dimension_argument)?,
        ),
        _ => {
            return Err(invalid_argument(
                "histc: expected x, binranges, and optional dim",
            ))
        }
    };
    let eval = evaluate(x, binranges, dim).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![Value::Tensor(eval.counts)]));
        }
        if out_count > 2 {
            return Err(invalid_argument("histc: too many output arguments"));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Tensor(eval.counts), Value::Tensor(eval.indices)],
        ));
    }
    Ok(Value::Tensor(eval.counts))
}

pub async fn evaluate(
    x: Value,
    binranges: Value,
    dim: Option<usize>,
) -> BuiltinResult<HistcResult> {
    let x = value_to_real_tensor(x).await?;
    let edges = EdgeSets::from_tensor(value_to_real_tensor(binranges).await?)?;
    histc_from_tensors(x, edges, dim)
}

#[derive(Debug)]
pub struct HistcResult {
    counts: Tensor,
    indices: Tensor,
}

#[derive(Clone)]
struct EdgeSets {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    vector_shape: Option<Vec<usize>>,
}

impl EdgeSets {
    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let shape = tensor.shape.clone();
        let data = tensor::tensor_into_values_f64(tensor);
        if is_vector_shape(&shape) {
            validate_edges(&data)?;
            return Ok(Self {
                rows: data.len(),
                cols: 1,
                data,
                vector_shape: Some(shape),
            });
        }
        if shape.len() > 2 {
            return Err(invalid_argument(
                "histc: matrix binranges must be two-dimensional",
            ));
        }
        let rows = shape.first().copied().unwrap_or(data.len());
        let cols = shape.get(1).copied().unwrap_or(1);
        for col in 0..cols {
            let start = col * rows;
            validate_edges(&data[start..start + rows])?;
        }
        Ok(Self {
            data,
            rows,
            cols,
            vector_shape: None,
        })
    }

    fn edge_slice(&self, slice: usize) -> &[f64] {
        let col = if self.cols == 1 { 0 } else { slice };
        let start = col * self.rows;
        &self.data[start..start + self.rows]
    }

    fn output_shape_for_vector_call(&self) -> Vec<usize> {
        self.vector_shape
            .clone()
            .unwrap_or_else(|| vec![self.rows, self.cols])
    }
}

fn histc_from_tensors(
    x: Tensor,
    edges: EdgeSets,
    dim: Option<usize>,
) -> BuiltinResult<HistcResult> {
    let x_values = tensor::tensor_values_f64(&x);
    let x_shape = canonical_shape(&x.shape);
    let vector_call = dim.is_none() && is_vector_shape(&x_shape);
    let dim_index = match dim {
        Some(dim) => dim - 1,
        None => first_nonsingleton_dim(&x_shape),
    };
    let shape = shape_with_dim(x_shape, dim_index);
    let dim_len = shape[dim_index];
    let stride = checked_product(&shape[..dim_index])?;
    let after = checked_product(&shape[dim_index + 1..])?;
    let slice_count = stride
        .checked_mul(after)
        .ok_or_else(|| internal_error("histc: slice count overflow"))?;

    if edges.vector_shape.is_none() && edges.cols != 1 && edges.cols != slice_count {
        return Err(invalid_argument(format!(
            "histc: binranges has {} columns but operation has {slice_count} slices",
            edges.cols
        )));
    }

    let counts_shape = if vector_call {
        edges.output_shape_for_vector_call()
    } else {
        let mut out = shape.clone();
        out[dim_index] = edges.rows;
        trim_trailing_added_dims(out, x.shape.len(), dim_index)
    };
    let mut counts = vec![0.0; checked_product(&counts_shape)?];
    let empty_edges = edges.rows == 0;
    let mut indices = if empty_edges {
        Vec::new()
    } else {
        vec![0.0; x_values.len()]
    };

    let out_stride = if vector_call {
        1
    } else {
        checked_product(&counts_shape[..dim_index])?
    };
    let out_dim_len = edges.rows;

    for high in 0..after {
        for low in 0..stride {
            let slice = low + high * stride;
            let edge_slice = edges.edge_slice(slice);
            for pos in 0..dim_len {
                let input_index = low + pos * stride + high * stride * dim_len;
                if input_index >= x_values.len() {
                    continue;
                }
                if let Some(bin) = classify_bin(x_values[input_index], edge_slice) {
                    if !empty_edges {
                        indices[input_index] = (bin + 1) as f64;
                    }
                    let count_index = if vector_call {
                        bin
                    } else {
                        low + bin * out_stride + high * out_stride * out_dim_len
                    };
                    if let Some(count) = counts.get_mut(count_index) {
                        *count += 1.0;
                    }
                }
            }
        }
    }

    Ok(HistcResult {
        counts: Tensor::new(counts, counts_shape)
            .map_err(|err| internal_error(format!("histc: {err}")))?,
        indices: Tensor::new(indices, if empty_edges { vec![0, 0] } else { x.shape })
            .map_err(|err| internal_error(format!("histc: {err}")))?,
    })
}

fn classify_bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value.is_nan() || edges.is_empty() {
        return None;
    }
    let last = *edges.last()?;
    if value == last {
        return Some(edges.len() - 1);
    }
    if value < edges[0] || value > last {
        return None;
    }
    let insertion = edges.partition_point(|edge| *edge <= value);
    if insertion == 0 {
        return None;
    }
    let bin = insertion - 1;
    (bin < edges.len().saturating_sub(1)).then_some(bin)
}

async fn value_to_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    let host = match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| internal_error(format!("histc: {err}")))?,
        other => other,
    };
    host_to_real_tensor(host)
}

fn validate_dimension_argument(dim: usize) -> BuiltinResult<usize> {
    if dim > MAX_DIMENSION_ARGUMENT {
        return Err(invalid_argument(format!(
            "histc: dim must be <= {MAX_DIMENSION_ARGUMENT}"
        )));
    }
    Ok(dim)
}

fn host_to_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(re, _) => {
            Tensor::new(vec![re], vec![1, 1]).map_err(|err| internal_error(format!("histc: {err}")))
        }
        Value::ComplexTensor(tensor) => complex_real_tensor(tensor),
        other => tensor::value_into_tensor_for(NAME, other).map_err(invalid_argument),
    }
}

fn complex_real_tensor(tensor: ComplexTensor) -> BuiltinResult<Tensor> {
    if let Some(storage) = tensor.integer_data {
        let data = storage
            .real
            .exact_values()
            .into_iter()
            .map(|value| value.to_f64())
            .collect();
        return Tensor::new(data, tensor.shape)
            .map_err(|err| internal_error(format!("histc: {err}")));
    }
    let data = tensor.data.into_iter().map(|(re, _)| re).collect();
    Tensor::new(data, tensor.shape).map_err(|err| internal_error(format!("histc: {err}")))
}

fn validate_edges(edges: &[f64]) -> BuiltinResult<()> {
    for &edge in edges {
        if edge.is_nan() {
            return Err(invalid_argument("histc: binranges must not contain NaN"));
        }
    }
    for pair in edges.windows(2) {
        if pair[1] < pair[0] {
            return Err(invalid_argument(
                "histc: binranges must be monotonically nondecreasing",
            ));
        }
    }
    Ok(())
}

fn canonical_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn shape_with_dim(mut shape: Vec<usize>, dim: usize) -> Vec<usize> {
    if shape.len() <= dim {
        shape.resize(dim + 1, 1);
    }
    shape
}

fn trim_trailing_added_dims(mut shape: Vec<usize>, original_rank: usize, dim: usize) -> Vec<usize> {
    while shape.len() > original_rank.max(dim + 1) && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|&&dim| dim != 1).count() <= 1
}

fn first_nonsingleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|&dim| dim != 1).unwrap_or(0)
}

fn checked_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| internal_error("histc: shape product overflow"))
}

fn type_shape(ty: &Type) -> Option<&[Option<usize>]> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            Some(shape.as_slice())
        }
        Type::Num | Type::Int | Type::Bool => Some(&[Some(1), Some(1)]),
        _ => None,
    }
}

fn edge_count_from_type(ty: &Type) -> Option<usize> {
    let shape = type_shape(ty)?;
    if shape.iter().any(Option::is_none) {
        return None;
    }
    if is_type_vector(shape) {
        return shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim.unwrap_or(1)));
    }
    shape.first().copied().flatten()
}

fn is_type_vector(shape: &[Option<usize>]) -> bool {
    shape.iter().filter(|dim| !matches!(dim, Some(1))).count() <= 1
}

fn first_nonsingleton_type_dim(shape: &[Option<usize>]) -> usize {
    shape
        .iter()
        .position(|dim| !matches!(dim, Some(1)))
        .unwrap_or(0)
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    descriptor_error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    descriptor_error(&ERROR_INTERNAL, message)
}

fn descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, IntegerComplexStorage, IntegerStorage, ResolveContext};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn values(value: Tensor) -> Vec<f64> {
        value.data
    }

    fn typed_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    #[test]
    fn histc_counts_vector_and_returns_indices() {
        let ages = tensor(
            vec![
                3.0, 12.0, 24.0, 15.0, 5.0, 74.0, 23.0, 54.0, 31.0, 23.0, 64.0, 75.0,
            ],
            vec![1, 12],
        );
        let edges = tensor(vec![0.0, 10.0, 25.0, 50.0, 75.0], vec![1, 5]);
        let result = block_on(evaluate(ages, edges, None)).expect("histc");
        assert_eq!(result.counts.shape, vec![1, 5]);
        assert_eq!(values(result.counts), vec![2.0, 5.0, 1.0, 3.0, 1.0]);
        assert_eq!(
            values(result.indices),
            vec![1.0, 2.0, 2.0, 2.0, 1.0, 4.0, 2.0, 4.0, 3.0, 2.0, 4.0, 5.0]
        );
    }

    #[test]
    fn histc_operates_down_matrix_columns_by_default() {
        let x = tensor(vec![0.0, 0.5, 1.5, 1.0, 2.0, 3.0], vec![3, 2]);
        let edges = tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]);
        let result = block_on(evaluate(x, edges, None)).expect("histc");
        assert_eq!(result.counts.shape, vec![4, 2]);
        assert_eq!(
            values(result.counts),
            vec![2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(values(result.indices), vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn histc_respects_explicit_dimension() {
        let x = tensor(vec![0.0, 2.0, 1.0, 3.0, 2.0, 4.0], vec![2, 3]);
        let edges = tensor(vec![0.0, 2.0, 4.0], vec![1, 3]);
        let result = block_on(evaluate(x, edges, Some(2))).expect("histc");
        assert_eq!(result.counts.shape, vec![2, 3]);
        assert_eq!(values(result.counts), vec![2.0, 0.0, 1.0, 2.0, 0.0, 1.0]);
        assert_eq!(values(result.indices), vec![1.0, 2.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn histc_supports_per_slice_edge_columns() {
        let x = tensor(vec![0.5, 1.5, 10.0, 20.0], vec![2, 2]);
        let edges = tensor(vec![0.0, 1.0, 2.0, 0.0, 15.0, 30.0], vec![3, 2]);
        let result = block_on(evaluate(x, edges, None)).expect("histc");
        assert_eq!(result.counts.shape, vec![3, 2]);
        assert_eq!(values(result.counts), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn histc_ignores_nan_out_of_range_and_imaginary_parts() {
        let x = Value::ComplexTensor(
            ComplexTensor::new(
                vec![
                    (f64::NAN, 7.0),
                    (-1.0, 2.0),
                    (0.5, 9.0),
                    (2.0, -4.0),
                    (3.0, 1.0),
                ],
                vec![1, 5],
            )
            .unwrap(),
        );
        let edges = tensor(vec![0.0, 1.0, 2.0], vec![1, 3]);
        let result = block_on(evaluate(x, edges, None)).expect("histc");
        assert_eq!(values(result.counts), vec![1.0, 0.0, 1.0]);
        assert_eq!(values(result.indices), vec![0.0, 0.0, 1.0, 3.0, 0.0]);
    }

    #[test]
    fn histc_accepts_duplicate_edges_with_rightmost_nonempty_bin() {
        let x = tensor(vec![1.0, 1.5], vec![1, 2]);
        let edges = tensor(vec![0.0, 1.0, 1.0, 2.0], vec![1, 4]);
        let result = block_on(evaluate(x, edges, None)).expect("histc");
        assert_eq!(values(result.counts), vec![0.0, 0.0, 2.0, 0.0]);
        assert_eq!(values(result.indices), vec![3.0, 3.0]);
    }

    #[test]
    fn histc_reads_typed_integer_inputs_and_edges_exactly() {
        let x = typed_tensor(IntegerStorage::U16(vec![0, 1, 2, 3]), vec![1, 4]);
        let edges = typed_tensor(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]);

        let result = block_on(evaluate(x, edges, None)).expect("histc");

        assert_eq!(result.counts.shape, vec![1, 3]);
        assert_eq!(values(result.counts), vec![1.0, 2.0, 1.0]);
        assert_eq!(values(result.indices), vec![1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn histc_reads_typed_integer_matrix_edge_columns_exactly() {
        let x = typed_tensor(IntegerStorage::I16(vec![0, 2, 10, 20]), vec![2, 2]);
        let edges = typed_tensor(IntegerStorage::I16(vec![0, 1, 3, 0, 15, 30]), vec![3, 2]);

        let result = block_on(evaluate(x, edges, None)).expect("histc");

        assert_eq!(result.counts.shape, vec![3, 2]);
        assert_eq!(values(result.counts), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn histc_reads_typed_complex_integer_real_storage_exactly() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I16(vec![0, 1, 2]),
                IntegerStorage::I16(vec![99, 99, 99]),
            )
            .unwrap(),
            vec![1, 3],
        )
        .unwrap();
        let edges = typed_tensor(IntegerStorage::I16(vec![0, 1, 2]), vec![1, 3]);

        let result = block_on(evaluate(Value::ComplexTensor(complex), edges, None)).expect("histc");

        assert_eq!(values(result.counts), vec![1.0, 1.0, 1.0]);
        assert_eq!(values(result.indices), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn histc_rejects_decreasing_or_nan_edges() {
        let x = tensor(vec![1.0], vec![1, 1]);
        let err = block_on(evaluate(
            x.clone(),
            tensor(vec![0.0, 2.0, 1.0], vec![1, 3]),
            None,
        ))
        .expect_err("decreasing edges should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);

        let err = block_on(evaluate(x, tensor(vec![0.0, f64::NAN], vec![1, 2]), None))
            .expect_err("NaN edge should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn histc_rejects_mismatched_edge_columns() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let edges = tensor(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], vec![2, 3]);
        let err = block_on(evaluate(x, edges, None)).expect_err("edge column mismatch");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn histc_empty_edges_return_empty_counts_and_indices() {
        let x = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let edges = tensor(Vec::new(), vec![1, 0]);
        let result = block_on(evaluate(x, edges, None)).expect("histc empty edges");
        assert_eq!(result.counts.shape, vec![1, 0]);
        assert!(result.counts.data.is_empty());
        assert_eq!(result.indices.shape, vec![0, 0]);
        assert!(result.indices.data.is_empty());
    }

    #[test]
    fn histc_rejects_excess_outputs_and_huge_dimensions() {
        let x = tensor(vec![0.0, 1.0], vec![1, 2]);
        let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(histc_builtin(x, edges, Vec::new())).expect_err("too many outputs");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
        drop(_guard);

        let x = tensor(vec![0.0, 1.0], vec![1, 2]);
        let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
        let err = block_on(histc_builtin(
            x,
            edges,
            vec![Value::Num((MAX_DIMENSION_ARGUMENT + 1) as f64)],
        ))
        .expect_err("dim cap");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn histc_type_replaces_operating_dimension() {
        let ctx = ResolveContext::new(Vec::new());
        let out = histc_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(4), Some(1)]),
                },
            ],
            &ctx,
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(2)])
            }
        );

        let ctx = ResolveContext::new(vec![runmat_builtins::LiteralValue::Number(2.0)]);
        let out = histc_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(5)]),
                },
                Type::Int,
            ],
            &ctx,
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(5)])
            }
        );

        let matrix_edges = histc_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(4), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            matrix_edges,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(2)])
            }
        );

        let unknown_edges = histc_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![None, Some(1)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(unknown_edges, Type::tensor());
    }

    #[test]
    fn histc_runtime_builtin_handles_output_count_padding() {
        let x = tensor(vec![0.0, 1.0], vec![1, 2]);
        let edges = tensor(vec![0.0, 1.0], vec![1, 2]);
        let result = block_on(histc_builtin(x, edges, Vec::new())).expect("histc builtin");
        let Value::Tensor(counts) = result else {
            panic!("expected tensor");
        };
        assert_eq!(counts.data, vec![1.0, 1.0]);
    }
}
