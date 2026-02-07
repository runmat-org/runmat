//! MATLAB-compatible `sortrows` builtin with GPU-aware semantics.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, SortComparison as ProviderSortComparison, SortOrder as ProviderSortOrder,
    SortResult as ProviderSortResult, SortRowsColumnSpec as ProviderSortRowsColumnSpec,
};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use super::type_resolvers::tensor_output_type;
use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::sortrows")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sortrows",
    op_kind: GpuOpKind::Custom("sortrows"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sortrows")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers may implement a row-sort kernel; explicit MissingPlacement overrides fall back to host memory until native support exists.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::sortrows"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sortrows",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`sortrows` terminates fusion chains and materialises results on the host; upstream tensors are gathered when necessary.",
};

fn sortrows_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("sortrows")
        .build()
}

#[runtime_builtin(
    name = "sortrows",
    category = "array/sorting_sets",
    summary = "Sort matrix rows lexicographically with optional column and direction control.",
    keywords = "sortrows,row sort,lexicographic,gpu",
    accel = "sink",
    sink = true,
    type_resolver(tensor_output_type),
    builtin_path = "crate::builtins::array::sorting_sets::sortrows"
)]
async fn sortrows_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let (sorted, indices) = eval.into_values();
        let mut outputs = vec![sorted];
        if out_count >= 2 {
            outputs.push(indices);
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            outputs,
        ));
    }
    Ok(eval.into_sorted_value())
}

/// Evaluate the `sortrows` builtin once and expose both outputs.
pub async fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
    match value {
        Value::GpuTensor(handle) => sortrows_gpu(handle, rest).await,
        other => sortrows_host(other, rest),
    }
}

async fn sortrows_gpu(
    handle: GpuTensorHandle,
    rest: &[Value],
) -> crate::BuiltinResult<SortRowsEvaluation> {
    ensure_matrix_shape(&handle.shape)?;
    let (_, cols) = rows_cols_from_shape(&handle.shape);
    let args = SortRowsArgs::parse(rest, cols)?;

    if args.missing_is_auto() {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let provider_columns = args.to_provider_columns();
            let provider_comparison = args.provider_comparison();
            match provider
                .sort_rows(&handle, &provider_columns, provider_comparison)
                .await
            {
                Ok(result) => return sortrows_from_provider_result(result),
                Err(_err) => {
                    // fall back to host path when provider cannot service the request
                }
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    sortrows_real_tensor_with_args(tensor, &args)
}

fn sortrows_from_provider_result(
    result: ProviderSortResult,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let sorted_tensor = Tensor::new(result.values.data, result.values.shape)
        .map_err(|e| sortrows_error(format!("sortrows: {e}")))?;
    let indices_tensor = Tensor::new(result.indices.data, result.indices.shape)
        .map_err(|e| sortrows_error(format!("sortrows: {e}")))?;
    Ok(SortRowsEvaluation {
        sorted: tensor::tensor_into_value(sorted_tensor),
        indices: indices_tensor,
    })
}

fn sortrows_host(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
    match value {
        Value::Tensor(tensor) => sortrows_real_tensor(tensor, rest),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| sortrows_error(e))?;
            sortrows_real_tensor(tensor, rest)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("sortrows", value)
                .map_err(|e| sortrows_error(e))?;
            sortrows_real_tensor(tensor, rest)
        }
        Value::ComplexTensor(ct) => sortrows_complex_tensor(ct, rest),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| sortrows_error(format!("sortrows: {e}")))?;
            sortrows_complex_tensor(tensor, rest)
        }
        Value::CharArray(ca) => sortrows_char_array(ca, rest),
        other => Err(sortrows_error(format!(
            "sortrows: unsupported input type {:?}; expected numeric, logical, complex, or char arrays",
            other
        ))
        .into()),
    }
}

fn sortrows_real_tensor(
    tensor: Tensor,
    rest: &[Value],
) -> crate::BuiltinResult<SortRowsEvaluation> {
    ensure_matrix_shape(&tensor.shape)?;
    let cols = tensor.cols();
    let args = SortRowsArgs::parse(rest, cols)?;
    sortrows_real_tensor_with_args(tensor, &args)
}

fn sortrows_real_tensor_with_args(
    tensor: Tensor,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let rows = tensor.rows();
    let cols = tensor.cols();

    if rows <= 1 || cols == 0 || tensor.data.is_empty() || args.columns.is_empty() {
        let indices = identity_indices(rows)?;
        return Ok(SortRowsEvaluation {
            sorted: tensor::tensor_into_value(tensor),
            indices,
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| compare_real_rows(&tensor, rows, args, a, b));

    let sorted_tensor = reorder_real_rows(&tensor, rows, cols, &order)?;
    let indices = permutation_indices(&order)?;
    Ok(SortRowsEvaluation {
        sorted: tensor::tensor_into_value(sorted_tensor),
        indices,
    })
}

fn sortrows_complex_tensor(
    tensor: ComplexTensor,
    rest: &[Value],
) -> crate::BuiltinResult<SortRowsEvaluation> {
    ensure_matrix_shape(&tensor.shape)?;
    let cols = tensor.cols;
    let args = SortRowsArgs::parse(rest, cols)?;
    sortrows_complex_tensor_with_args(tensor, &args)
}

fn sortrows_complex_tensor_with_args(
    tensor: ComplexTensor,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let rows = tensor.rows;
    let cols = tensor.cols;

    if rows <= 1 || cols == 0 || tensor.data.is_empty() || args.columns.is_empty() {
        let indices = identity_indices(rows)?;
        return Ok(SortRowsEvaluation {
            sorted: complex_tensor_into_value(tensor),
            indices,
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| compare_complex_rows(&tensor, rows, args, a, b));

    let sorted_tensor = reorder_complex_rows(&tensor, rows, cols, &order)?;
    let indices = permutation_indices(&order)?;
    Ok(SortRowsEvaluation {
        sorted: complex_tensor_into_value(sorted_tensor),
        indices,
    })
}

fn sortrows_char_array(ca: CharArray, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
    let cols = ca.cols;
    let args = SortRowsArgs::parse(rest, cols)?;
    sortrows_char_array_with_args(ca, &args)
}

fn sortrows_char_array_with_args(
    ca: CharArray,
    args: &SortRowsArgs,
) -> crate::BuiltinResult<SortRowsEvaluation> {
    let rows = ca.rows;
    let cols = ca.cols;

    if rows <= 1 || cols == 0 || ca.data.is_empty() || args.columns.is_empty() {
        let indices = identity_indices(rows)?;
        return Ok(SortRowsEvaluation {
            sorted: Value::CharArray(ca),
            indices,
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| compare_char_rows(&ca, args, a, b));

    let sorted = reorder_char_rows(&ca, rows, cols, &order)?;
    let indices = permutation_indices(&order)?;
    Ok(SortRowsEvaluation {
        sorted: Value::CharArray(sorted),
        indices,
    })
}

fn ensure_matrix_shape(shape: &[usize]) -> crate::BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(sortrows_error("sortrows: input must be a 2-D matrix"))
    }
}

fn rows_cols_from_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        _ => (shape[0], shape[1]),
    }
}

fn compare_real_rows(
    tensor: &Tensor,
    rows: usize,
    args: &SortRowsArgs,
    a: usize,
    b: usize,
) -> Ordering {
    for spec in &args.columns {
        if spec.index >= tensor.cols() {
            continue;
        }
        let idx_a = a + spec.index * rows;
        let idx_b = b + spec.index * rows;
        let va = tensor.data[idx_a];
        let vb = tensor.data[idx_b];
        let missing = args.missing_for_direction(spec.direction);
        let ord = compare_real_scalars(va, vb, spec.direction, args.comparison, missing);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_complex_rows(
    tensor: &ComplexTensor,
    rows: usize,
    args: &SortRowsArgs,
    a: usize,
    b: usize,
) -> Ordering {
    for spec in &args.columns {
        if spec.index >= tensor.cols {
            continue;
        }
        let idx_a = a + spec.index * rows;
        let idx_b = b + spec.index * rows;
        let va = tensor.data[idx_a];
        let vb = tensor.data[idx_b];
        let missing = args.missing_for_direction(spec.direction);
        let ord = compare_complex_scalars(va, vb, spec.direction, args.comparison, missing);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_char_rows(ca: &CharArray, args: &SortRowsArgs, a: usize, b: usize) -> Ordering {
    for spec in &args.columns {
        if spec.index >= ca.cols {
            continue;
        }
        let idx_a = a * ca.cols + spec.index;
        let idx_b = b * ca.cols + spec.index;
        let va = ca.data[idx_a];
        let vb = ca.data[idx_b];
        let ord = match spec.direction {
            SortDirection::Ascend => va.cmp(&vb),
            SortDirection::Descend => vb.cmp(&va),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn reorder_real_rows(
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    order: &[usize],
) -> crate::BuiltinResult<Tensor> {
    let mut data = vec![0.0; tensor.data.len()];
    for col in 0..cols {
        for (dest_row, &src_row) in order.iter().enumerate() {
            let src_idx = src_row + col * rows;
            let dst_idx = dest_row + col * rows;
            data[dst_idx] = tensor.data[src_idx];
        }
    }
    Tensor::new(data, tensor.shape.clone()).map_err(|e| sortrows_error(format!("sortrows: {e}")))
}

fn reorder_complex_rows(
    tensor: &ComplexTensor,
    rows: usize,
    cols: usize,
    order: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    let mut data = vec![(0.0, 0.0); tensor.data.len()];
    for col in 0..cols {
        for (dest_row, &src_row) in order.iter().enumerate() {
            let src_idx = src_row + col * rows;
            let dst_idx = dest_row + col * rows;
            data[dst_idx] = tensor.data[src_idx];
        }
    }
    ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| sortrows_error(format!("sortrows: {e}")))
}

fn reorder_char_rows(
    ca: &CharArray,
    rows: usize,
    cols: usize,
    order: &[usize],
) -> crate::BuiltinResult<CharArray> {
    let mut data = vec!['\0'; ca.data.len()];
    for (dest_row, &src_row) in order.iter().enumerate() {
        for col in 0..cols {
            let src_idx = src_row * cols + col;
            let dst_idx = dest_row * cols + col;
            data[dst_idx] = ca.data[src_idx];
        }
    }
    CharArray::new(data, rows, cols).map_err(|e| sortrows_error(format!("sortrows: {e}")))
}

fn compare_real_scalars(
    a: f64,
    b: f64,
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_real_finite_scalars(a, b, direction, comparison),
    }
}

fn compare_real_finite_scalars(
    a: f64,
    b: f64,
    direction: SortDirection,
    comparison: ComparisonMethod,
) -> Ordering {
    if matches!(comparison, ComparisonMethod::Abs) {
        let abs_cmp = a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal);
        if abs_cmp != Ordering::Equal {
            return match direction {
                SortDirection::Ascend => abs_cmp,
                SortDirection::Descend => abs_cmp.reverse(),
            };
        }
    }
    match direction {
        SortDirection::Ascend => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        SortDirection::Descend => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

fn compare_complex_scalars(
    a: (f64, f64),
    b: (f64, f64),
    direction: SortDirection,
    comparison: ComparisonMethod,
    missing: MissingPlacementResolved,
) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => match missing {
            MissingPlacementResolved::First => Ordering::Less,
            MissingPlacementResolved::Last => Ordering::Greater,
        },
        (false, true) => match missing {
            MissingPlacementResolved::First => Ordering::Greater,
            MissingPlacementResolved::Last => Ordering::Less,
        },
        (false, false) => compare_complex_finite_scalars(a, b, direction, comparison),
    }
}

fn compare_complex_finite_scalars(
    a: (f64, f64),
    b: (f64, f64),
    direction: SortDirection,
    comparison: ComparisonMethod,
) -> Ordering {
    match comparison {
        ComparisonMethod::Real => compare_complex_real_first(a, b, direction),
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            let abs_cmp = complex_abs(a)
                .partial_cmp(&complex_abs(b))
                .unwrap_or(Ordering::Equal);
            if abs_cmp != Ordering::Equal {
                return match direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            compare_complex_real_first(a, b, direction)
        }
    }
}

fn compare_complex_real_first(a: (f64, f64), b: (f64, f64), direction: SortDirection) -> Ordering {
    let real_cmp = match direction {
        SortDirection::Ascend => a.0.partial_cmp(&b.0),
        SortDirection::Descend => b.0.partial_cmp(&a.0),
    }
    .unwrap_or(Ordering::Equal);
    if real_cmp != Ordering::Equal {
        return real_cmp;
    }
    match direction {
        SortDirection::Ascend => a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal),
        SortDirection::Descend => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

fn complex_is_nan(value: (f64, f64)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_abs(value: (f64, f64)) -> f64 {
    value.0.hypot(value.1)
}

fn permutation_indices(order: &[usize]) -> crate::BuiltinResult<Tensor> {
    let rows = order.len();
    let mut data = Vec::with_capacity(rows);
    for &idx in order {
        data.push((idx + 1) as f64);
    }
    Tensor::new(data, vec![rows, 1]).map_err(|e| sortrows_error(format!("sortrows: {e}")))
}

fn identity_indices(rows: usize) -> crate::BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(rows);
    for i in 0..rows {
        data.push((i + 1) as f64);
    }
    Tensor::new(data, vec![rows, 1]).map_err(|e| sortrows_error(format!("sortrows: {e}")))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        Value::Complex(tensor.data[0].0, tensor.data[0].1)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortDirection {
    Ascend,
    Descend,
}

impl SortDirection {
    fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ascend" | "ascending" => Some(SortDirection::Ascend),
            "descend" | "descending" => Some(SortDirection::Descend),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonMethod {
    Auto,
    Real,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingPlacement {
    Auto,
    First,
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingPlacementResolved {
    First,
    Last,
}

impl MissingPlacement {
    fn resolve(self, direction: SortDirection) -> MissingPlacementResolved {
        match self {
            MissingPlacement::First => MissingPlacementResolved::First,
            MissingPlacement::Last => MissingPlacementResolved::Last,
            MissingPlacement::Auto => match direction {
                SortDirection::Ascend => MissingPlacementResolved::Last,
                SortDirection::Descend => MissingPlacementResolved::First,
            },
        }
    }

    fn is_auto(self) -> bool {
        matches!(self, MissingPlacement::Auto)
    }
}

#[derive(Debug, Clone)]
struct ColumnSpec {
    index: usize,
    direction: SortDirection,
}

#[derive(Debug, Clone)]
struct SortRowsArgs {
    columns: Vec<ColumnSpec>,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
}

impl SortRowsArgs {
    fn parse(rest: &[Value], num_cols: usize) -> crate::BuiltinResult<Self> {
        let mut columns: Option<Vec<ColumnSpec>> = None;
        let mut override_direction: Option<SortDirection> = None;
        let mut comparison = ComparisonMethod::Auto;
        let mut missing = MissingPlacement::Auto;
        let mut i = 0usize;

        while i < rest.len() {
            if columns.is_none() {
                if let Some(parsed) = parse_column_vector(&rest[i], num_cols)? {
                    columns = Some(parsed);
                    i += 1;
                    continue;
                }
            }
            if let Some(direction) = parse_direction(&rest[i]) {
                override_direction = Some(direction);
                i += 1;
                continue;
            }
            let Some(keyword) = tensor::value_to_string(&rest[i]) else {
                return Err(sortrows_error(format!(
                    "sortrows: invalid argument {:?}",
                    rest[i]
                )));
            };
            let lowered = keyword.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "comparisonmethod" => {
                    i += 1;
                    if i >= rest.len() {
                        return Err(sortrows_error(
                            "sortrows: expected a value for 'ComparisonMethod'",
                        ));
                    }
                    let Some(value_str) = tensor::value_to_string(&rest[i]) else {
                        return Err(sortrows_error(
                            "sortrows: 'ComparisonMethod' expects a string value",
                        )
                        .into());
                    };
                    comparison = match value_str.trim().to_ascii_lowercase().as_str() {
                        "auto" => ComparisonMethod::Auto,
                        "real" => ComparisonMethod::Real,
                        "abs" | "magnitude" => ComparisonMethod::Abs,
                        other => {
                            return Err(sortrows_error(format!(
                                "sortrows: unsupported ComparisonMethod '{other}'"
                            ))
                            .into())
                        }
                    };
                    i += 1;
                }
                "missingplacement" => {
                    i += 1;
                    if i >= rest.len() {
                        return Err(sortrows_error(
                            "sortrows: expected a value for 'MissingPlacement'",
                        )
                        .into());
                    }
                    let Some(value_str) = tensor::value_to_string(&rest[i]) else {
                        return Err(sortrows_error(
                            "sortrows: 'MissingPlacement' expects a string value",
                        )
                        .into());
                    };
                    missing = match value_str.trim().to_ascii_lowercase().as_str() {
                        "auto" => MissingPlacement::Auto,
                        "first" => MissingPlacement::First,
                        "last" => MissingPlacement::Last,
                        other => {
                            return Err(sortrows_error(format!(
                                "sortrows: unsupported MissingPlacement '{other}'"
                            ))
                            .into())
                        }
                    };
                    i += 1;
                }
                other => {
                    return Err(sortrows_error(format!(
                        "sortrows: unexpected argument '{other}'"
                    )));
                }
            }
        }

        let mut columns = columns.unwrap_or_else(|| default_columns(num_cols));
        if let Some(dir) = override_direction {
            for spec in &mut columns {
                spec.direction = dir;
            }
        }
        validate_columns(&columns, num_cols)?;

        Ok(SortRowsArgs {
            columns,
            comparison,
            missing,
        })
    }

    fn to_provider_columns(&self) -> Vec<ProviderSortRowsColumnSpec> {
        self.columns
            .iter()
            .map(|spec| ProviderSortRowsColumnSpec {
                index: spec.index,
                order: match spec.direction {
                    SortDirection::Ascend => ProviderSortOrder::Ascend,
                    SortDirection::Descend => ProviderSortOrder::Descend,
                },
            })
            .collect()
    }

    fn provider_comparison(&self) -> ProviderSortComparison {
        match self.comparison {
            ComparisonMethod::Auto => ProviderSortComparison::Auto,
            ComparisonMethod::Real => ProviderSortComparison::Real,
            ComparisonMethod::Abs => ProviderSortComparison::Abs,
        }
    }

    fn missing_for_direction(&self, direction: SortDirection) -> MissingPlacementResolved {
        self.missing.resolve(direction)
    }

    fn missing_is_auto(&self) -> bool {
        self.missing.is_auto()
    }
}

fn parse_column_vector(
    value: &Value,
    num_cols: usize,
) -> crate::BuiltinResult<Option<Vec<ColumnSpec>>> {
    match value {
        Value::Int(i) => parse_single_column(i.to_i64(), num_cols).map(Some),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(sortrows_error("sortrows: column indices must be finite"));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(sortrows_error("sortrows: column indices must be integers"));
            }
            parse_single_column(rounded as i64, num_cols).map(Some)
        }
        Value::Tensor(tensor) => {
            if !is_vector(&tensor.shape) {
                return Err(sortrows_error(
                    "sortrows: column specification must be a vector",
                ));
            }
            let mut specs = Vec::with_capacity(tensor.data.len());
            for &entry in &tensor.data {
                if !entry.is_finite() {
                    return Err(sortrows_error("sortrows: column indices must be finite"));
                }
                let rounded = entry.round();
                if (rounded - entry).abs() > f64::EPSILON {
                    return Err(sortrows_error("sortrows: column indices must be integers"));
                }
                let column = parse_single_column_i64(rounded as i64, num_cols)?;
                specs.push(column);
            }
            Ok(Some(specs))
        }
        _ => Ok(None),
    }
}

fn parse_single_column(value: i64, num_cols: usize) -> crate::BuiltinResult<Vec<ColumnSpec>> {
    parse_single_column_i64(value, num_cols).map(|spec| vec![spec])
}

fn parse_single_column_i64(value: i64, num_cols: usize) -> crate::BuiltinResult<ColumnSpec> {
    if value == 0 {
        return Err(sortrows_error("sortrows: column indices must be non-zero"));
    }
    let abs = value.unsigned_abs() as usize;
    if abs == 0 {
        return Err(sortrows_error("sortrows: column indices must be >= 1"));
    }
    if num_cols == 0 {
        return Err(sortrows_error(
            "sortrows: column index exceeds matrix with 0 columns",
        ));
    }
    if abs > num_cols {
        return Err(sortrows_error(format!(
            "sortrows: column index {} exceeds matrix with {} columns",
            abs, num_cols
        ))
        .into());
    }
    let direction = if value > 0 {
        SortDirection::Ascend
    } else {
        SortDirection::Descend
    };
    Ok(ColumnSpec {
        index: abs - 1,
        direction,
    })
}

fn parse_direction(value: &Value) -> Option<SortDirection> {
    tensor::value_to_string(value).and_then(|s| SortDirection::from_str(&s))
}

fn default_columns(num_cols: usize) -> Vec<ColumnSpec> {
    let mut columns = Vec::with_capacity(num_cols);
    for col in 0..num_cols {
        columns.push(ColumnSpec {
            index: col,
            direction: SortDirection::Ascend,
        });
    }
    columns
}

fn validate_columns(columns: &[ColumnSpec], num_cols: usize) -> crate::BuiltinResult<()> {
    if num_cols == 0 && columns.iter().any(|spec| spec.index > 0) {
        return Err(sortrows_error(
            "sortrows: column index exceeds matrix with 0 columns",
        ));
    }
    for spec in columns {
        if num_cols > 0 && spec.index >= num_cols {
            return Err(sortrows_error(format!(
                "sortrows: column index {} exceeds matrix with {} columns",
                spec.index + 1,
                num_cols
            ))
            .into());
        }
    }
    Ok(())
}

fn is_vector(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

#[derive(Debug)]
pub struct SortRowsEvaluation {
    sorted: Value,
    indices: Tensor,
}

impl SortRowsEvaluation {
    pub fn into_sorted_value(self) -> Value {
        self.sorted
    }

    pub fn into_values(self) -> (Value, Value) {
        let indices = tensor::tensor_into_value(self.indices);
        (self.sorted, indices)
    }

    pub fn indices_value(&self) -> Value {
        tensor::tensor_into_value(self.indices.clone())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, ResolveContext, Type, Value};

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortRowsEvaluation> {
        futures::executor::block_on(super::evaluate(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_default_matrix() {
        let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0, 1.0, 5.0], vec![3, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 1.0, 5.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[test]
    fn sortrows_type_resolver_tensor() {
        assert_eq!(
            tensor_output_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_with_column_vector() {
        let tensor = Tensor::new(
            vec![1.0, 3.0, 3.0, 4.0, 2.0, 2.0, 2.0, 5.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        let cols = Tensor::new(vec![2.0, 3.0, 1.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::Tensor(cols)]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![3.0, 3.0, 1.0, 2.0, 2.0, 4.0, 1.0, 5.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_direction_descend() {
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::from("descend")]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0, 3.0, 4.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_mixed_directions() {
        let tensor = Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 7.0, 2.0], vec![3, 2]).unwrap();
        let cols = Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[Value::Tensor(cols)]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0, 1.0, 7.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_returns_indices() {
        let tensor = Tensor::new(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (_, indices) = eval.into_values();
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_char_array() {
        let chars = CharArray::new(
            "bob "
                .chars()
                .chain("al  ".chars())
                .chain("ally".chars())
                .collect(),
            3,
            4,
        )
        .unwrap();
        let eval = evaluate(Value::CharArray(chars), &[]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 3);
                assert_eq!(ca.cols, 4);
                let strings: Vec<String> = (0..ca.rows)
                    .map(|r| {
                        ca.data[r * ca.cols..(r + 1) * ca.cols]
                            .iter()
                            .collect::<String>()
                    })
                    .collect();
                assert_eq!(
                    strings,
                    vec!["al  ".to_string(), "ally".to_string(), "bob ".to_string()]
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_complex_abs() {
        let tensor = ComplexTensor::new(vec![(1.0, 2.0), (-2.0, 1.0)], vec![2, 1]).unwrap();
        let eval = evaluate(
            Value::ComplexTensor(tensor),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.data, vec![(-2.0, 1.0), (1.0, 2.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_invalid_column_index_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = error_message(
            evaluate(Value::Tensor(tensor), &[Value::Int(IntValue::I32(3))]).unwrap_err(),
        );
        assert!(
            err.contains("column index"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_missingplacement_first_moves_nan_first() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("MissingPlacement"), Value::from("first")],
        )
        .expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert!(t.data[0].is_nan());
                assert_eq!(t.data[1], 1.0);
                assert_eq!(t.data[2], 3.0);
                assert_eq!(t.data[3], 2.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_missingplacement_last_descend_moves_nan_last() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("descend"),
                Value::from("MissingPlacement"),
                Value::from("last"),
            ],
        )
        .expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.data[0], 5.0);
                assert!(t.data[1].is_nan());
                assert_eq!(t.data[2], 2.0);
                assert_eq!(t.data[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0]),
            Value::Num(_) => panic!("expected tensor indices"),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_missingplacement_invalid_value_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = error_message(
            evaluate(
                Value::Tensor(tensor),
                &[Value::from("MissingPlacement"), Value::from("middle")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("MissingPlacement"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sortrows_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0, 1.0, 5.0], vec![3, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");
            let (sorted, indices) = eval.into_values();
            match sorted {
                Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 3.0, 1.0, 5.0, 4.0]),
                other => panic!("expected tensor, got {other:?}"),
            }
            match indices {
                Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
                other => panic!("unexpected indices {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sortrows_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![4.0, 2.0, 3.0, 1.0, 2.0, 5.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu evaluate");
        let (cpu_sorted_val, cpu_indices_val) = cpu_eval.into_values();
        let cpu_sorted = match cpu_sorted_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let cpu_indices = match cpu_indices_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices, got {other:?}"),
        };

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle.clone()), &[]).expect("gpu evaluate");
        let (gpu_sorted_val, gpu_indices_val) = gpu_eval.into_values();
        let gpu_sorted = match gpu_sorted_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let gpu_indices = match gpu_indices_val {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices, got {other:?}"),
        };

        assert_eq!(gpu_sorted.shape, cpu_sorted.shape);
        assert_eq!(gpu_sorted.data, cpu_sorted.data);
        assert_eq!(gpu_indices.shape, cpu_indices.shape);
        assert_eq!(gpu_indices.data, cpu_indices.data);

        let _ = provider.free(&handle);
    }
}
