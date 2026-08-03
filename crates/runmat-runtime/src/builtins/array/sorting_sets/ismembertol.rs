//! MATLAB-compatible `ismembertol` builtin for real numeric arrays.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, LiteralValue, LogicalArray, NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use super::type_resolvers::logical_output_type;
use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
    tensor,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "ismembertol";
const DEFAULT_TOL_DOUBLE: f64 = 1.0e-12;
const DEFAULT_TOL_SINGLE: f64 = 1.0e-6;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::sorting_sets::ismembertol"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("ismembertol"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Tolerance membership is host materialised today; gpuArray inputs are gathered before comparison.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::ismembertol"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`ismembertol` materialises logical and index outputs and terminates fusion chains.",
};

const OUTPUT_MASK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "LIA",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask over A.",
}];

const OUTPUT_MASK_LOC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "LIA",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical mask over A.",
    },
    BuiltinParamDescriptor {
        name: "LocB",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First or all matching indices into B.",
    },
];

const INPUTS_AB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query values or rows.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference values or rows.",
    },
];

const INPUTS_AB_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query values or rows.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reference values or rows.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1e-12 for double, 1e-6 for single"),
        description: "Relative tolerance.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: ByRows, DataScale, OutputAllIndices.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "LIA = ismembertol(A, B)",
        inputs: &INPUTS_AB,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "LIA = ismembertol(A, B, tol, Name, Value)",
        inputs: &INPUTS_AB_OPTIONS,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "[LIA, LocB] = ismembertol(A, B)",
        inputs: &INPUTS_AB,
        outputs: &OUTPUT_MASK_LOC,
    },
    BuiltinSignatureDescriptor {
        label: "[LIA, LocB] = ismembertol(A, B, tol, Name, Value)",
        inputs: &INPUTS_AB_OPTIONS,
        outputs: &OUTPUT_MASK_LOC,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.INVALID_INPUT",
    identifier: Some("RunMat:ismembertol:InvalidInput"),
    when: "A or B is not a supported real full numeric input.",
    message: "ismembertol: inputs must be real full numeric arrays",
};

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ismembertol:InvalidArgument"),
    when: "Tolerance or name-value arguments are malformed.",
    message: "ismembertol: invalid argument",
};

const ERROR_ROWS_COLUMN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.ROWS_COLUMN_MISMATCH",
    identifier: Some("RunMat:ismembertol:RowsColumnMismatch"),
    when: "ByRows is true and A/B column counts differ.",
    message: "ismembertol: inputs must have the same number of columns when ByRows is true",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMEMBERTOL.INTERNAL",
    identifier: Some("RunMat:ismembertol:Internal"),
    when: "Internal conversion or allocation fails.",
    message: "ismembertol: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_INPUT,
    ERROR_INVALID_ARGUMENT,
    ERROR_ROWS_COLUMN_MISMATCH,
    ERROR_INTERNAL,
];

pub const ISMEMBERTOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn ismembertol_output_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if literal_by_rows_enabled(ctx) {
        return match args.first().and_then(type_row_count) {
            Some(rows) => Type::Logical {
                shape: Some(vec![rows, Some(1)]),
            },
            None => Type::logical(),
        };
    }
    logical_output_type(args, ctx)
}

fn literal_by_rows_enabled(ctx: &ResolveContext) -> bool {
    let mut idx = 2usize;
    if !matches!(ctx.literal_args.get(idx), Some(LiteralValue::String(_))) {
        idx += 1;
    }
    while idx + 1 < ctx.literal_args.len() {
        if matches!(
            ctx.literal_args.get(idx),
            Some(LiteralValue::String(name)) if name.eq_ignore_ascii_case("ByRows")
        ) {
            return literal_truthy(ctx.literal_args.get(idx + 1));
        }
        idx += 2;
    }
    false
}

fn literal_truthy(value: Option<&LiteralValue>) -> bool {
    match value {
        Some(LiteralValue::Bool(flag)) => *flag,
        Some(LiteralValue::Number(num)) => *num == 1.0,
        _ => false,
    }
}

fn type_row_count(ty: &Type) -> Option<Option<usize>> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if shape.is_empty() {
                Some(Some(1))
            } else {
                Some(shape[0])
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Some(None),
        Type::Num | Type::Int | Type::Bool => Some(Some(1)),
        _ => None,
    }
}

#[runtime_builtin(
    name = "ismembertol",
    category = "array/sorting_sets",
    summary = "Identify numeric array elements or rows that are within tolerance of another array.",
    keywords = "ismembertol,membership,tolerance,set,rows,indices",
    accel = "array_construct",
    sink = true,
    type_resolver(ismembertol_output_type),
    descriptor(crate::builtins::array::sorting_sets::ismembertol::ISMEMBERTOL_DESCRIPTOR),
    builtin_path = "crate::builtins::array::sorting_sets::ismembertol"
)]
async fn ismembertol_builtin(a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let opts = parse_options(&rest)?;
        if out_count == 1 {
            let eval = evaluate_with_options(a, b, opts, LocRequest::None).await?;
            return Ok(Value::OutputList(vec![eval.into_mask_value()]));
        }
        let loc_request = if opts.output_all_indices {
            LocRequest::All
        } else {
            LocRequest::First
        };
        let eval = evaluate_with_options(a, b, opts, loc_request).await?;
        let (mask, loc) = eval.into_pair();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![mask, loc],
        ));
    }
    let opts = parse_options(&rest)?;
    let eval = evaluate_with_options(a, b, opts, LocRequest::None).await?;
    Ok(eval.into_mask_value())
}

pub async fn evaluate(a: Value, b: Value, rest: &[Value]) -> BuiltinResult<IsMemberTolEvaluation> {
    let opts = parse_options(rest)?;
    let loc_request = if opts.output_all_indices {
        LocRequest::All
    } else {
        LocRequest::First
    };
    evaluate_with_options(a, b, opts, loc_request).await
}

async fn evaluate_with_options(
    a: Value,
    b: Value,
    opts: IsMemberTolOptions,
    loc_request: LocRequest,
) -> BuiltinResult<IsMemberTolEvaluation> {
    let a = gather_if_gpu(a).await?;
    let b = gather_if_gpu(b).await?;
    let tensor_a =
        tensor::value_into_tensor_for(NAME, a).map_err(|_| error(&ERROR_INVALID_INPUT))?;
    let tensor_b =
        tensor::value_into_tensor_for(NAME, b).map_err(|_| error(&ERROR_INVALID_INPUT))?;
    evaluate_tensors(tensor_a, tensor_b, opts, loc_request)
}

async fn gather_if_gpu(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}"))),
        other => Ok(other),
    }
}

fn evaluate_tensors(
    a: Tensor,
    b: Tensor,
    mut opts: IsMemberTolOptions,
    loc_request: LocRequest,
) -> BuiltinResult<IsMemberTolEvaluation> {
    if opts.tol.is_none() {
        opts.tol = Some(default_tolerance(a.numeric_dtype(), b.numeric_dtype()));
    }
    if opts.by_rows {
        evaluate_rows(a, b, opts, loc_request)
    } else {
        evaluate_elements(a, b, opts, loc_request)
    }
}

#[derive(Debug, Clone)]
struct IsMemberTolOptions {
    tol: Option<f64>,
    by_rows: bool,
    data_scale: Option<Vec<f64>>,
    output_all_indices: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum LocRequest {
    None,
    First,
    All,
}

fn parse_options(rest: &[Value]) -> BuiltinResult<IsMemberTolOptions> {
    let mut opts = IsMemberTolOptions {
        tol: None,
        by_rows: false,
        data_scale: None,
        output_all_indices: false,
    };

    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if option_name(first).is_none() {
            opts.tol = Some(parse_positive_scalar(first, "tolerance")?);
            idx = 1;
        }
    }

    let remaining = &rest[idx..];
    if !remaining.len().is_multiple_of(2) {
        return Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            "ismembertol: name-value arguments must appear in pairs",
        ));
    }

    for pair in remaining.chunks_exact(2) {
        let name = option_name(&pair[0])
            .ok_or_else(|| {
                error_with(&ERROR_INVALID_ARGUMENT, "ismembertol: expected option name")
            })?
            .trim()
            .to_ascii_lowercase();
        match name.as_str() {
            "byrows" => opts.by_rows = parse_bool_option(&pair[1])?,
            "outputallindices" => opts.output_all_indices = parse_bool_option(&pair[1])?,
            "datascale" => opts.data_scale = Some(parse_data_scale(&pair[1])?),
            other => {
                return Err(error_with(
                    &ERROR_INVALID_ARGUMENT,
                    format!("ismembertol: unknown option '{other}'"),
                ))
            }
        }
    }

    Ok(opts)
}

fn option_name(value: &Value) -> Option<String> {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            tensor::value_to_string(value)
        }
        _ => None,
    }
}

fn evaluate_elements(
    a: Tensor,
    b: Tensor,
    opts: IsMemberTolOptions,
    loc_request: LocRequest,
) -> BuiltinResult<IsMemberTolEvaluation> {
    let a_shape = a.shape.clone();
    let a_values = materialize_tolerance_values(a)?;
    let b_values = materialize_tolerance_values(b)?;
    let tol = opts.tol.expect("defaulted tolerance");
    let scale = opts
        .data_scale
        .as_ref()
        .map(|values| {
            if values.len() == 1 {
                Ok(values[0])
            } else {
                Err(error_with(
                    &ERROR_INVALID_ARGUMENT,
                    "ismembertol: DataScale must be scalar unless ByRows is true",
                ))
            }
        })
        .transpose()?
        .unwrap_or_else(|| default_element_scale(&a_values, &b_values));
    let threshold = tol * scale.abs();

    let mut mask_data = Vec::with_capacity(a_values.len());
    let mut loc_data = match loc_request {
        LocRequest::First => Vec::with_capacity(a_values.len()),
        LocRequest::None | LocRequest::All => Vec::new(),
    };
    let mut all_cells = match loc_request {
        LocRequest::All => Vec::with_capacity(a_values.len()),
        LocRequest::None | LocRequest::First => Vec::new(),
    };

    for &value in a_values.iter() {
        match loc_request {
            LocRequest::None => {
                mask_data.push(
                    if first_element_match(value, &b_values, threshold).is_some() {
                        1
                    } else {
                        0
                    },
                );
            }
            LocRequest::First => {
                let first = first_element_match(value, &b_values, threshold);
                mask_data.push(if first.is_some() { 1 } else { 0 });
                loc_data.push(first.unwrap_or(0) as f64);
            }
            LocRequest::All => {
                let matches = all_element_matches(value, &b_values, threshold);
                mask_data.push(if matches.is_empty() { 0 } else { 1 });
                all_cells.push(indices_cell(matches)?);
            }
        }
    }

    let mask = LogicalArray::new(mask_data, a_shape.clone())
        .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))?;
    let loc = match loc_request {
        LocRequest::None => None,
        LocRequest::First => Some(LocOutput::First(
            Tensor::new(loc_data, a_shape.clone())
                .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))?,
        )),
        LocRequest::All => Some(LocOutput::All(cell_array_from_linear_matches(
            all_cells, a_shape,
        )?)),
    };
    Ok(IsMemberTolEvaluation { mask, loc })
}

fn evaluate_rows(
    a: Tensor,
    b: Tensor,
    opts: IsMemberTolOptions,
    loc_request: LocRequest,
) -> BuiltinResult<IsMemberTolEvaluation> {
    let (rows_a, cols_a) = rows_cols(&a)?;
    let (rows_b, cols_b) = rows_cols(&b)?;
    if cols_a != cols_b {
        return Err(error(&ERROR_ROWS_COLUMN_MISMATCH));
    }
    let a_values = materialize_tolerance_values(a)?;
    let b_values = materialize_tolerance_values(b)?;
    let tol = opts.tol.expect("defaulted tolerance");
    let scales = row_scales(
        &a_values,
        rows_a,
        &b_values,
        rows_b,
        opts.data_scale.as_deref(),
        cols_a,
    )?;
    let thresholds: Vec<f64> = scales.iter().map(|scale| tol * scale.abs()).collect();

    let mut mask_data = vec![0u8; rows_a];
    let mut loc_data = match loc_request {
        LocRequest::First => vec![0.0f64; rows_a],
        LocRequest::None | LocRequest::All => Vec::new(),
    };
    let mut all_cells = match loc_request {
        LocRequest::All => Vec::with_capacity(rows_a),
        LocRequest::None | LocRequest::First => Vec::new(),
    };

    for row_a in 0..rows_a {
        match loc_request {
            LocRequest::None => {
                mask_data[row_a] = if first_row_match(
                    &a_values,
                    row_a,
                    rows_a,
                    &b_values,
                    rows_b,
                    cols_a,
                    &thresholds,
                )
                .is_some()
                {
                    1
                } else {
                    0
                };
            }
            LocRequest::First => {
                let first = first_row_match(
                    &a_values,
                    row_a,
                    rows_a,
                    &b_values,
                    rows_b,
                    cols_a,
                    &thresholds,
                );
                mask_data[row_a] = if first.is_some() { 1 } else { 0 };
                loc_data[row_a] = first.unwrap_or(0) as f64;
            }
            LocRequest::All => {
                let matches = all_row_matches(
                    &a_values,
                    row_a,
                    rows_a,
                    &b_values,
                    rows_b,
                    cols_a,
                    &thresholds,
                );
                mask_data[row_a] = if matches.is_empty() { 0 } else { 1 };
                all_cells.push(indices_cell(matches)?);
            }
        }
    }

    let shape = vec![rows_a, 1];
    let mask = LogicalArray::new(mask_data, shape.clone())
        .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))?;
    let loc = match loc_request {
        LocRequest::None => None,
        LocRequest::First => Some(LocOutput::First(
            Tensor::new(loc_data, shape)
                .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))?,
        )),
        LocRequest::All => Some(LocOutput::All(cell_array_from_linear_matches(
            all_cells, shape,
        )?)),
    };
    Ok(IsMemberTolEvaluation { mask, loc })
}

fn materialize_tolerance_values(tensor: Tensor) -> BuiltinResult<Vec<f64>> {
    // Tolerance and DataScale arithmetic have one intentional f64 computation domain. Native single and exact integer storage remain authoritative until this boundary.
    tensor
        .into_numeric_storage()
        .map(|storage| storage.materialize_f64())
        .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn first_element_match(value: f64, candidates: &[f64], threshold: f64) -> Option<usize> {
    candidates.iter().enumerate().find_map(|(idx, &candidate)| {
        within_tolerance(value, candidate, threshold).then_some(idx + 1)
    })
}

fn all_element_matches(value: f64, candidates: &[f64], threshold: f64) -> Vec<usize> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, &candidate)| {
            within_tolerance(value, candidate, threshold).then_some(idx + 1)
        })
        .collect()
}

fn first_row_match(
    a: &[f64],
    row_a: usize,
    rows_a: usize,
    b: &[f64],
    rows_b: usize,
    cols: usize,
    thresholds: &[f64],
) -> Option<usize> {
    (0..rows_b).find_map(|row_b| {
        rows_match(a, row_a, rows_a, b, row_b, rows_b, cols, thresholds).then_some(row_b + 1)
    })
}

fn all_row_matches(
    a: &[f64],
    row_a: usize,
    rows_a: usize,
    b: &[f64],
    rows_b: usize,
    cols: usize,
    thresholds: &[f64],
) -> Vec<usize> {
    (0..rows_b)
        .filter_map(|row_b| {
            rows_match(a, row_a, rows_a, b, row_b, rows_b, cols, thresholds).then_some(row_b + 1)
        })
        .collect()
}

fn rows_match(
    a: &[f64],
    row_a: usize,
    rows_a: usize,
    b: &[f64],
    row_b: usize,
    rows_b: usize,
    cols: usize,
    thresholds: &[f64],
) -> bool {
    for col in 0..cols {
        let threshold = thresholds[col];
        if threshold.is_infinite() {
            continue;
        }
        let lhs = a[row_a + col * rows_a];
        let rhs = b[row_b + col * rows_b];
        if !within_tolerance(lhs, rhs, threshold) {
            return false;
        }
    }
    true
}

fn within_tolerance(lhs: f64, rhs: f64, threshold: f64) -> bool {
    if lhs.is_nan() || rhs.is_nan() {
        return false;
    }
    if lhs == rhs {
        return true;
    }
    (lhs - rhs).abs() <= threshold
}

fn default_tolerance(a: NumericDType, b: NumericDType) -> f64 {
    if matches!(a, NumericDType::F32) || matches!(b, NumericDType::F32) {
        DEFAULT_TOL_SINGLE
    } else {
        DEFAULT_TOL_DOUBLE
    }
}

fn default_element_scale(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .chain(b.iter())
        .filter_map(|value| (!value.is_nan()).then_some(value.abs()))
        .fold(0.0, f64::max)
}

fn row_scales(
    a: &[f64],
    rows_a: usize,
    b: &[f64],
    rows_b: usize,
    supplied: Option<&[f64]>,
    cols: usize,
) -> BuiltinResult<Vec<f64>> {
    if let Some(values) = supplied {
        if values.len() == 1 {
            return Ok(vec![values[0]; cols]);
        }
        if values.len() == cols {
            return Ok(values.to_vec());
        }
        return Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            "ismembertol: DataScale vector length must match the number of columns",
        ));
    }

    let mut scales = vec![0.0f64; cols];
    for (col, scale) in scales.iter_mut().enumerate() {
        for row in 0..rows_a {
            let value = a[row + col * rows_a];
            if !value.is_nan() {
                *scale = (*scale).max(value.abs());
            }
        }
        for row in 0..rows_b {
            let value = b[row + col * rows_b];
            if !value.is_nan() {
                *scale = (*scale).max(value.abs());
            }
        }
    }
    Ok(scales)
}

fn rows_cols(tensor: &Tensor) -> BuiltinResult<(usize, usize)> {
    match tensor.shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((tensor.shape[0], 1)),
        2 => Ok((tensor.shape[0], tensor.shape[1])),
        _ => Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            "ismembertol: ByRows requires 2-D arrays",
        )),
    }
}

fn indices_cell(indices: Vec<usize>) -> BuiltinResult<Value> {
    if indices.is_empty() {
        return Ok(Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).map_err(
            |e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")),
        )?));
    }
    let data = indices
        .into_iter()
        .map(|idx| idx as f64)
        .collect::<Vec<_>>();
    let len = data.len();
    Tensor::new(data, vec![len, 1])
        .map(Value::Tensor)
        .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn cell_array_from_linear_matches(
    cells_column_major: Vec<Value>,
    shape: Vec<usize>,
) -> BuiltinResult<CellArray> {
    let cells = if shape.len() == 2 && shape[0] > 1 && shape[1] > 1 {
        let rows = shape[0];
        let cols = shape[1];
        let mut row_major = Vec::with_capacity(cells_column_major.len());
        for row in 0..rows {
            for col in 0..cols {
                row_major.push(cells_column_major[row + col * rows].clone());
            }
        }
        row_major
    } else {
        cells_column_major
    };
    CellArray::new_with_shape(cells, shape)
        .map_err(|e| error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn parse_positive_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    let scalar = numeric_scalar(value).ok_or_else(|| {
        error_with(
            &ERROR_INVALID_ARGUMENT,
            format!("ismembertol: {label} must be a positive real scalar"),
        )
    })?;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            format!("ismembertol: {label} must be a positive real scalar"),
        ));
    }
    Ok(scalar)
}

fn parse_bool_option(value: &Value) -> BuiltinResult<bool> {
    match numeric_scalar(value) {
        Some(0.0) => Ok(false),
        Some(1.0) => Ok(true),
        _ => Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            "ismembertol: boolean options must be true/false or 0/1",
        )),
    }
}

fn parse_data_scale(value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => validate_data_scale(vec![*n]),
        Value::Int(i) => validate_data_scale(vec![i.to_f64()]),
        Value::Bool(flag) => validate_data_scale(vec![if *flag { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => validate_data_scale(tensor::tensor_values_f64(tensor)),
        Value::LogicalArray(logical) => validate_data_scale(
            logical
                .data
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect(),
        ),
        _ => Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            "ismembertol: DataScale must be a numeric scalar or vector",
        )),
    }
}

fn validate_data_scale(values: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    if values.is_empty() || values.iter().any(|value| value.is_nan() || *value < 0.0) {
        return Err(error_with(
            &ERROR_INVALID_ARGUMENT,
            "ismembertol: DataScale values must be non-negative or Inf",
        ));
    }
    Ok(values)
}

fn numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            tensor::tensor_values_f64(t).first().copied()
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Some(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct IsMemberTolEvaluation {
    mask: LogicalArray,
    loc: Option<LocOutput>,
}

#[derive(Debug, Clone)]
enum LocOutput {
    First(Tensor),
    All(CellArray),
}

impl IsMemberTolEvaluation {
    pub fn into_mask_value(self) -> Value {
        logical_array_into_value(self.mask)
    }

    pub fn mask_value(&self) -> Value {
        logical_array_into_value(self.mask.clone())
    }

    pub fn into_pair(self) -> (Value, Value) {
        let mask = logical_array_into_value(self.mask);
        let loc = match self.loc.expect("LocB was not requested") {
            LocOutput::First(tensor) => tensor::tensor_into_value(tensor),
            LocOutput::All(cell) => Value::Cell(cell),
        };
        (mask, loc)
    }
}

fn logical_array_into_value(logical: LogicalArray) -> Value {
    if logical.data.len() == 1 {
        Value::Bool(logical.data[0] != 0)
    } else {
        Value::LogicalArray(logical)
    }
}

fn error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    error_with(error, error.message)
}

fn error_with(error: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, LiteralValue, ResolveContext, Type};

    fn eval(a: Value, b: Value, rest: &[Value]) -> BuiltinResult<IsMemberTolEvaluation> {
        block_on(evaluate(a, b, rest))
    }

    #[test]
    fn type_resolver_returns_logical_shape() {
        assert_eq!(
            ismembertol_output_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)])
                }],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Logical {
                shape: Some(vec![Some(1), Some(3)])
            }
        );
    }

    #[test]
    fn type_resolver_returns_row_shape_for_literal_by_rows() {
        assert_eq!(
            ismembertol_output_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(4), Some(3)])
                }],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::Unknown,
                    LiteralValue::Number(0.01),
                    LiteralValue::String("ByRows".to_string()),
                    LiteralValue::Bool(true),
                ]),
            ),
            Type::Logical {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn default_tolerance_uses_scaled_double_data() {
        let a = Tensor::new(vec![0.1, 1.0e10], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![0.1 + 1.0e-14, 1.0e10 + 5.0e-3], vec![1, 2]).unwrap();
        let result = eval(Value::Tensor(a), Value::Tensor(b), &[]).unwrap();
        assert_eq!(result.mask.data, vec![1, 1]);
    }

    #[test]
    fn default_tolerance_uses_native_single_class_before_materialization() {
        let single_a = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let single_b = Tensor::from_f32(vec![1.0 + 5.0e-7], vec![1, 1]).unwrap();
        let single = eval(Value::Tensor(single_a), Value::Tensor(single_b), &[]).unwrap();
        assert_eq!(single.mask_value(), Value::Bool(true));

        let double_a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let double_b = Tensor::new(vec![1.0 + 5.0e-7], vec![1, 1]).unwrap();
        let double = eval(Value::Tensor(double_a), Value::Tensor(double_b), &[]).unwrap();
        assert_eq!(double.mask_value(), Value::Bool(false));
    }

    #[test]
    fn data_scale_one_uses_absolute_tolerance() {
        let a = Tensor::new(vec![1.0e10], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![1.0e10 + 5.0e-3], vec![1, 1]).unwrap();
        let result = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(1.0e-6),
                Value::from("DataScale"),
                Value::Num(1.0),
            ],
        )
        .unwrap();
        assert_eq!(result.mask_value(), Value::Bool(false));
    }

    #[test]
    fn typed_integer_inputs_are_compared_from_exact_storage() {
        let a = Tensor::new_integer(IntegerStorage::I16(vec![10, 20, 30]), vec![1, 3]).unwrap();
        let b = Tensor::new_integer(IntegerStorage::I16(vec![9, 31]), vec![1, 2]).unwrap();

        let result = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[Value::Num(0.2), Value::from("DataScale"), Value::Num(10.0)],
        )
        .unwrap();

        assert_eq!(result.mask.data, vec![1, 0, 1]);
    }

    #[test]
    fn wide_integers_enter_explicit_f64_tolerance_domain() {
        let a = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
            .unwrap();
        let b = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_992]), vec![1, 1])
            .unwrap();

        let result = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(1.0e-12),
                Value::from("DataScale"),
                Value::Num(1.0),
            ],
        )
        .unwrap();

        assert_eq!(result.mask_value(), Value::Bool(true));
    }

    #[test]
    fn typed_integer_tolerance_and_datascale_read_exact_storage() {
        let a = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![11.0, 24.0], vec![1, 2]).unwrap();
        let tolerance = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();
        let scale = Tensor::new_integer(IntegerStorage::U16(vec![1, 4]), vec![1, 2]).unwrap();

        let result = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Tensor(tolerance),
                Value::from("DataScale"),
                Value::Tensor(scale),
                Value::from("ByRows"),
                Value::Bool(true),
            ],
        )
        .unwrap();

        assert_eq!(result.mask.data, vec![1]);
    }

    #[test]
    fn loc_returns_first_matching_index() {
        let a = Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![2.01, 2.02, 4.2], vec![1, 3]).unwrap();
        let (_, loc) = eval(Value::Tensor(a), Value::Tensor(b), &[Value::Num(0.05)])
            .unwrap()
            .into_pair();
        match loc {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![0.0, 1.0, 3.0]),
            other => panic!("expected tensor loc, got {other:?}"),
        }
    }

    #[test]
    fn mask_only_request_does_not_materialize_locations() {
        let a = Tensor::new(vec![2.0, 9.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.99, 2.01, 9.5], vec![1, 3]).unwrap();
        let opts = parse_options(&[
            Value::Num(0.01),
            Value::from("OutputAllIndices"),
            Value::Bool(true),
        ])
        .unwrap();
        let result = block_on(evaluate_with_options(
            Value::Tensor(a),
            Value::Tensor(b),
            opts,
            LocRequest::None,
        ))
        .unwrap();
        assert_eq!(result.mask.data, vec![1, 0]);
        assert!(result.loc.is_none());
    }

    #[test]
    fn output_all_indices_returns_cell_array() {
        let a = Tensor::new(vec![2.0, 9.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.99, 2.01, 9.5], vec![1, 3]).unwrap();
        let (_, loc) = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(0.01),
                Value::from("OutputAllIndices"),
                Value::Bool(true),
            ],
        )
        .unwrap()
        .into_pair();
        let Value::Cell(cell) = loc else {
            panic!("expected cell loc");
        };
        assert_eq!(cell.shape, vec![1, 2]);
        match &cell.data[0] {
            Value::Tensor(indices) => assert_eq!(indices.materialize_f64(), vec![1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
        match &cell.data[1] {
            Value::Tensor(indices) => assert_eq!(indices.materialize_f64(), vec![0.0]),
            other => panic!("expected tensor zero indices, got {other:?}"),
        }
    }

    #[test]
    fn output_all_indices_matrix_cells_follow_public_cell_indexing() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![10.0, 2.0, 3.0, 20.0], vec![1, 4]).unwrap();
        let (_, loc) = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(1.0e-12),
                Value::from("OutputAllIndices"),
                Value::Bool(true),
            ],
        )
        .unwrap()
        .into_pair();
        let Value::Cell(cell) = loc else {
            panic!("expected cell loc");
        };
        assert_eq!(cell.shape, vec![2, 2]);
        let top_right = cell.get(0, 1).unwrap();
        match top_right {
            Value::Tensor(indices) => assert_eq!(indices.materialize_f64(), vec![2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
        let bottom_left = cell.get(1, 0).unwrap();
        match bottom_left {
            Value::Tensor(indices) => assert_eq!(indices.materialize_f64(), vec![3.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn by_rows_uses_column_scales_and_inf_ignores_column() {
        let a = Tensor::new(vec![0.0, 10.0, 0.5, 20.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.05, 10.2, 999.0, -999.0], vec![2, 2]).unwrap();
        let result = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(0.1),
                Value::from("ByRows"),
                Value::Bool(true),
                Value::from("DataScale"),
                Value::Tensor(Tensor::new(vec![1.0, f64::INFINITY], vec![1, 2]).unwrap()),
            ],
        )
        .unwrap();
        assert_eq!(result.mask.data, vec![1, 0]);
        let (_, loc) = result.into_pair();
        match loc {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 0.0]);
            }
            other => panic!("expected tensor loc, got {other:?}"),
        }
    }

    #[test]
    fn by_rows_typed_integer_inputs_are_compared_from_exact_storage() {
        let a = Tensor::new_integer(IntegerStorage::U16(vec![10, 20, 30, 40]), vec![2, 2]).unwrap();
        let b = Tensor::new_integer(IntegerStorage::U16(vec![11, 21, 31, 41]), vec![2, 2]).unwrap();

        let result = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(0.2),
                Value::from("ByRows"),
                Value::Bool(true),
                Value::from("DataScale"),
                Value::Num(10.0),
            ],
        )
        .unwrap();

        assert_eq!(result.mask.data, vec![1, 1]);
    }

    #[test]
    fn by_rows_output_all_indices_returns_row_cells() {
        let a = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![0.01, -0.01, 5.0, 1.0, 1.0, 5.0], vec![3, 2]).unwrap();
        let (_, loc) = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[
                Value::Num(0.02),
                Value::from("ByRows"),
                Value::Bool(true),
                Value::from("OutputAllIndices"),
                Value::Bool(true),
            ],
        )
        .unwrap()
        .into_pair();
        let Value::Cell(cell) = loc else {
            panic!("expected cell loc");
        };
        assert_eq!(cell.shape, vec![1, 1]);
        match &cell.data[0] {
            Value::Tensor(indices) => assert_eq!(indices.materialize_f64(), vec![1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_options_and_shape_mismatch() {
        let err = parse_options(&[Value::Num(-1.0)]).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:ismembertol:InvalidArgument")
        );

        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = eval(
            Value::Tensor(a),
            Value::Tensor(b),
            &[Value::from("ByRows"), Value::Bool(true)],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:ismembertol:RowsColumnMismatch")
        );
    }

    #[test]
    fn gpu_inputs_gather_to_host() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![1.0 + 1e-13, 3.0], vec![2, 1]).unwrap();
            let handle_a = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &a.materialize_f64(),
                    shape: &a.shape,
                })
                .expect("upload a");
            let handle_b = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &b.materialize_f64(),
                    shape: &b.shape,
                })
                .expect("upload b");
            let result = eval(Value::GpuTensor(handle_a), Value::GpuTensor(handle_b), &[]).unwrap();
            assert_eq!(result.mask.data, vec![1, 0]);
        });
    }
}
