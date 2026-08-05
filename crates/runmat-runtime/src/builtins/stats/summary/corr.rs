//! MATLAB-compatible `corr` builtin.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{standard_normal_cdf, student_t_cdf};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "corr";

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise correlation coefficients.",
}];

const OUTPUT_R_P: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pairwise correlation coefficients.",
    },
    BuiltinParamDescriptor {
        name: "PValue",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "P-values for testing the hypothesis of no correlation.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations by variables.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Second observations-by-variables input.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Type, Rows, Tail, and Weights.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const SIGNATURES_WITH_P: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "R = corr(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_R_P,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.corr.INVALID_ARGUMENT",
    identifier: Some("RunMat:corr:InvalidArgument"),
    when: "Inputs, row counts, Type, Rows, Tail, or Weights options are malformed or unsupported.",
    message: "corr: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.corr.INTERNAL",
    identifier: Some("RunMat:corr:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "corr: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

const CORR_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corr-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corr with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrIntegerDataExtension"),
};

const CORR_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corr-integer-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corr with typed-integer observation weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrIntegerWeightsExtension"),
};

const CORR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [CORR_INTEGER_DATA_EXTENSION, CORR_INTEGER_WEIGHTS_EXTENSION];

const CORR_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X_or_Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented observation domain is single or double; RunMat mode additionally accepts all eight real integer classes.",
    }];

const CORR_INTEGER_WEIGHTS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented observation-weight domain is single or double; RunMat mode additionally accepts exact nonnegative typed-integer column vectors.",
    }];

const CORR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[rho, pval] = corr(integer_X_or_Y, Type=type, Rows=rows, Weights=weights)",
        inputs: &CORR_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat's integer-data extension preserves exact same-class ordering for Spearman/Kendall and exact integer differences for Pearson centering before producing double rho/p-values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[rho, pval] = corr(X, Y, Weights=integer_weights)",
        inputs: &CORR_INTEGER_WEIGHTS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed integer weights are validated exactly, then enter the documented nonnegative weighted-correlation domain; weighted p-values are NaN.",
    },
];

pub const CORR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES_WITH_P,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn corr_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, None]),
    }
}

fn corr_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:corr:InvalidArgument")
        .build()
}

fn corr_internal(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:corr:Internal")
        .build()
}

#[derive(Clone, Copy)]
enum CorrType {
    Pearson,
    Spearman,
    Kendall,
}

#[derive(Clone, Copy)]
enum RowsMode {
    All,
    Complete,
    Pairwise,
}

#[derive(Clone, Copy)]
enum Tail {
    Both,
    Right,
    Left,
}

struct CorrArgs {
    left: Tensor,
    right: Option<Tensor>,
    corr_type: CorrType,
    rows: RowsMode,
    tail: Tail,
    weights: Option<Vec<f64>>,
}

#[runtime_builtin(
    name = "corr",
    category = "stats/summary",
    summary = "Compute pairwise linear or rank correlations between variables.",
    keywords = "corr,correlation,pearson,spearman,kendall,statistics,rows,tail,weights",
    type_resolver(corr_type),
    descriptor(crate::builtins::stats::summary::corr::CORR_DESCRIPTOR),
    extensions(CORR_EXTENSIONS),
    integer_capabilities(CORR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::corr"
)]
async fn corr_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_corr_integer_extensions(&value, &rest)?;
    let args = parse_args(value, rest).await?;
    corr_compute(args)
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn ensure_corr_integer_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    let second_is_data = rest
        .first()
        .is_some_and(|candidate| keyword_of(candidate).is_none());
    if is_typed_integer_value(value)
        || second_is_data && rest.first().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&CORR_INTEGER_DATA_EXTENSION, NAME)?;
    }
    if rest.windows(2).any(|pair| {
        keyword_of(&pair[0]).is_some_and(|name| name.eq_ignore_ascii_case("weights"))
            && is_typed_integer_value(&pair[1])
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CORR_INTEGER_WEIGHTS_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

async fn parse_args(value: Value, rest: Vec<Value>) -> BuiltinResult<CorrArgs> {
    let left = value_to_tensor(value).await?;
    let mut right = None;
    let mut corr_type = CorrType::Pearson;
    let mut rows = RowsMode::All;
    let mut tail = Tail::Both;
    let mut weights = None;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if keyword_of(first).is_none() {
            right = Some(value_to_tensor(first.clone()).await?);
            idx = 1;
        }
    }
    while idx < rest.len() {
        if idx + 1 >= rest.len() {
            return Err(corr_error(
                "corr: name-value options must be provided in pairs",
            ));
        }
        let name = keyword_of(&rest[idx])
            .ok_or_else(|| corr_error("corr: option names must be string scalars"))?;
        match name.to_ascii_lowercase().as_str() {
            "type" => {
                let option = keyword_of(&rest[idx + 1])
                    .ok_or_else(|| corr_error("corr: Type must be a string scalar"))?;
                match option.to_ascii_lowercase().as_str() {
                    "pearson" => corr_type = CorrType::Pearson,
                    "spearman" => corr_type = CorrType::Spearman,
                    "kendall" => corr_type = CorrType::Kendall,
                    other => return Err(corr_error(format!("corr: unsupported Type '{other}'"))),
                }
            }
            "rows" => {
                let option = keyword_of(&rest[idx + 1])
                    .ok_or_else(|| corr_error("corr: Rows must be a string scalar"))?;
                match option.to_ascii_lowercase().as_str() {
                    "all" => rows = RowsMode::All,
                    "complete" => rows = RowsMode::Complete,
                    "pairwise" => rows = RowsMode::Pairwise,
                    other => return Err(corr_error(format!("corr: unsupported Rows '{other}'"))),
                }
            }
            "tail" => {
                let option = keyword_of(&rest[idx + 1])
                    .ok_or_else(|| corr_error("corr: Tail must be a string scalar"))?;
                match option.to_ascii_lowercase().as_str() {
                    "both" => tail = Tail::Both,
                    "right" => tail = Tail::Right,
                    "left" => tail = Tail::Left,
                    other => return Err(corr_error(format!("corr: unsupported Tail '{other}'"))),
                }
            }
            "weights" => {
                if weights.is_some() {
                    return Err(corr_error("corr: Weights can be specified only once"));
                }
                weights =
                    Some(value_to_weights(rest[idx + 1].clone(), matrix_shape(&left).0).await?);
            }
            other => return Err(corr_error(format!("corr: unsupported option '{other}'"))),
        }
        idx += 2;
    }
    Ok(CorrArgs {
        left,
        right,
        corr_type,
        rows,
        tail,
        weights,
    })
}

async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| corr_error(format!("corr: {err}")))?;
    tensor::value_into_tensor_for(NAME, gathered).map_err(|err| corr_error(format!("corr: {err}")))
}

async fn value_to_weights(value: Value, expected_rows: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_tensor(value).await?;
    let (rows, cols) = matrix_shape(&tensor);
    if rows != expected_rows || cols != 1 {
        return Err(corr_error(format!(
            "corr: Weights must be an {expected_rows}-by-1 column vector"
        )));
    }
    let weights = tensor::tensor_into_values_f64(tensor);
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(corr_error(
            "corr: Weights must contain finite nonnegative values",
        ));
    }
    Ok(weights)
}

fn matrix_shape(tensor: &Tensor) -> (usize, usize) {
    let len = tensor.len();
    let shape = tensor::default_shape_for(&tensor.shape, len);
    match shape.as_slice() {
        [] => (1, 1),
        [_] => (len, 1),
        [rows, cols, ..] => (*rows, *cols),
    }
}

#[derive(Clone)]
enum CorrColumn {
    Floating(Vec<f64>),
    Integer(Vec<IntValue>),
}

impl CorrColumn {
    fn from_tensor(tensor: &Tensor, rows: usize, col: usize) -> BuiltinResult<Self> {
        let start = col * rows;
        let end = start + rows;
        if let Some(storage) = tensor.integer_storage() {
            let values = (start..end)
                .map(|index| {
                    storage.value_at(index).ok_or_else(|| {
                        corr_internal(format!("corr: integer column index {index} is invalid"))
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Self::Integer(values))
        } else {
            let values = tensor::tensor_values_f64_cow(tensor);
            Ok(Self::Floating(values[start..end].to_vec()))
        }
    }

    fn is_nan(&self, index: usize) -> bool {
        matches!(self, Self::Floating(values) if values[index].is_nan())
    }

    fn compare(&self, left: usize, right: usize) -> Ordering {
        match self {
            Self::Floating(values) => values[left]
                .partial_cmp(&values[right])
                .unwrap_or(Ordering::Equal),
            Self::Integer(values) => {
                exact_int_to_i128(&values[left]).cmp(&exact_int_to_i128(&values[right]))
            }
        }
    }

    fn centered_values(&self, indices: &[usize]) -> Vec<f64> {
        match self {
            Self::Floating(values) => indices.iter().map(|index| values[*index]).collect(),
            Self::Integer(values) => {
                let anchor = indices
                    .first()
                    .map(|index| exact_int_to_i128(&values[*index]))
                    .unwrap_or(0);
                indices
                    .iter()
                    .map(|index| (exact_int_to_i128(&values[*index]) - anchor) as f64)
                    .collect()
            }
        }
    }

    fn ranks(&self, indices: &[usize], weights: Option<&[f64]>) -> Vec<f64> {
        let mut order = (0..indices.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| self.compare(indices[*left], indices[*right]));
        let mut ranks = vec![0.0; indices.len()];
        let mut start = 0usize;
        let mut cumulative_weight = 0.0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len()
                && self.compare(indices[order[start]], indices[order[end]]) == Ordering::Equal
            {
                end += 1;
            }
            let rank = if let Some(weights) = weights {
                let group_weight = order[start..end]
                    .iter()
                    .map(|position| weights[indices[*position]])
                    .sum::<f64>();
                let group_count = (end - start) as f64;
                let mean_weight = group_weight / group_count;
                let rank = cumulative_weight + (group_count + 1.0) * mean_weight / 2.0;
                cumulative_weight += group_weight;
                rank
            } else {
                (start + 1 + end) as f64 / 2.0
            };
            for position in &order[start..end] {
                ranks[*position] = rank;
            }
            start = end;
        }
        ranks
    }
}

fn exact_int_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn corr_compute(args: CorrArgs) -> BuiltinResult<Value> {
    let (left_rows, left_cols) = matrix_shape(&args.left);
    let right = args.right.as_ref().unwrap_or(&args.left);
    let (right_rows, right_cols) = matrix_shape(right);
    if left_rows != right_rows {
        return Err(corr_error(
            "corr: X and Y must have the same number of observations",
        ));
    }
    let mut rho_data = Vec::with_capacity(left_cols * right_cols);
    let mut p_data = Vec::with_capacity(left_cols * right_cols);
    let complete_mask = match args.rows {
        RowsMode::Complete => Some(complete_rows(
            &args.left, right, left_rows, left_cols, right_cols,
        )),
        RowsMode::All | RowsMode::Pairwise => None,
    };
    let all_mask = vec![true; left_rows];
    for right_col in 0..right_cols {
        let y = CorrColumn::from_tensor(right, right_rows, right_col)?;
        for left_col in 0..left_cols {
            let x = CorrColumn::from_tensor(&args.left, left_rows, left_col)?;
            let pair = match complete_mask.as_ref() {
                Some(mask) => corr_pair_masked(
                    &x,
                    &y,
                    mask,
                    args.corr_type,
                    RowsMode::All,
                    args.weights.as_deref(),
                ),
                None => corr_pair_masked(
                    &x,
                    &y,
                    &all_mask,
                    args.corr_type,
                    args.rows,
                    args.weights.as_deref(),
                ),
            };
            rho_data.push(pair.rho);
            p_data.push(if args.weights.is_some() {
                f64::NAN
            } else {
                correlation_pvalue(pair.rho, pair.n, args.corr_type, args.tail)
            });
        }
    }
    let shape = vec![left_cols, right_cols];
    let rho = Tensor::new(rho_data, shape.clone())
        .map(tensor::tensor_into_value)
        .map_err(|err| corr_internal(format!("corr: {err}")))?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![rho]));
        }
        let p = Tensor::new(p_data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| corr_internal(format!("corr: {err}")))?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![rho, p],
        ));
    }
    Ok(rho)
}

fn complete_rows(
    left: &Tensor,
    right: &Tensor,
    rows: usize,
    left_cols: usize,
    right_cols: usize,
) -> Vec<bool> {
    let mut mask = vec![true; rows];
    for row in 0..rows {
        for col in 0..left_cols {
            if matches!(
                left.numeric_value_at(col * rows + row),
                Some(runmat_builtins::NumericScalar::F64(value)) if value.is_nan()
            ) || matches!(
                left.numeric_value_at(col * rows + row),
                Some(runmat_builtins::NumericScalar::F32(value)) if value.is_nan()
            ) {
                mask[row] = false;
            }
        }
        for col in 0..right_cols {
            if matches!(
                right.numeric_value_at(col * rows + row),
                Some(runmat_builtins::NumericScalar::F64(value)) if value.is_nan()
            ) || matches!(
                right.numeric_value_at(col * rows + row),
                Some(runmat_builtins::NumericScalar::F32(value)) if value.is_nan()
            ) {
                mask[row] = false;
            }
        }
    }
    mask
}

#[derive(Clone, Copy)]
struct CorrPair {
    rho: f64,
    n: usize,
}

fn corr_pair_masked(
    x: &CorrColumn,
    y: &CorrColumn,
    mask: &[bool],
    corr_type: CorrType,
    rows: RowsMode,
    weights: Option<&[f64]>,
) -> CorrPair {
    let mut indices = Vec::new();
    for idx in 0..mask.len() {
        if !mask[idx] {
            continue;
        }
        match rows {
            RowsMode::All => indices.push(idx),
            RowsMode::Complete | RowsMode::Pairwise => {
                if !x.is_nan(idx) && !y.is_nan(idx) {
                    indices.push(idx);
                }
            }
        }
    }
    if matches!(rows, RowsMode::All)
        && indices
            .iter()
            .any(|index| x.is_nan(*index) || y.is_nan(*index))
    {
        return CorrPair {
            rho: f64::NAN,
            n: indices.len(),
        };
    }
    if indices.len() < 2 {
        return CorrPair {
            rho: f64::NAN,
            n: indices.len(),
        };
    }
    let rho = match corr_type {
        CorrType::Pearson => pearson_columns(x, y, &indices, weights),
        CorrType::Spearman => {
            let xr = x.ranks(&indices, weights);
            let yr = y.ranks(&indices, weights);
            let selected_weights = weights.map(|weights| {
                indices
                    .iter()
                    .map(|index| weights[*index])
                    .collect::<Vec<_>>()
            });
            pearson(&xr, &yr, selected_weights.as_deref())
        }
        CorrType::Kendall => kendall_tau_b(x, y, &indices, weights),
    };
    CorrPair {
        rho,
        n: indices.len(),
    }
}

fn pearson_columns(
    x: &CorrColumn,
    y: &CorrColumn,
    indices: &[usize],
    weights: Option<&[f64]>,
) -> f64 {
    let xs = x.centered_values(indices);
    let ys = y.centered_values(indices);
    let selected_weights = weights.map(|weights| {
        indices
            .iter()
            .map(|index| weights[*index])
            .collect::<Vec<_>>()
    });
    pearson(&xs, &ys, selected_weights.as_deref())
}

fn pearson(x: &[f64], y: &[f64], weights: Option<&[f64]>) -> f64 {
    let weight_sum = weights
        .map(|weights| weights.iter().sum::<f64>())
        .unwrap_or(x.len() as f64);
    if weight_sum <= 0.0 {
        return f64::NAN;
    }
    let mean_x = x
        .iter()
        .enumerate()
        .map(|(index, value)| weights.map_or(*value, |weights| weights[index] * *value))
        .sum::<f64>()
        / weight_sum;
    let mean_y = y
        .iter()
        .enumerate()
        .map(|(index, value)| weights.map_or(*value, |weights| weights[index] * *value))
        .sum::<f64>()
        / weight_sum;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (index, (a, b)) in x.iter().zip(y.iter()).enumerate() {
        let weight = weights.map_or(1.0, |weights| weights[index]);
        let dx = *a - mean_x;
        let dy = *b - mean_y;
        sxx += weight * dx * dx;
        syy += weight * dy * dy;
        sxy += weight * dx * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        f64::NAN
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

fn kendall_tau_b(
    x: &CorrColumn,
    y: &CorrColumn,
    indices: &[usize],
    weights: Option<&[f64]>,
) -> f64 {
    if indices.len() < 2 {
        return f64::NAN;
    }
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut ties_x = 0.0;
    let mut ties_y = 0.0;
    for i in 0..indices.len() {
        for j in i + 1..indices.len() {
            let left = indices[i];
            let right = indices[j];
            let pair_weight = weights.map_or(1.0, |weights| weights[left] * weights[right]);
            let dx = x.compare(left, right);
            let dy = y.compare(left, right);
            if dx == Ordering::Equal && dy == Ordering::Equal {
                continue;
            }
            if dx == Ordering::Equal {
                ties_x += pair_weight;
            } else if dy == Ordering::Equal {
                ties_y += pair_weight;
            } else {
                if dx == dy {
                    concordant += pair_weight;
                } else {
                    discordant += pair_weight;
                }
            }
        }
    }
    let denominator =
        f64::sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y));
    if denominator == 0.0 {
        f64::NAN
    } else {
        (concordant - discordant) / denominator
    }
}

fn correlation_pvalue(rho: f64, n: usize, corr_type: CorrType, tail: Tail) -> f64 {
    if rho.is_nan() || n < 2 {
        return f64::NAN;
    }
    let cdf = match corr_type {
        CorrType::Pearson | CorrType::Spearman => {
            if n < 3 {
                return f64::NAN;
            }
            let df = (n - 2) as f64;
            let denom = 1.0 - rho * rho;
            let t = if denom <= 0.0 {
                if rho > 0.0 {
                    f64::INFINITY
                } else if rho < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    0.0
                }
            } else {
                rho * f64::sqrt(df / denom)
            };
            student_t_cdf(t, df)
        }
        CorrType::Kendall => {
            if n < 3 {
                return f64::NAN;
            }
            let n = n as f64;
            let variance = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0));
            standard_normal_cdf(rho / f64::sqrt(variance))
        }
    };
    tail_pvalue(cdf, tail)
}

fn tail_pvalue(cdf: f64, tail: Tail) -> f64 {
    if cdf.is_nan() {
        return f64::NAN;
    }
    match tail {
        Tail::Both => (2.0 * cdf.min(1.0 - cdf)).clamp(0.0, 1.0),
        Tail::Right => (1.0 - cdf).clamp(0.0, 1.0),
        Tail::Left => cdf.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    #[test]
    fn corr_matrix_defaults_to_columns() {
        let out = block_on(corr_builtin(
            tensor(vec![1.0, 2.0, 1.0, 4.0], vec![2, 2]),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!((tensor.materialize_f64()[1] - 1.0).abs() < 1e-12);
                assert!((tensor.materialize_f64()[2] - 1.0).abs() < 1e-12);
                assert!((tensor.materialize_f64()[3] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corr_accepts_typed_integer_matrix_inputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = block_on(corr_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 1, 4]), vec![2, 2], 0.0),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1.0e-12);
                assert!((tensor.materialize_f64()[3] - 1.0).abs() < 1.0e-12);
            }
            other => panic!("expected tensor correlation, got {other:?}"),
        }
    }

    #[test]
    fn corr_complete_rows_reads_typed_integer_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = poisoned_int_tensor(
            IntegerStorage::I16(vec![1, 2, 3, 2, 4, 6]),
            vec![3, 2],
            f64::NAN,
        );
        let y = poisoned_int_tensor(IntegerStorage::U16(vec![5, 10, 15]), vec![3, 1], f64::NAN);

        let out = block_on(corr_builtin(
            x,
            vec![y, Value::from("Rows"), Value::from("complete")],
        ))
        .unwrap();

        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1.0e-12);
                assert!((tensor.materialize_f64()[1] - 1.0).abs() < 1.0e-12);
            }
            other => panic!("expected tensor correlation, got {other:?}"),
        }
    }

    #[test]
    fn corr_xy_returns_cross_correlation_matrix() {
        let x = tensor(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]);
        let y = tensor(vec![2.0, 4.0, 6.0], vec![3, 1]);
        let out = block_on(corr_builtin(x, vec![y])).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!((tensor.materialize_f64()[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corr_pairwise_omits_nan_pair() {
        let x = tensor(vec![1.0, 2.0, f64::NAN, 4.0], vec![4, 1]);
        let y = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let out = block_on(corr_builtin(
            x,
            vec![y, Value::from("Rows"), Value::from("pairwise")],
        ))
        .unwrap();
        match out {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn corr_returns_pvalues_when_requested() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = tensor(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(corr_builtin(x, vec![y])).unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                assert!(matches!(&values[0], Value::Num(value) if (*value - 1.0).abs() < 1e-12));
                assert!(matches!(&values[1], Value::Num(value) if *value <= 1e-12));
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn corr_supports_kendall_and_tail_option() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = tensor(vec![1.0, 3.0, 2.0, 4.0], vec![4, 1]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(corr_builtin(
            x,
            vec![
                y,
                Value::from("Type"),
                Value::from("Kendall"),
                Value::from("Tail"),
                Value::from("right"),
            ],
        ))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                assert!(
                    matches!(&values[0], Value::Num(value) if (*value - (2.0 / 3.0)).abs() < 1e-12)
                );
                assert!(matches!(&values[1], Value::Num(value) if *value > 0.0 && *value < 0.2));
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn corr_integer_extensions_are_independently_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let data_error = block_on(corr_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1], 0.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            data_error.identifier(),
            Some("RunMat:compatibility:CorrIntegerDataExtension")
        );

        let weight_error = block_on(corr_builtin(
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            vec![
                Value::from("Weights"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1, 2, 3]), vec![3, 1], 0.0),
            ],
        ))
        .unwrap_err();
        assert_eq!(
            weight_error.identifier(),
            Some("RunMat:compatibility:CorrIntegerWeightsExtension")
        );
    }

    #[test]
    fn corr_resident_integer_data_rejects_before_gather() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                block_on(corr_builtin(Value::GpuTensor(handle.clone()), Vec::new())).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:CorrIntegerDataExtension")
            );
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn corr_supports_all_eight_integer_data_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![1, 2, 3]),
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I32(vec![1, 2, 3]),
            IntegerStorage::I64(vec![1, 2, 3]),
            IntegerStorage::U8(vec![1, 2, 3]),
            IntegerStorage::U16(vec![1, 2, 3]),
            IntegerStorage::U32(vec![1, 2, 3]),
            IntegerStorage::U64(vec![1, 2, 3]),
        ];
        for storage in storages {
            let class = storage.class_name();
            let weight_storage = storage.clone();
            let out = block_on(corr_builtin(
                poisoned_int_tensor(storage, vec![3, 1], 0.0),
                Vec::new(),
            ))
            .unwrap_or_else(|error| panic!("{class}: {error}"));
            assert!(matches!(out, Value::Num(value) if (value - 1.0).abs() < 1.0e-12));

            let weighted = block_on(corr_builtin(
                tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
                vec![
                    Value::from("Weights"),
                    poisoned_int_tensor(weight_storage, vec![3, 1], 0.0),
                ],
            ))
            .unwrap_or_else(|error| panic!("{class} weights: {error}"));
            assert!(matches!(weighted, Value::Num(value) if (value - 1.0).abs() < 1.0e-12));
        }
    }

    #[test]
    fn corr_wide_integer_rank_and_pearson_paths_remain_distinct() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let base = 1_u64 << 53;
        let x = poisoned_int_tensor(
            IntegerStorage::U64(vec![base, base + 1, base + 2]),
            vec![3, 1],
            0.0,
        );
        let y = poisoned_int_tensor(
            IntegerStorage::U64(vec![base + 10, base + 11, base + 12]),
            vec![3, 1],
            0.0,
        );
        for method in ["Pearson", "Spearman", "Kendall"] {
            let out = block_on(corr_builtin(
                x.clone(),
                vec![y.clone(), Value::from("Type"), Value::from(method)],
            ))
            .unwrap();
            assert!(
                matches!(out, Value::Num(value) if (value - 1.0).abs() < 1.0e-12),
                "{method}: {out:?}"
            );
        }
    }

    #[test]
    fn corr_weights_affect_all_types_and_force_nan_pvalues() {
        let x = tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let y = tensor(vec![1.0, 4.0, 2.0], vec![3, 1]);
        let weights = tensor(vec![1.0, 0.0, 1.0], vec![3, 1]);
        for method in ["Pearson", "Spearman", "Kendall"] {
            let _outputs = crate::output_count::push_output_count(Some(2));
            let out = block_on(corr_builtin(
                x.clone(),
                vec![
                    y.clone(),
                    Value::from("Type"),
                    Value::from(method),
                    Value::from("Weights"),
                    weights.clone(),
                ],
            ))
            .unwrap();
            let Value::OutputList(values) = out else {
                panic!("{method}: expected two outputs");
            };
            assert!(
                matches!(values[0], Value::Num(value) if (value - 1.0).abs() < 1.0e-12),
                "{method}: {:?}",
                values[0]
            );
            assert!(
                matches!(values[1], Value::Num(value) if value.is_nan()),
                "{method}: {:?}",
                values[1]
            );
        }
    }

    #[test]
    fn corr_weighted_spearman_uses_documented_weighted_midrank_formula() {
        let column = CorrColumn::Floating(vec![1.0, 2.0, 2.0]);
        let ranks = column.ranks(&[0, 1, 2], Some(&[1.0, 2.0, 4.0]));
        assert_eq!(ranks, vec![1.0, 5.5, 5.5]);
    }

    #[test]
    fn corr_kendall_excludes_joint_ties_from_tau_b_denominator() {
        let x = CorrColumn::Integer(vec![IntValue::U64(7), IntValue::U64(7)]);
        let y = CorrColumn::Integer(vec![IntValue::U64(9), IntValue::U64(9)]);
        assert!(kendall_tau_b(&x, &y, &[0, 1], None).is_nan());
    }
}
