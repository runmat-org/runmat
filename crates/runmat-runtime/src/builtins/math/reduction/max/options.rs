use super::*;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::RuntimeError;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub(super) enum ParsedCall {
    Reduction(ReductionArgs),
    Elementwise(ElementwiseArgs),
}

#[derive(Debug, Clone)]
pub(super) struct ReductionArgs {
    pub(super) selection: DimSelection,
    pub(super) nan_mode: ReductionNaN,
    pub(super) comparison: ComparisonMethod,
    pub(super) linear_index: bool,
}

impl Default for ReductionArgs {
    fn default() -> Self {
        Self {
            selection: DimSelection::Auto,
            nan_mode: ReductionNaN::Include,
            comparison: ComparisonMethod::Auto,
            linear_index: false,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Debug, Clone)]
pub(super) struct ElementwiseArgs {
    pub(super) other: Value,
    pub(super) nan_mode: ReductionNaN,
    pub(super) comparison: ComparisonMethod,
}

pub(super) async fn parse_call(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedCall> {
    if rest.is_empty() {
        return Ok(ParsedCall::Reduction(ReductionArgs::default()));
    }

    let first = &rest[0];
    if !is_empty_placeholder(first) {
        let (nan_mode, comparison) = parse_elementwise_options(rest.range(1..)?)?;
        return Ok(ParsedCall::Elementwise(ElementwiseArgs {
            other: first.clone(),
            nan_mode,
            comparison,
        }));
    }

    let mut args = ReductionArgs::default();
    parse_reduction_options(&mut args, rest.range(1..)?).await?;
    Ok(ParsedCall::Reduction(args))
}

pub(super) fn debug_value_kind(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "Num",
        Value::Int(_) => "Int",
        Value::Bool(_) => "Bool",
        Value::Tensor(t) => {
            if tensor::tensor_element_len(t) == 0 {
                "Tensor(empty)"
            } else {
                "Tensor"
            }
        }
        Value::GpuTensor(_) => "GpuTensor",
        Value::String(_) => "String",
        Value::CharArray(_) => "CharArray",
        Value::StringArray(sa) => {
            if sa.data().is_empty() {
                "StringArray(empty)"
            } else {
                "StringArray"
            }
        }
        Value::LogicalArray(l) => {
            if l.data.is_empty() {
                "LogicalArray(empty)"
            } else {
                "LogicalArray"
            }
        }
        Value::Cell(c) => {
            if c.data().is_empty() {
                "Cell(empty)"
            } else {
                "Cell"
            }
        }
        _ => "Other",
    }
}

pub(super) fn is_empty_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor_len(t) == 0,
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data().is_empty(),
        Value::CharArray(ca) => ca.data().is_empty(),
        Value::Cell(cell) => cell.data().is_empty(),
        Value::String(runmat_value::StringElement::Text(s)) => s.is_empty(),
        _ => false,
    }
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor::tensor_element_len(tensor)
}

async fn parse_reduction_options(
    args: &mut ReductionArgs,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut idx = 0usize;
    let mut selection_set = !matches!(args.selection, DimSelection::Auto);
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx])? {
            match MaxControl::from_public_option(&keyword) {
                Some(MaxControl::Missing(mode)) => {
                    args.nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(control @ (MaxControl::All | MaxControl::Linear)) => {
                    if selection_set {
                        return Err(max_invalid_argument(format!(
                            "max: '{keyword}' cannot be combined with an explicit dimension"
                        )));
                    }
                    args.selection = DimSelection::All;
                    args.linear_index = matches!(control, MaxControl::Linear);
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                Some(MaxControl::ComparisonMethod) => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(max_invalid_argument(
                            "max: expected a value after 'ComparisonMethod'",
                        ));
                    };
                    args.comparison = parse_comparison_method(value)?;
                    idx += 2;
                    continue;
                }
                None => {}
            }
        }
        if !selection_set {
            if let Some(selection) = parse_dimension_value(&rest[idx]).await? {
                args.selection = selection;
                selection_set = true;
                idx += 1;
                continue;
            }
        }
        return Err(max_invalid_argument(format!(
            "max: unrecognised argument {:?}",
            rest[idx]
        )));
    }
    Ok(())
}

fn parse_elementwise_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(ReductionNaN, ComparisonMethod)> {
    let mut nan_mode = ReductionNaN::Include;
    let mut comparison = ComparisonMethod::Auto;
    let mut idx = 0usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx])? {
            match MaxControl::from_public_option(&keyword) {
                Some(MaxControl::Missing(mode)) => {
                    nan_mode = mode;
                    idx += 1;
                    continue;
                }
                Some(MaxControl::ComparisonMethod) => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(max_invalid_argument(
                            "max: expected a value after 'ComparisonMethod'",
                        ));
                    };
                    comparison = parse_comparison_method(value)?;
                    idx += 2;
                    continue;
                }
                Some(MaxControl::All | MaxControl::Linear) => {
                    return Err(max_invalid_argument(format!(
                        "max: '{keyword}' is only supported for reduction calls"
                    )));
                }
                None => {}
            }
        }
        return Err(max_invalid_argument(format!(
            "max: unrecognised argument {:?}",
            rest[idx]
        )));
    }
    Ok((nan_mode, comparison))
}

fn parse_comparison_method(value: &Value) -> BuiltinResult<ComparisonMethod> {
    let Some(keyword) = keyword_of(value)? else {
        return Err(max_invalid_argument(
            "max: 'ComparisonMethod' expects a string value",
        ));
    };
    ComparisonMethod::from_public_option(&keyword).ok_or_else(|| {
        max_invalid_argument(format!("max: unsupported ComparisonMethod '{keyword}'"))
    })
}

async fn parse_dimension_value(value: &Value) -> BuiltinResult<Option<DimSelection>> {
    match value {
        Value::Int(_) | Value::Num(_) => tensor::scalar_dimension_from_value_typed(value, false)
            .map_err(map_scalar_dim_error)
            .map(|dim| dim.map(DimSelection::Dim)),
        Value::Tensor(t) => parse_dimension_tensor(value, t.shape()).await,
        Value::LogicalArray(logical) => parse_dimension_tensor(value, &logical.shape).await,
        Value::GpuTensor(_) => Err(max_invalid_argument(
            "max: dimension arguments must reside on the host",
        )),
        _ => Ok(None),
    }
}

async fn parse_dimension_tensor(
    value: &Value,
    shape: &[usize],
) -> BuiltinResult<Option<DimSelection>> {
    if tensor::element_count(shape) == 0 {
        return Ok(Some(DimSelection::Auto));
    }
    let is_vector = shape.len() == 1
        || shape.get(0).copied().unwrap_or(1) == 1
        || shape.get(1).copied().unwrap_or(1) == 1;
    if !is_vector {
        return Err(max_invalid_argument(
            "max: dimension vector must be a row or column vector",
        ));
    }
    let dims = tensor::dims_from_value_async_typed(value)
        .await
        .map_err(map_vector_dim_error)?;
    let Some(dims) = dims else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Ok(Some(DimSelection::Auto));
    }
    let mut seen = BTreeSet::new();
    let mut uniq = Vec::with_capacity(dims.len());
    for dim in dims {
        if dim < 1 {
            return Err(max_invalid_argument("max: dimension indices must be >= 1"));
        }
        if seen.insert(dim) {
            uniq.push(dim);
        }
    }
    Ok(Some(DimSelection::Vec(uniq)))
}

fn map_scalar_dim_error(error: tensor::DimensionDecodeError) -> RuntimeError {
    use tensor::DimensionDecodeError;
    let message = match &error {
        DimensionDecodeError::Fractional => "max: dimension must be integral".to_string(),
        DimensionDecodeError::NonFinite => "max: dimension must be finite".to_string(),
        DimensionDecodeError::Negative | DimensionDecodeError::BelowMinimum => {
            "max: dimension must be >= 1".to_string()
        }
        DimensionDecodeError::OutOfPlatformRange
        | DimensionDecodeError::IntegerNegative
        | DimensionDecodeError::IntegerOutOfPlatformRange => {
            "max: dimension is outside the supported range".to_string()
        }
        DimensionDecodeError::Gather(_) => error.to_string(),
    };
    error.attach_cause(max_invalid_argument(message))
}

fn map_vector_dim_error(error: tensor::DimensionDecodeError) -> RuntimeError {
    use tensor::DimensionDecodeError;
    let message = match &error {
        DimensionDecodeError::Negative
        | DimensionDecodeError::IntegerNegative
        | DimensionDecodeError::IntegerOutOfPlatformRange
        | DimensionDecodeError::BelowMinimum => "max: dimension indices must be >= 1".to_string(),
        DimensionDecodeError::NonFinite => "max: dimension entries must be finite".to_string(),
        DimensionDecodeError::Fractional => "max: dimension entries must be integers".to_string(),
        DimensionDecodeError::OutOfPlatformRange | DimensionDecodeError::Gather(_) => {
            error.to_string()
        }
    };
    error.attach_cause(max_invalid_argument(message))
}
