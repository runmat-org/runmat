//! MATLAB-compatible `findobj` builtin over RunMat graphics handles.

use regex::Regex;
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    FINDOBJ_DESCRIPTOR, FINDOBJ_INTEGER_CAPABILITIES,
};
use runmat_builtins::{
    FINDOBJ_ERROR_INVALID_ARGUMENT, FINDOBJ_ERROR_PROPERTY_ACCESS, FINDOBJ_INTEGER_ROOT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use super::properties::{get_properties_async, map_figure_error, resolve_plot_handle, PlotHandle};
use super::state::{self, FigureHandle, PlotObjectKind};
use super::style::{parse_color_value, LineStyleParseOptions};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::handles::numeric_handle_from_integer;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "findobj";

#[runtime_builtin(
    name = "findobj",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::findobj"
)]
pub async fn findobj_builtin_async(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let request = FindRequest::parse(args)?;
    let mut matches = Vec::new();
    for root in request.roots {
        collect_matches(root, request.max_depth, &request.expr, &mut matches).await?;
    }
    Ok(handle_array(matches))
}

#[cfg(test)]
pub fn findobj_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    futures::executor::block_on(findobj_builtin_async(args))
}

#[derive(Clone)]
struct FindRequest {
    roots: Vec<f64>,
    max_depth: Option<usize>,
    expr: Expr,
}

impl FindRequest {
    fn parse(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
        if args.first().is_some_and(is_host_integer_handle_alias) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FINDOBJ_INTEGER_ROOT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        let (roots, rest) = if let Some((roots, rest)) = split_root_handles(args.view())? {
            (roots, rest)
        } else {
            (vec![0.0], args.view())
        };

        let (max_depth, rest) = parse_depth(rest)?;
        let expr = if rest.is_empty() {
            Expr::True
        } else {
            let (expr, consumed) = parse_expr(rest)?;
            if consumed != rest.len() {
                return Err(invalid_argument("unable to parse trailing predicates"));
            }
            expr
        };

        Ok(Self {
            roots,
            max_depth,
            expr,
        })
    }
}

fn split_root_handles(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Option<(Vec<f64>, crate::call::FunctionArgumentView<'_>)>> {
    let Some(first) = args.first() else {
        return Ok(None);
    };
    match first {
        Value::Num(value) => Ok(Some((vec![*value], args.range(1..).expect("root tail")))),
        Value::Int(value) => Ok(Some((
            vec![exact_numeric_root(value)?],
            args.range(1..).expect("root tail"),
        ))),
        Value::Tensor(tensor) => {
            let roots = if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .iter()
                    .map(exact_numeric_root)
                    .collect::<crate::BuiltinResult<Vec<_>>>()?
            } else {
                tensor_utils::tensor_values_f64(tensor)
            };
            Ok(Some((roots, args.range(1..).expect("root tail"))))
        }
        _ => Ok(None),
    }
}

fn exact_numeric_root(value: &runmat_value::IntValue) -> crate::BuiltinResult<f64> {
    numeric_handle_from_integer(value).ok_or_else(|| {
        invalid_argument("integer root handle must be nonnegative and exactly representable")
    })
}

fn is_host_integer_handle_alias(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn parse_depth(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<(Option<usize>, crate::call::FunctionArgumentView<'_>)> {
    let mut max_depth = None;
    let mut rest = args;
    loop {
        let Some(option) = rest.first().map(text_scalar).transpose()?.flatten() else {
            break;
        };
        if option.eq_ignore_ascii_case("flat") {
            max_depth = Some(0);
            rest = rest.range(1..).expect("flat tail");
            continue;
        }
        if option.eq_ignore_ascii_case("-depth") {
            let Some(depth_value) = rest.get(1) else {
                return Err(invalid_argument("-depth requires a nonnegative integer"));
            };
            max_depth = Some(depth_from_value(depth_value)?);
            rest = rest.range(2..).expect("depth tail");
            continue;
        }
        break;
    }
    Ok((max_depth, rest))
}

fn depth_from_value(value: &Value) -> crate::BuiltinResult<usize> {
    if let Some(integer) = tensor_utils::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .ok_or_else(|| invalid_argument("-depth must be a nonnegative integer or Inf"));
    }
    let number = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor_utils::tensor_value_f64(tensor, 0)
        }
        _ => return Err(invalid_argument("-depth must be numeric")),
    };
    if number.is_infinite() && number.is_sign_positive() {
        return Ok(usize::MAX);
    }
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(invalid_argument(
            "-depth must be a nonnegative integer or Inf",
        ));
    }
    Ok(number as usize)
}

#[derive(Clone)]
enum Expr {
    True,
    HasProperty(String),
    Equals(String, Value),
    Regex(String, Vec<Regex>),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn matches<'a>(
        &'a self,
        handle: f64,
    ) -> futures::future::LocalBoxFuture<'a, crate::BuiltinResult<bool>> {
        Box::pin(async move {
            Ok(match self {
                Expr::True => true,
                Expr::HasProperty(prop) => property_value_async(handle, prop).await?.is_some(),
                Expr::Equals(prop, expected) => match property_value_async(handle, prop).await? {
                    Some(actual) => values_match(prop, &actual, expected)?,
                    None => false,
                },
                Expr::Regex(prop, regexes) => match property_value_async(handle, prop).await? {
                    Some(actual) => value_matches_regex(&actual, regexes)?,
                    None => false,
                },
                Expr::Not(expr) => !expr.matches(handle).await?,
                Expr::And(left, right) => {
                    left.matches(handle).await? && right.matches(handle).await?
                }
                Expr::Or(left, right) => {
                    left.matches(handle).await? || right.matches(handle).await?
                }
                Expr::Xor(left, right) => {
                    left.matches(handle).await? ^ right.matches(handle).await?
                }
            })
        })
    }
}

fn parse_expr(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<(Expr, usize)> {
    let (mut expr, mut consumed) = parse_factor(args)?;
    while consumed < args.len() {
        let (op, op_len) = match args.get(consumed).map(text_scalar).transpose()?.flatten() {
            Some(text) if text.eq_ignore_ascii_case("-and") => (LogicalOp::And, 1),
            Some(text) if text.eq_ignore_ascii_case("-or") => (LogicalOp::Or, 1),
            Some(text) if text.eq_ignore_ascii_case("-xor") => (LogicalOp::Xor, 1),
            Some(text)
                if text.eq_ignore_ascii_case("flat") || text.eq_ignore_ascii_case("-depth") =>
            {
                return Err(invalid_argument(
                    "flat and -depth must appear before predicates",
                ))
            }
            _ => (LogicalOp::And, 0),
        };
        let (rhs, rhs_len) = parse_factor(
            args.range(consumed + op_len..)
                .expect("bounded predicate tail"),
        )?;
        expr = match op {
            LogicalOp::And => Expr::And(Box::new(expr), Box::new(rhs)),
            LogicalOp::Or => Expr::Or(Box::new(expr), Box::new(rhs)),
            LogicalOp::Xor => Expr::Xor(Box::new(expr), Box::new(rhs)),
        };
        consumed += op_len + rhs_len;
    }
    Ok((expr, consumed))
}

fn parse_factor(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<(Expr, usize)> {
    let Some(first) = args.first() else {
        return Err(invalid_argument("expected a predicate"));
    };
    if let Value::Cell(cell) = first {
        let (expr, consumed) =
            parse_expr(crate::call::FunctionArgumentView::from_slice(cell.data()))?;
        if consumed != cell.data().len() {
            return Err(invalid_argument("unable to parse grouped predicate"));
        }
        return Ok((expr, 1));
    }

    let Some(text) = text_scalar(first)? else {
        return Err(invalid_argument("expected a property name or operator"));
    };

    if text.eq_ignore_ascii_case("-not") {
        let (expr, consumed) = parse_factor(args.range(1..).expect("not predicate tail"))?;
        return Ok((Expr::Not(Box::new(expr)), consumed + 1));
    }
    if text.eq_ignore_ascii_case("-property") {
        let Some(prop) = args.get(1).map(text_scalar).transpose()?.flatten() else {
            return Err(invalid_argument("-property requires a property name"));
        };
        return Ok((Expr::HasProperty(prop), 2));
    }
    if text.eq_ignore_ascii_case("-regexp") {
        let Some(prop) = args.get(1).map(text_scalar).transpose()?.flatten() else {
            return Err(invalid_argument("-regexp requires a property name"));
        };
        let Some(pattern_value) = args.get(2) else {
            return Err(invalid_argument("-regexp requires a pattern"));
        };
        return Ok((Expr::Regex(prop, regexes_from_value(pattern_value)?), 3));
    }
    if matches!(
        text.to_ascii_lowercase().as_str(),
        "-and" | "-or" | "-xor" | "flat" | "-depth"
    ) {
        return Err(invalid_argument(format!("unexpected operator `{text}`")));
    }

    let Some(value) = args.get(1) else {
        return Err(invalid_argument("property predicates require a value"));
    };
    Ok((Expr::Equals(text, value.clone()), 2))
}

#[derive(Clone, Copy)]
enum LogicalOp {
    And,
    Or,
    Xor,
}

fn regexes_from_value(value: &Value) -> crate::BuiltinResult<Vec<Regex>> {
    let patterns =
        match value {
            Value::Cell(cell) => cell
                .data()
                .iter()
                .map(|value| {
                    text_scalar(value)?.ok_or_else(|| {
                        invalid_argument("-regexp cell patterns must contain text values")
                    })
                })
                .collect::<crate::BuiltinResult<Vec<_>>>()?,
            Value::StringArray(strings) => strings
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid_argument("-regexp pattern must be text"))
                })
                .collect::<crate::BuiltinResult<Vec<_>>>()?,
            _ => vec![text_scalar(value)?
                .ok_or_else(|| invalid_argument("-regexp pattern must be text"))?],
        };
    if patterns.is_empty() {
        return Err(invalid_argument("-regexp requires at least one pattern"));
    }
    patterns
        .into_iter()
        .map(|pattern| Regex::new(&pattern).map_err(|err| invalid_argument(err.to_string())))
        .collect()
}

async fn collect_matches(
    root: f64,
    max_depth: Option<usize>,
    expr: &Expr,
    out: &mut Vec<f64>,
) -> crate::BuiltinResult<()> {
    let mut stack = vec![(root, 0usize)];
    while let Some((handle, depth)) = stack.pop() {
        if expr.matches(handle).await? {
            out.push(handle);
        }
        if max_depth.is_some_and(|max| depth >= max) {
            continue;
        }
        for child in child_handles(handle)?.into_iter().rev() {
            stack.push((child, depth.saturating_add(1)));
        }
    }
    Ok(())
}

fn child_handles(handle: f64) -> crate::BuiltinResult<Vec<f64>> {
    let resolved = resolve_plot_handle(&Value::Num(handle), BUILTIN_NAME)
        .map_err(|err| invalid_argument(err.message))?;
    match resolved {
        PlotHandle::Root => Ok(state::root_figure_handles()
            .into_iter()
            .map(|figure| figure.as_u32() as f64)
            .collect()),
        PlotHandle::Figure(figure) => figure_children(figure),
        PlotHandle::Axes(figure, axes_index) => axes_children(figure, axes_index),
        PlotHandle::Ruler(_, _, _)
        | PlotHandle::Text(_, _, _)
        | PlotHandle::Legend(_, _)
        | PlotHandle::PlotChild(_, _) => Ok(Vec::new()),
    }
}

fn figure_children(figure: FigureHandle) -> crate::BuiltinResult<Vec<f64>> {
    let mut children = state::axes_handles_for_figure(figure)
        .map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
    if state::figure_has_sg_title(figure) {
        children.push(state::encode_plot_object_handle(
            figure,
            0,
            PlotObjectKind::SuperTitle,
        ));
    }
    Ok(children)
}

fn axes_children(figure: FigureHandle, axes_index: usize) -> crate::BuiltinResult<Vec<f64>> {
    let meta = state::axes_metadata_snapshot(figure, axes_index)
        .map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
    let mut children = state::plot_child_handles_for_axes(figure, axes_index);
    if meta.title.is_some() {
        children.push(state::encode_plot_object_handle(
            figure,
            axes_index,
            PlotObjectKind::Title,
        ));
    }
    if meta.subtitle.is_some() {
        children.push(state::encode_plot_object_handle(
            figure,
            axes_index,
            PlotObjectKind::Subtitle,
        ));
    }
    if meta.x_label.is_some() {
        children.push(state::encode_plot_object_handle(
            figure,
            axes_index,
            PlotObjectKind::XLabel,
        ));
    }
    if meta.y_label.is_some() {
        children.push(state::encode_plot_object_handle(
            figure,
            axes_index,
            PlotObjectKind::YLabel,
        ));
    }
    if meta.z_label.is_some() {
        children.push(state::encode_plot_object_handle(
            figure,
            axes_index,
            PlotObjectKind::ZLabel,
        ));
    }
    children.push(state::encode_plot_object_handle(
        figure,
        axes_index,
        PlotObjectKind::XAxis,
    ));
    children.push(state::encode_plot_object_handle(
        figure,
        axes_index,
        PlotObjectKind::YAxis,
    ));
    if meta.legend_enabled
        && !state::legend_entries_snapshot(figure, axes_index)
            .map_err(|err| map_figure_error(BUILTIN_NAME, err))?
            .is_empty()
    {
        children.push(state::encode_plot_object_handle(
            figure,
            axes_index,
            PlotObjectKind::Legend,
        ));
    }
    Ok(children)
}

async fn property_value_async(handle: f64, property: &str) -> crate::BuiltinResult<Option<Value>> {
    let Ok(resolved) = resolve_plot_handle(&Value::Num(handle), BUILTIN_NAME) else {
        return Ok(None);
    };
    match get_properties_async(resolved, Some(property), BUILTIN_NAME).await {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.source.is_none() => Ok(None),
        Err(error) => {
            let span = error.span;
            let retry = error.gpu_gather_retry();
            let mut builder = build_runtime_error(format!(
                "{}: {}",
                FINDOBJ_ERROR_PROPERTY_ACCESS.message, error.message
            ))
            .with_builtin(BUILTIN_NAME)
            .with_gpu_gather_retry(retry)
            .with_source(error);
            if let Some(identifier) = FINDOBJ_ERROR_PROPERTY_ACCESS.identifier {
                builder = builder.with_identifier(identifier);
            }
            if let Some(span) = span {
                builder = builder.with_span(span);
            }
            Err(builder.build())
        }
    }
}

#[cfg(test)]
fn property_value(handle: f64, property: &str) -> Option<Value> {
    futures::executor::block_on(property_value_async(handle, property))
        .expect("property lookup should not fail operationally")
}

fn values_match(property: &str, actual: &Value, expected: &Value) -> crate::BuiltinResult<bool> {
    Ok({
        let prop = property.trim();
        if prop.eq_ignore_ascii_case("Color") || prop.to_ascii_lowercase().ends_with("color") {
            if let (Some(left), Some(right)) =
                (color_from_value(actual)?, color_from_value(expected)?)
            {
                return Ok((left - right).abs().max_element() <= 1e-6);
            }
        }

        if prop.eq_ignore_ascii_case("Type") {
            if let (Some(left), Some(right)) = (single_text(actual)?, single_text(expected)?) {
                return Ok(left.eq_ignore_ascii_case(&right));
            }
        }

        if let Some(expected_bool) = on_off_bool(expected) {
            if let Some(actual_bool) = on_off_bool(actual) {
                return Ok(actual_bool == expected_bool);
            }
        }

        match (actual, expected) {
            (Value::Num(left), Value::Num(right)) => numbers_equal(*left, *right),
            (Value::Int(left), Value::Int(right)) => left == right,
            (Value::Num(_), Value::Int(_)) | (Value::Int(_), Value::Num(_)) => false,
            (Value::Bool(left), Value::Bool(right)) => left == right,
            (Value::String(left), Value::String(right)) => {
                !left.is_missing() && !right.is_missing() && left == right
            }
            (Value::CharArray(left), Value::CharArray(right)) => left.data() == right.data(),
            (Value::String(left), Value::CharArray(right)) => left.as_str().is_some_and(|left| {
                left.encode_utf16()
                    .eq(right.data().iter().map(|unit| unit.raw()))
            }),
            (Value::CharArray(left), Value::String(right)) => {
                left.data().iter().map(|unit| unit.raw()).eq(right
                    .as_str()
                    .map(str::encode_utf16)
                    .into_iter()
                    .flatten())
                    && !right.is_missing()
            }
            (Value::Tensor(left), Value::Tensor(right)) => {
                left.shape() == right.shape() && tensor_values_match(left, right)
            }
            (Value::StringArray(left), Value::StringArray(right)) => {
                left.shape == right.shape && left.data() == right.data()
            }
            _ => false,
        }
    })
}

fn tensor_values_match(left: &Tensor, right: &Tensor) -> bool {
    let dtype = left.numeric_dtype();
    let right_dtype = right.numeric_dtype();
    if matches!(
        dtype,
        NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64
    ) || matches!(
        right_dtype,
        NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64
    ) {
        if dtype != right_dtype {
            return false;
        }
        return (0..left.len())
            .all(|index| left.numeric_value_at(index) == right.numeric_value_at(index));
    }
    left.materialize_f64()
        .iter()
        .zip(right.materialize_f64())
        .all(|(left, right)| numbers_equal(*left, right))
}

fn numbers_equal(left: f64, right: f64) -> bool {
    if left.is_nan() || right.is_nan() {
        left.is_nan() && right.is_nan()
    } else {
        (left - right).abs() <= 1e-12
    }
}

fn color_from_value(value: &Value) -> crate::BuiltinResult<Option<glam::Vec4>> {
    let text = single_text(value)?;
    Ok(
        parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)
            .ok()
            .or_else(|| text.and_then(|text| color_from_bracketed_text(&text))),
    )
}

fn color_from_bracketed_text(text: &str) -> Option<glam::Vec4> {
    let trimmed = text.trim();
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    let parts = inner
        .split([',', ' '])
        .filter(|part| !part.is_empty())
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if parts.len() != 3 {
        return None;
    }
    Some(glam::Vec4::new(parts[0], parts[1], parts[2], 1.0))
}

fn on_off_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            match array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "on" | "true" => Some(true),
                "off" | "false" => Some(false),
                _ => None,
            }
        }
        _ => {
            let eq =
                |key| crate::builtins::plotting::style::value_text_equals_ascii(value, key, true);
            if eq("on") || eq("true") {
                Some(true)
            } else if eq("off") || eq("false") {
                Some(false)
            } else {
                None
            }
        }
    }
}

fn value_matches_regex(value: &Value, regexes: &[Regex]) -> crate::BuiltinResult<bool> {
    let matches = |text: &str| regexes.iter().any(|regex| regex.is_match(text));
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => matches(text.as_str()),
        Value::CharArray(chars) => matches(
            &runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = invalid_argument(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        Value::StringArray(strings) => strings
            .data()
            .iter()
            .any(|text| text.as_str().is_some_and(|text| matches(text))),
        Value::Cell(cell) => {
            for child in cell.data() {
                if value_matches_regex(child, regexes)? {
                    return Ok(true);
                }
            }
            false
        }
        _ => false,
    })
}

fn single_text(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::CharArray(chars) => Some(
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = invalid_argument(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        Value::StringArray(strings)
            if strings.data().len() == 1 && !strings.data()[0].is_missing() =>
        {
            strings
                .data()
                .first()
                .and_then(|text| text.as_str().map(str::to_owned))
        }
        _ => None,
    })
}

fn text_scalar(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(single_text(value)?.map(|text| text.trim().to_string()))
}

fn handle_array(handles: Vec<f64>) -> Value {
    if handles.is_empty() {
        return Value::Tensor(
            Tensor::new_with_dtype(Vec::new(), vec![0, 0], NumericDType::F64)
                .expect("empty handle tensor shape is valid"),
        );
    }
    let len = handles.len();
    Value::Tensor(Tensor::new(handles, vec![len, 1]).expect("findobj handle column"))
}

fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        FINDOBJ_ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = FINDOBJ_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run, reset_plot_state};
    use futures::executor::block_on;
    use runmat_value::{CellArray, IntegerStorage};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn handles(value: Value) -> Vec<f64> {
        tensor(value).materialize_f64()
    }

    fn row(data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![1, data.len()]).expect("tensor"))
    }

    #[test]
    fn findobj_depth_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U8(vec![3]), vec![1, 1]).unwrap();

        assert_eq!(depth_from_value(&Value::Tensor(tensor)).unwrap(), 3);
    }

    #[test]
    fn findobj_depth_reads_all_integer_storage_classes_with_poisoned_f64_mirrors() {
        let storages = [
            IntegerStorage::I8(vec![3]),
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I32(vec![3]),
            IntegerStorage::I64(vec![3]),
            IntegerStorage::U8(vec![3]),
            IntegerStorage::U16(vec![3]),
            IntegerStorage::U32(vec![3]),
            IntegerStorage::U64(vec![3]),
        ];

        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("depth");
            assert_eq!(depth_from_value(&Value::Tensor(tensor)).expect("depth"), 3);
        }
    }

    #[test]
    fn findobj_integer_capabilities_cover_roots_depth_properties_and_residency() {
        assert_eq!(FINDOBJ_INTEGER_CAPABILITIES.len(), 4);
        assert_eq!(
            FINDOBJ_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            FINDOBJ_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            FINDOBJ_INTEGER_CAPABILITIES[2].computation_domain,
            BuiltinIntegerComputationDomain::ExactInteger
        );
        assert_eq!(
            FINDOBJ_INTEGER_CAPABILITIES[3].backend,
            BuiltinIntegerBackendRule::GpuRestricted
        );
    }

    #[test]
    fn findobj_integer_root_aliases_are_exact_and_compatibility_gated() {
        for storage in [
            IntegerStorage::I8(vec![3]),
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I32(vec![3]),
            IntegerStorage::I64(vec![3]),
            IntegerStorage::U8(vec![3]),
            IntegerStorage::U16(vec![3]),
            IntegerStorage::U32(vec![3]),
            IntegerStorage::U64(vec![3]),
        ] {
            let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer root");
            let values = [Value::Tensor(tensor)];
            let (roots, _) =
                split_root_handles(crate::call::FunctionArgumentView::from_slice(&values))
                    .expect("root parse")
                    .expect("root list");
            assert_eq!(roots, vec![3.0]);
        }
        assert!(exact_numeric_root(&runmat_value::IntValue::I64(-1)).is_err());
        assert!(exact_numeric_root(&runmat_value::IntValue::U64((1_u64 << 53) + 1)).is_err());

        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = findobj_builtin(vec![Value::Int(runmat_value::IntValue::U8(1))])
            .expect_err("integer root aliases must be gated");
        assert_eq!(
            error.identifier(),
            FINDOBJ_INTEGER_ROOT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn findobj_root_handles_reads_typed_integer_storage_exactly() {
        let roots = Tensor::new_integer(IntegerStorage::U16(vec![11, 13]), vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(roots), Value::String("Type".into())];
        let (handles, rest) =
            split_root_handles(crate::call::FunctionArgumentView::from_slice(&args))
                .expect("split roots")
                .expect("root handle list");

        assert_eq!(handles, vec![11.0, 13.0]);
        assert_eq!(rest.len(), 1);
    }

    #[test]
    fn findobj_tensor_property_equality_preserves_wide_integer_exactness() {
        let left = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_992]), vec![1, 1])
                .unwrap(),
        );
        let right = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap(),
        );

        assert!(!values_match("UserData", &left, &right).unwrap());
        assert!(values_match("UserData", &left, &left).unwrap());
        let raw = Value::CharArray(
            runmat_value::CharArray::from_code_units(
                vec![runmat_value::Utf16CodeUnit::from_raw(0xd800)],
                1,
                1,
            )
            .unwrap(),
        );
        assert!(values_match("UserData", &raw, &raw).unwrap());
        assert!(value_matches_regex(&raw, &[Regex::new(".*").unwrap()]).is_err());

        let other_class = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I64(vec![9_007_199_254_740_992]), vec![1, 1])
                .unwrap(),
        );
        assert!(!values_match("UserData", &left, &other_class).unwrap());
        assert!(!values_match(
            "UserData",
            &Value::Num(9_007_199_254_740_992.0),
            &Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_992)),
        )
        .unwrap());
    }

    #[test]
    fn findobj_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FINDOBJ_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = findobj()"));
        assert!(labels.contains(&"h = findobj(prop, value, ...)"));
        assert!(labels.contains(&"h = findobj(objhandles, '-depth', d, filters...)"));
    }

    #[test]
    fn findobj_returns_root_figure_axes_and_plot_children() {
        let _guard = setup();
        let line = block_on(plot_builtin(vec![row(&[1.0, 2.0, 3.0])])).unwrap();
        let figure = state::current_figure_handle().as_u32() as f64;

        let all = handles(findobj_builtin(Vec::new()).unwrap());
        assert!(all.contains(&0.0));
        assert!(all.contains(&figure));
        assert!(all.contains(&line));
        assert!(all.iter().any(|handle| {
            matches!(
                resolve_plot_handle(&Value::Num(*handle), "test"),
                Ok(PlotHandle::Axes(_, _))
            )
        }));
    }

    #[test]
    fn findobj_filters_by_type_property_and_explicit_root() {
        let _guard = setup();
        let line = block_on(plot_builtin(vec![
            row(&[1.0, 2.0]),
            Value::String("DisplayName".into()),
            Value::String("series-a".into()),
            Value::String("Color".into()),
            Value::String("red".into()),
        ]))
        .unwrap();
        let axes = state::current_axes_state();
        let ax = state::encode_axes_handle(axes.handle, axes.active_index);

        let by_type = handles(
            findobj_builtin(vec![
                Value::String("Type".into()),
                Value::String("line".into()),
            ])
            .unwrap(),
        );
        assert_eq!(by_type, vec![line]);

        let by_name = handles(
            findobj_builtin(vec![
                Value::Num(ax),
                Value::String("DisplayName".into()),
                Value::String("series-a".into()),
            ])
            .unwrap(),
        );
        assert_eq!(by_name, vec![line]);

        let by_rgb_color = handles(
            findobj_builtin(vec![
                Value::String("Color".into()),
                Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).expect("rgb tensor")),
            ])
            .unwrap(),
        );
        assert_eq!(by_rgb_color, vec![line]);

        let by_full_color_name = handles(
            findobj_builtin(vec![
                Value::String("Color".into()),
                Value::String("red".into()),
            ])
            .unwrap(),
        );
        assert_eq!(by_full_color_name, vec![line]);

        let rulers = handles(
            findobj_builtin(vec![
                Value::Num(ax),
                Value::String("Type".into()),
                Value::String("numericruler".into()),
            ])
            .unwrap(),
        );
        assert_eq!(
            rulers,
            vec![
                state::encode_plot_object_handle(
                    axes.handle,
                    axes.active_index,
                    PlotObjectKind::XAxis
                ),
                state::encode_plot_object_handle(
                    axes.handle,
                    axes.active_index,
                    PlotObjectKind::YAxis
                ),
            ]
        );
    }

    #[test]
    fn findobj_supports_depth_flat_property_not_and_regexp() {
        let _guard = setup();
        let line = block_on(plot_builtin(vec![row(&[1.0, 2.0])])).unwrap();
        let figure = state::current_figure_handle().as_u32() as f64;
        set_builtin(vec![
            Value::Num(line),
            Value::String("DisplayName".into()),
            Value::String("linear".into()),
        ])
        .unwrap();

        let flat = handles(
            findobj_builtin(vec![Value::Num(figure), Value::String("flat".into())]).unwrap(),
        );
        assert_eq!(flat, vec![figure]);

        let shallow = handles(
            findobj_builtin(vec![
                Value::Num(figure),
                Value::String("-depth".into()),
                Value::Num(1.0),
            ])
            .unwrap(),
        );
        assert!(shallow.contains(&figure));
        assert!(!shallow.contains(&line));

        let has_display_name = handles(
            findobj_builtin(vec![
                Value::String("-property".into()),
                Value::String("DisplayName".into()),
            ])
            .unwrap(),
        );
        assert_eq!(has_display_name, vec![line]);

        let not_axes = handles(
            findobj_builtin(vec![
                Value::String("-not".into()),
                Value::String("Type".into()),
                Value::String("axes".into()),
            ])
            .unwrap(),
        );
        assert!(not_axes.contains(&line));
        assert!(not_axes.contains(&figure));
        assert!(!not_axes.iter().any(|handle| {
            property_value(*handle, "Type")
                .and_then(|value| single_text(&value).unwrap())
                .is_some_and(|text| text == "axes")
        }));

        let regexp = handles(
            findobj_builtin(vec![
                Value::String("-regexp".into()),
                Value::String("DisplayName".into()),
                Value::String("lin.*".into()),
            ])
            .unwrap(),
        );
        assert_eq!(regexp, vec![line]);
    }

    #[test]
    fn findobj_supports_grouped_logical_expressions() {
        let _guard = setup();
        let line = block_on(plot_builtin(vec![
            row(&[1.0, 2.0]),
            Value::String("DisplayName".into()),
            Value::String("match".into()),
        ]))
        .unwrap();

        let group = Value::Cell(
            CellArray::new(
                vec![
                    Value::String("Type".into()),
                    Value::String("line".into()),
                    Value::String("-and".into()),
                    Value::String("DisplayName".into()),
                    Value::String("match".into()),
                ],
                1,
                5,
            )
            .expect("cell"),
        );
        let result = handles(findobj_builtin(vec![group]).unwrap());
        assert_eq!(result, vec![line]);
    }

    #[test]
    fn findobj_rejects_malformed_arguments() {
        let _guard = setup();
        let err = findobj_builtin(vec![Value::String("-depth".into())])
            .expect_err("missing depth should fail");
        assert_eq!(err.identifier(), FINDOBJ_ERROR_INVALID_ARGUMENT.identifier);

        let err = findobj_builtin(vec![
            Value::String("-regexp".into()),
            Value::String("Tag".into()),
            Value::String("[".into()),
        ])
        .expect_err("bad regex should fail");
        assert_eq!(err.identifier(), FINDOBJ_ERROR_INVALID_ARGUMENT.identifier);
    }
}
