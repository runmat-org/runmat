//! MATLAB-compatible `findobj` builtin over RunMat graphics handles.

use regex::Regex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{get_properties, map_figure_error, resolve_plot_handle, PlotHandle};
use super::state::{self, FigureHandle, PlotObjectKind};
use super::style::{parse_color_value, LineStyleParseOptions};
use crate::builtins::plotting::type_resolvers::handle_array_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "findobj";

const FINDOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics handles that satisfy the requested hierarchy and property filters.",
}];

const FINDOBJ_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const FINDOBJ_INPUTS_FILTERS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filters",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Property/value predicates, -property, -regexp, -not, and logical operators.",
}];

const FINDOBJ_INPUTS_ROOTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "objhandles",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle or handle array that restricts the search root.",
    },
    BuiltinParamDescriptor {
        name: "filters",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional depth/flat options and property predicates.",
    },
];

const FINDOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "h = findobj()",
        inputs: &FINDOBJ_INPUTS_NONE,
        outputs: &FINDOBJ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj(prop, value, ...)",
        inputs: &FINDOBJ_INPUTS_FILTERS,
        outputs: &FINDOBJ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj(objhandles, '-depth', d, filters...)",
        inputs: &FINDOBJ_INPUTS_ROOTS,
        outputs: &FINDOBJ_OUTPUT,
    },
];

const FINDOBJ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FINDOBJ.INVALID_ARGUMENT",
    identifier: Some("RunMat:findobj:InvalidArgument"),
    when: "Arguments, search roots, depth, operators, property names, or regexp patterns are malformed.",
    message: "findobj: invalid argument",
};

const FINDOBJ_ERRORS: [BuiltinErrorDescriptor; 1] = [FINDOBJ_ERROR_INVALID_ARGUMENT];

pub const FINDOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FINDOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FINDOBJ_ERRORS,
};

#[runtime_builtin(
    name = "findobj",
    category = "plotting",
    summary = "Find graphics objects with matching properties.",
    keywords = "findobj,graphics,handle,property,plotting",
    suppress_auto_output = true,
    type_resolver(handle_array_type),
    descriptor(crate::builtins::plotting::findobj::FINDOBJ_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::findobj"
)]
pub fn findobj_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let request = FindRequest::parse(args)?;
    let mut matches = Vec::new();
    for root in request.roots {
        collect_matches(root, 0, request.max_depth, &request.expr, &mut matches)?;
    }
    Ok(handle_array(matches))
}

#[derive(Clone)]
struct FindRequest {
    roots: Vec<f64>,
    max_depth: Option<usize>,
    expr: Expr,
}

impl FindRequest {
    fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let (roots, rest) = if let Some((roots, rest)) = split_root_handles(&args)? {
            (roots, rest)
        } else {
            (vec![0.0], args.as_slice())
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

fn split_root_handles(args: &[Value]) -> crate::BuiltinResult<Option<(Vec<f64>, &[Value])>> {
    let Some(first) = args.first() else {
        return Ok(None);
    };
    match first {
        Value::Num(value) => Ok(Some((vec![*value], &args[1..]))),
        Value::Int(value) => Ok(Some((vec![value.to_f64()], &args[1..]))),
        Value::Tensor(tensor) => Ok(Some((tensor.data.clone(), &args[1..]))),
        _ => Ok(None),
    }
}

fn parse_depth(args: &[Value]) -> crate::BuiltinResult<(Option<usize>, &[Value])> {
    let mut max_depth = None;
    let mut rest = args;
    loop {
        let Some(option) = rest.first().and_then(text_scalar) else {
            break;
        };
        if option.eq_ignore_ascii_case("flat") {
            max_depth = Some(0);
            rest = &rest[1..];
            continue;
        }
        if option.eq_ignore_ascii_case("-depth") {
            let Some(depth_value) = rest.get(1) else {
                return Err(invalid_argument("-depth requires a nonnegative integer"));
            };
            max_depth = Some(depth_from_value(depth_value)?);
            rest = &rest[2..];
            continue;
        }
        break;
    }
    Ok((max_depth, rest))
}

fn depth_from_value(value: &Value) -> crate::BuiltinResult<usize> {
    let number = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
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
    fn matches(&self, handle: f64) -> bool {
        match self {
            Expr::True => true,
            Expr::HasProperty(prop) => property_value(handle, prop).is_some(),
            Expr::Equals(prop, expected) => property_value(handle, prop)
                .is_some_and(|actual| values_match(prop, &actual, expected)),
            Expr::Regex(prop, regexes) => property_value(handle, prop)
                .map(value_texts)
                .is_some_and(|texts| {
                    texts
                        .iter()
                        .any(|text| regexes.iter().any(|regex| regex.is_match(text)))
                }),
            Expr::Not(expr) => !expr.matches(handle),
            Expr::And(left, right) => left.matches(handle) && right.matches(handle),
            Expr::Or(left, right) => left.matches(handle) || right.matches(handle),
            Expr::Xor(left, right) => left.matches(handle) ^ right.matches(handle),
        }
    }
}

fn parse_expr(args: &[Value]) -> crate::BuiltinResult<(Expr, usize)> {
    let (mut expr, mut consumed) = parse_factor(args)?;
    while consumed < args.len() {
        let (op, op_len) = match args.get(consumed).and_then(text_scalar) {
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
        let (rhs, rhs_len) = parse_factor(&args[consumed + op_len..])?;
        expr = match op {
            LogicalOp::And => Expr::And(Box::new(expr), Box::new(rhs)),
            LogicalOp::Or => Expr::Or(Box::new(expr), Box::new(rhs)),
            LogicalOp::Xor => Expr::Xor(Box::new(expr), Box::new(rhs)),
        };
        consumed += op_len + rhs_len;
    }
    Ok((expr, consumed))
}

fn parse_factor(args: &[Value]) -> crate::BuiltinResult<(Expr, usize)> {
    let Some(first) = args.first() else {
        return Err(invalid_argument("expected a predicate"));
    };
    if let Value::Cell(cell) = first {
        let (expr, consumed) = parse_expr(&cell.data)?;
        if consumed != cell.data.len() {
            return Err(invalid_argument("unable to parse grouped predicate"));
        }
        return Ok((expr, 1));
    }

    let Some(text) = text_scalar(first) else {
        return Err(invalid_argument("expected a property name or operator"));
    };

    if text.eq_ignore_ascii_case("-not") {
        let (expr, consumed) = parse_factor(&args[1..])?;
        return Ok((Expr::Not(Box::new(expr)), consumed + 1));
    }
    if text.eq_ignore_ascii_case("-property") {
        let Some(prop) = args.get(1).and_then(text_scalar) else {
            return Err(invalid_argument("-property requires a property name"));
        };
        return Ok((Expr::HasProperty(prop), 2));
    }
    if text.eq_ignore_ascii_case("-regexp") {
        let Some(prop) = args.get(1).and_then(text_scalar) else {
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
                .data
                .iter()
                .map(|value| {
                    text_scalar(value).ok_or_else(|| {
                        invalid_argument("-regexp cell patterns must contain text values")
                    })
                })
                .collect::<crate::BuiltinResult<Vec<_>>>()?,
            Value::StringArray(strings) => strings.data.clone(),
            _ => vec![text_scalar(value)
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

fn collect_matches(
    handle: f64,
    depth: usize,
    max_depth: Option<usize>,
    expr: &Expr,
    out: &mut Vec<f64>,
) -> crate::BuiltinResult<()> {
    if expr.matches(handle) {
        out.push(handle);
    }
    if max_depth.is_some_and(|max| depth >= max) {
        return Ok(());
    }
    for child in child_handles(handle)? {
        collect_matches(child, depth.saturating_add(1), max_depth, expr, out)?;
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

fn property_value(handle: f64, property: &str) -> Option<Value> {
    let resolved = resolve_plot_handle(&Value::Num(handle), BUILTIN_NAME).ok()?;
    get_properties(resolved, Some(property), BUILTIN_NAME).ok()
}

fn values_match(property: &str, actual: &Value, expected: &Value) -> bool {
    let prop = property.trim();
    if prop.eq_ignore_ascii_case("Color") || prop.to_ascii_lowercase().ends_with("color") {
        if let (Some(left), Some(right)) = (color_from_value(actual), color_from_value(expected)) {
            return (left - right).abs().max_element() <= 1e-6;
        }
    }

    if prop.eq_ignore_ascii_case("Type") {
        if let (Some(left), Some(right)) = (single_text(actual), single_text(expected)) {
            return left.eq_ignore_ascii_case(&right);
        }
    }

    if let Some(expected_bool) = on_off_bool(expected) {
        if let Some(actual_bool) = on_off_bool(actual) {
            return actual_bool == expected_bool;
        }
    }

    match (actual, expected) {
        (Value::Num(left), Value::Num(right)) => numbers_equal(*left, *right),
        (Value::Int(left), Value::Int(right)) => left == right,
        (Value::Num(left), Value::Int(right)) => numbers_equal(*left, right.to_f64()),
        (Value::Int(left), Value::Num(right)) => numbers_equal(left.to_f64(), *right),
        (Value::Bool(left), Value::Bool(right)) => left == right,
        (Value::String(left), Value::String(right)) => left == right,
        (Value::CharArray(left), Value::CharArray(right)) => left.data == right.data,
        (Value::String(left), Value::CharArray(right)) => {
            left == &right.data.iter().collect::<String>()
        }
        (Value::CharArray(left), Value::String(right)) => {
            &left.data.iter().collect::<String>() == right
        }
        (Value::Tensor(left), Value::Tensor(right)) => {
            left.shape == right.shape
                && left
                    .data
                    .iter()
                    .zip(&right.data)
                    .all(|(left, right)| numbers_equal(*left, *right))
        }
        (Value::StringArray(left), Value::StringArray(right)) => {
            left.shape == right.shape && left.data == right.data
        }
        _ => false,
    }
}

fn numbers_equal(left: f64, right: f64) -> bool {
    if left.is_nan() || right.is_nan() {
        left.is_nan() && right.is_nan()
    } else {
        (left - right).abs() <= 1e-12
    }
}

fn color_from_value(value: &Value) -> Option<glam::Vec4> {
    parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)
        .ok()
        .or_else(|| single_text(value).and_then(|text| color_from_bracketed_text(&text)))
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
        _ => single_text(value).and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" => Some(true),
            "off" | "false" => Some(false),
            _ => None,
        }),
    }
}

fn value_texts(value: Value) -> Vec<String> {
    match value {
        Value::String(text) => vec![text],
        Value::CharArray(chars) => vec![chars.data.into_iter().collect()],
        Value::StringArray(strings) => strings.data,
        Value::Cell(cell) => cell.data.into_iter().flat_map(value_texts).collect(),
        Value::Num(value) => vec![format!("{value}")],
        Value::Int(value) => vec![format!("{}", value.to_f64())],
        Value::Bool(value) => vec![if value { "on" } else { "off" }.to_string()],
        _ => Vec::new(),
    }
}

fn single_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        Value::StringArray(strings) if strings.data.len() == 1 => strings.data.first().cloned(),
        _ => None,
    }
}

fn text_scalar(value: &Value) -> Option<String> {
    single_text(value).map(|text| text.trim().to_string())
}

fn handle_array(handles: Vec<f64>) -> Value {
    if handles.is_empty() {
        return Value::Tensor(
            Tensor::new_with_dtype(Vec::new(), vec![0, 0], NumericDType::F64)
                .expect("empty handle tensor shape is valid"),
        );
    }
    Value::Tensor(Tensor {
        rows: handles.len(),
        cols: 1,
        shape: vec![handles.len(), 1],
        data: handles,
        dtype: NumericDType::F64,
    })
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
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use futures::executor::block_on;
    use runmat_builtins::CellArray;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
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
        tensor(value).data
    }

    fn row(data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![1, data.len()]).expect("tensor"))
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
                .and_then(|value| single_text(&value))
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
