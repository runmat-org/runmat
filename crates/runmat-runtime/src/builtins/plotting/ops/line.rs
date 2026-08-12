//! MATLAB-compatible low-level `line` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{Figure, Line3Plot, LinePlot};

use super::common::{numeric_pair, numeric_triplet};
use super::op_common::line_inputs::NumericInput;
use super::op_common::{apply_axes_target, split_leading_axes_handle, AxesTarget};
use super::plot::build_line_plot_for_builtin;
use super::plot3::build_line3_plot_for_builtin;
use super::plotting_error;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    append_active_plot, current_axes_state, current_figure_handle,
    line_color_for_target_axes_series_index, register_line3_handle, register_line_handle,
    PlotRenderOptions,
};
use super::style::{
    looks_like_option_name, parse_line_style_args, value_as_bool, value_as_string, LineAppearance,
    LineStyleParseOptions,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "line";
type Series2D = (Vec<f64>, Vec<f64>);
type Series3D = (Vec<f64>, Vec<f64>, Vec<f64>);

const LINE_OUTPUT_HANDLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Line graphics handle scalar or row vector when matrix data creates multiple lines.",
}];

const LINE_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
];

const LINE_INPUTS_X_Y_Z: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y data.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z data.",
    },
];

const LINE_INPUTS_PROPS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name/value pairs such as XData, YData, ZData, Color, LineWidth, and DisplayName.",
}];

const LINE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "h = line(X, Y)",
        inputs: &LINE_INPUTS_X_Y,
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(X, Y, Z)",
        inputs: &LINE_INPUTS_X_Y_Z,
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(X, Y, Name, Value, ...)",
        inputs: &LINE_INPUTS_PROPS,
        outputs: &LINE_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "h = line(Name, Value, ...)",
        inputs: &LINE_INPUTS_PROPS,
        outputs: &LINE_OUTPUT_HANDLES,
    },
];

const LINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:line:InvalidArgument"),
    when: "Line data, axes target, style arguments, or Name/Value pairs are invalid.",
    message: "line: invalid argument",
};

const LINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINE.INTERNAL",
    identifier: Some("RunMat:line:Internal"),
    when: "Internal plotting state update fails.",
    message: "line: internal operation failed",
};

const LINE_ERRORS: [BuiltinErrorDescriptor; 2] = [LINE_ERROR_INVALID_ARGUMENT, LINE_ERROR_INTERNAL];

pub const LINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINE_ERRORS,
};

#[runtime_builtin(
    name = "line",
    category = "plotting",
    summary = "Create low-level 2-D or 3-D line graphics objects.",
    keywords = "line,graphics,plotting,2d,3d",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::line::LINE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::line"
)]
pub async fn line_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (leading_axes, args) = split_leading_axes_handle(args, BUILTIN_NAME)?;
    let parsed = parse_line_args(leading_axes, args).await?;
    apply_axes_target(parsed.axes_target, BUILTIN_NAME)?;

    let figure_handle = current_figure_handle();
    let plot_indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices);
    let axes_index_slot = std::rc::Rc::new(std::cell::RefCell::new(None));
    let axes_index_out = std::rc::Rc::clone(&axes_index_slot);
    let mut objects = Some(parsed.objects);
    let render_result = append_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure, axes_index| {
            *axes_index_out.borrow_mut() = Some(axes_index);
            let objects = objects.take().expect("line objects consumed exactly once");
            plot_indices_slot
                .borrow_mut()
                .extend(append_line_objects_to_figure(figure, axes_index, objects));
            Ok(())
        },
    );

    let axes_index = axes_index_slot.borrow().unwrap_or(0);
    let handles = plot_indices
        .borrow()
        .iter()
        .map(|(plot_index, is_3d)| {
            if *is_3d {
                register_line3_handle(figure_handle, axes_index, *plot_index)
            } else {
                register_line_handle(figure_handle, axes_index, *plot_index)
            }
        })
        .collect::<Vec<_>>();

    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(plotting_error(BUILTIN_NAME, err.to_string()));
        }
    }
    Ok(handles_value(handles))
}

pub(super) struct ParsedLineCall {
    pub(super) axes_target: AxesTarget,
    pub(super) objects: Vec<LineObject>,
}

pub(super) enum LineObject {
    TwoD(LinePlot),
    ThreeD(Line3Plot),
}

pub(super) fn append_line_objects_to_figure(
    figure: &mut Figure,
    axes_index: usize,
    objects: Vec<LineObject>,
) -> Vec<(usize, bool)> {
    objects
        .into_iter()
        .map(|object| match object {
            LineObject::TwoD(plot) => (figure.add_line_plot_on_axes(plot, axes_index), false),
            LineObject::ThreeD(plot) => {
                figure.set_axes_z_limits(axes_index, None);
                (figure.add_line3_plot_on_axes(plot, axes_index), true)
            }
        })
        .collect()
}

struct LineOptions {
    x: Option<Value>,
    y: Option<Value>,
    z: Option<Value>,
    axes_target: AxesTarget,
    style_tokens: Vec<Value>,
    visible: Option<bool>,
}

pub(super) async fn parse_line_args(
    leading_axes: AxesTarget,
    args: Vec<Value>,
) -> BuiltinResult<ParsedLineCall> {
    let mut options = if args.is_empty() {
        LineOptions {
            x: Some(vector_value(&[0.0, 1.0])),
            y: Some(vector_value(&[0.0, 1.0])),
            z: None,
            axes_target: leading_axes,
            style_tokens: Vec::new(),
            visible: None,
        }
    } else if begins_with_name_value(&args) {
        parse_property_constructor(leading_axes, &args)?
    } else {
        parse_positional_constructor(leading_axes, &args)?
    };

    let parsed_style = parse_line_style_args(
        &options.style_tokens,
        &LineStyleParseOptions::generic(BUILTIN_NAME),
    )?;
    let color_explicit = parsed_style.color_explicit;
    let appearance = parsed_style.appearance;
    let label = parsed_style.label.unwrap_or_default();
    let color_target = options.axes_target.unwrap_or_else(|| {
        let axes = current_axes_state();
        (axes.handle, axes.active_index)
    });
    let x = options
        .x
        .take()
        .unwrap_or_else(|| vector_value(&[0.0, 1.0]));
    let y = options
        .y
        .take()
        .unwrap_or_else(|| vector_value(&[0.0, 1.0]));

    let objects = if let Some(z) = options.z.take() {
        build_3d_objects(
            x,
            y,
            z,
            &label,
            &appearance,
            color_explicit,
            color_target,
            options.visible,
        )
        .await?
    } else {
        build_2d_objects(
            x,
            y,
            &label,
            &appearance,
            color_explicit,
            color_target,
            options.visible,
        )
        .await?
    };
    Ok(ParsedLineCall {
        axes_target: options.axes_target,
        objects,
    })
}

fn parse_positional_constructor(
    mut axes_target: AxesTarget,
    args: &[Value],
) -> BuiltinResult<LineOptions> {
    let mut idx = 0usize;
    let x = args
        .get(idx)
        .cloned()
        .ok_or_else(|| line_err("expected X data"))?;
    if !is_numeric_value(&x) {
        return Err(line_err("expected numeric X data"));
    }
    idx += 1;
    let y = args
        .get(idx)
        .cloned()
        .ok_or_else(|| line_err("expected Y data"))?;
    if !is_numeric_value(&y) {
        return Err(line_err("expected numeric Y data"));
    }
    idx += 1;

    let z = if matches!(args.get(idx), Some(value) if is_numeric_value(value)) {
        let value = args[idx].clone();
        idx += 1;
        Some(value)
    } else {
        None
    };

    let mut style_tokens = Vec::new();
    let mut visible = None;
    while idx < args.len() {
        let token = args[idx].clone();
        if !matches!(token, Value::String(_) | Value::CharArray(_)) {
            return Err(line_err("style and option names must be strings"));
        }
        idx += 1;
        let token_text =
            value_as_string(&token).ok_or_else(|| line_err("style tokens must be strings"))?;
        let lower = token_text.trim().to_ascii_lowercase();
        if lower == "parent" {
            if idx >= args.len() {
                return Err(line_err("name-value arguments must come in pairs"));
            }
            let PlotHandle::Axes(handle, axes_index) =
                resolve_plot_handle(&args[idx], BUILTIN_NAME)?
            else {
                return Err(line_err("Parent must be an axes handle"));
            };
            axes_target = Some((handle, axes_index));
            idx += 1;
            continue;
        }
        if lower == "visible" {
            if idx >= args.len() {
                return Err(line_err("name-value arguments must come in pairs"));
            }
            visible = Some(
                value_as_bool(&args[idx])
                    .ok_or_else(|| line_err("Visible must be logical or 'on'/'off'"))?,
            );
            idx += 1;
            continue;
        }
        style_tokens.push(token);
        if looks_like_option_name(&lower) {
            if idx >= args.len() {
                return Err(line_err("name-value arguments must come in pairs"));
            }
            style_tokens.push(args[idx].clone());
            idx += 1;
        }
    }

    Ok(LineOptions {
        x: Some(x),
        y: Some(y),
        z,
        axes_target,
        style_tokens,
        visible,
    })
}

fn parse_property_constructor(
    axes_target: AxesTarget,
    args: &[Value],
) -> BuiltinResult<LineOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(line_err("name-value arguments must come in pairs"));
    }
    let mut options = LineOptions {
        x: None,
        y: None,
        z: None,
        axes_target,
        style_tokens: Vec::new(),
        visible: None,
    };
    for pair in args.chunks_exact(2) {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| line_err("option names must be char arrays or strings"))?;
        match key.trim().to_ascii_lowercase().as_str() {
            "xdata" => options.x = Some(pair[1].clone()),
            "ydata" => options.y = Some(pair[1].clone()),
            "zdata" => options.z = Some(pair[1].clone()),
            "parent" => {
                let PlotHandle::Axes(handle, axes_index) =
                    resolve_plot_handle(&pair[1], BUILTIN_NAME)?
                else {
                    return Err(line_err("Parent must be an axes handle"));
                };
                options.axes_target = Some((handle, axes_index));
            }
            "visible" => {
                options.visible = Some(
                    value_as_bool(&pair[1])
                        .ok_or_else(|| line_err("Visible must be logical or 'on'/'off'"))?,
                );
            }
            _ => {
                options.style_tokens.push(pair[0].clone());
                options.style_tokens.push(pair[1].clone());
            }
        }
    }
    Ok(options)
}

async fn build_2d_objects(
    x: Value,
    y: Value,
    label: &str,
    appearance: &LineAppearance,
    color_explicit: bool,
    color_target: (super::state::FigureHandle, usize),
    visible: Option<bool>,
) -> BuiltinResult<Vec<LineObject>> {
    let x = NumericInput::from_value(x, BUILTIN_NAME)?;
    let y = NumericInput::from_value(y, BUILTIN_NAME)?;
    let mut base_appearance = appearance.clone();
    if !color_explicit {
        base_appearance.color =
            line_color_for_target_axes_series_index(color_target.0, color_target.1, 0);
    }

    if let (Some(x_gpu), Some(y_gpu)) = (x.gpu_handle(), y.gpu_handle()) {
        if x.len() == y.len()
            && x.len() >= 2
            && is_gpu_vector_shape(&x_gpu.shape)
            && is_gpu_vector_shape(&y_gpu.shape)
        {
            if let Ok(mut plot) = super::plot::build_line_gpu_plot_async_for_builtin(
                BUILTIN_NAME,
                x_gpu,
                y_gpu,
                label,
                &base_appearance,
            )
            .await
            {
                if label.is_empty() {
                    plot.label = None;
                }
                if let Some(visible) = visible {
                    plot.set_visible(visible);
                }
                return Ok(vec![LineObject::TwoD(plot)]);
            }
        }
    }

    let x_tensor = x.into_tensor_async(BUILTIN_NAME).await?;
    let y_tensor = y.into_tensor_async(BUILTIN_NAME).await?;
    let series = expand_xy_series(&x_tensor, &y_tensor)?;
    series
        .into_iter()
        .enumerate()
        .map(|(idx, (x, y))| {
            let mut appearance = base_appearance.clone();
            if !color_explicit {
                appearance.color =
                    line_color_for_target_axes_series_index(color_target.0, color_target.1, idx);
            }
            let mut plot = build_line_plot_for_builtin(BUILTIN_NAME, x, y, label, &appearance)?;
            if label.is_empty() {
                plot.label = None;
            }
            if let Some(visible) = visible {
                plot.set_visible(visible);
            }
            Ok(LineObject::TwoD(plot))
        })
        .collect()
}

async fn build_3d_objects(
    x: Value,
    y: Value,
    z: Value,
    label: &str,
    appearance: &LineAppearance,
    color_explicit: bool,
    color_target: (super::state::FigureHandle, usize),
    visible: Option<bool>,
) -> BuiltinResult<Vec<LineObject>> {
    let x = NumericInput::from_value(x, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    let y = NumericInput::from_value(y, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    let z = NumericInput::from_value(z, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    let series = expand_xyz_series(&x, &y, &z)?;
    series
        .into_iter()
        .enumerate()
        .map(|(idx, (x, y, z))| {
            let mut appearance = appearance.clone();
            if !color_explicit {
                appearance.color =
                    line_color_for_target_axes_series_index(color_target.0, color_target.1, idx);
            }
            let mut plot = build_line3_plot_for_builtin(BUILTIN_NAME, x, y, z, label, &appearance)?;
            if label.is_empty() {
                plot.label = None;
            }
            if let Some(visible) = visible {
                plot.set_visible(visible);
            }
            Ok(LineObject::ThreeD(plot))
        })
        .collect()
}

fn expand_xy_series(x: &Tensor, y: &Tensor) -> BuiltinResult<Vec<Series2D>> {
    if is_vector_tensor(x) && is_vector_tensor(y) {
        let (x, y) = numeric_pair(x.clone(), y.clone(), BUILTIN_NAME)?;
        return Ok(vec![(x, y)]);
    }
    if same_matrix_shape(x, y) {
        return Ok((0..x.cols())
            .map(|col| (column(x, col), column(y, col)))
            .collect());
    }
    if is_vector_tensor(x) && vector_len(x) == y.rows() {
        return Ok((0..y.cols())
            .map(|col| (tensor_utils::tensor_values_f64(x), column(y, col)))
            .collect());
    }
    if is_vector_tensor(y) && vector_len(y) == x.rows() {
        return Ok((0..x.cols())
            .map(|col| (column(x, col), tensor_utils::tensor_values_f64(y)))
            .collect());
    }
    Err(line_err("X and Y inputs must be vectors with equal length, matrices with the same size, or a vector paired with each matrix column"))
}

fn expand_xyz_series(x: &Tensor, y: &Tensor, z: &Tensor) -> BuiltinResult<Vec<Series3D>> {
    if is_vector_tensor(x) && is_vector_tensor(y) && is_vector_tensor(z) {
        let (x, y, z) = numeric_triplet(x.clone(), y.clone(), z.clone(), BUILTIN_NAME)?;
        return Ok(vec![(x, y, z)]);
    }
    if same_matrix_shape(x, y) && same_matrix_shape(x, z) {
        return Ok((0..x.cols())
            .map(|col| (column(x, col), column(y, col), column(z, col)))
            .collect());
    }
    Err(line_err(
        "X, Y, and Z inputs must be vectors with equal length or matrices with the same size",
    ))
}

fn column(tensor: &Tensor, col: usize) -> Vec<f64> {
    (0..tensor.rows())
        .map(|row| tensor_utils::tensor_value_f64(tensor, row + col * tensor.rows()))
        .collect()
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    tensor.rows() == 1 || tensor.cols() == 1
}

fn vector_len(tensor: &Tensor) -> usize {
    tensor_utils::tensor_element_len(tensor)
}

fn same_matrix_shape(a: &Tensor, b: &Tensor) -> bool {
    a.rows() == b.rows() && a.cols() == b.cols() && !is_vector_tensor(a) && !is_vector_tensor(b)
}

pub(super) fn handles_value(handles: Vec<f64>) -> Value {
    if handles.len() == 1 {
        Value::Num(handles[0])
    } else {
        let len = handles.len();
        Value::Tensor(Tensor::new(handles, vec![1, len]).expect("line handle row vector"))
    }
}

fn vector_value(data: &[f64]) -> Value {
    Value::Tensor(
        Tensor::new(data.to_vec(), vec![1, data.len()]).expect("line property row vector"),
    )
}

fn begins_with_name_value(args: &[Value]) -> bool {
    let Some(first) = args.first().and_then(value_as_string) else {
        return false;
    };
    looks_like_option_name(&first.trim().to_ascii_lowercase())
        || matches!(
            first.trim().to_ascii_lowercase().as_str(),
            "xdata" | "ydata" | "zdata" | "parent"
        )
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn is_gpu_vector_shape(shape: &[usize]) -> bool {
    shape.len() <= 1 || shape.iter().filter(|&&dim| dim > 1).count() <= 1
}

fn line_err(message: impl Into<String>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, format!("line: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clone_figure, configure_subplot, current_figure_handle};
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;
    use runmat_plot::plots::PlotElement;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        guard
    }

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).expect("line test tensor"))
    }

    fn cleared_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Tensor {
        Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor")
    }

    #[test]
    fn descriptor_lists_common_forms() {
        let labels = LINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"h = line(X, Y)"));
        assert!(labels.contains(&"h = line(X, Y, Z)"));
        assert!(labels.contains(&"h = line(Name, Value, ...)"));
    }

    #[test]
    fn expand_xy_series_reads_typed_integer_storage_without_mirror() {
        let x = cleared_int_tensor(IntegerStorage::I16(vec![1, 2]), 1, 2);
        let y = cleared_int_tensor(IntegerStorage::I16(vec![10, 20, 30, 40]), 2, 2);

        let series = expand_xy_series(&x, &y).expect("series");

        assert_eq!(
            series,
            vec![
                (vec![1.0, 2.0], vec![10.0, 20.0]),
                (vec![1.0, 2.0], vec![30.0, 40.0])
            ]
        );
    }

    #[test]
    fn line_appends_without_requiring_hold_on() {
        let _guard = setup();
        let _ = block_on(plot_builtin(vec![tensor(&[1.0, 2.0], 1, 2)])).unwrap();
        let h = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0], 1, 2),
            tensor(&[3.0, 4.0], 1, 2),
            Value::String("Color".into()),
            Value::String("r".into()),
        ]))
        .unwrap();
        assert!(matches!(h, Value::Num(_)));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 2);
    }

    #[test]
    fn line_property_constructor_creates_styled_handle() {
        let _guard = setup();
        let h = block_on(line_builtin(vec![
            Value::String("XData".into()),
            tensor(&[1.0, 2.0, 3.0], 1, 3),
            Value::String("YData".into()),
            tensor(&[4.0, 5.0, 6.0], 1, 3),
            Value::String("LineWidth".into()),
            Value::Num(3.0),
            Value::String("DisplayName".into()),
            Value::String("trace".into()),
        ]))
        .unwrap();
        let Value::Num(handle) = h else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("trace".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("LineWidth".into())]).unwrap(),
            Value::Num(3.0)
        );
    }

    #[test]
    fn line_default_display_name_is_empty() {
        let _guard = setup();
        let h = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0], 1, 2),
            tensor(&[3.0, 4.0], 1, 2),
        ]))
        .unwrap();
        let Value::Num(handle) = h else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String(String::new())
        );
    }

    #[test]
    fn line_constructor_accepts_parent_and_visible_properties() {
        let _guard = setup();
        configure_subplot(1, 2, 1).unwrap();
        let first_axes = crate::builtins::plotting::state::current_axes_handle_for_figure(
            current_figure_handle(),
        )
        .unwrap();
        configure_subplot(1, 2, 0).unwrap();
        let h = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0], 1, 2),
            tensor(&[3.0, 4.0], 1, 2),
            Value::String("Parent".into()),
            Value::Num(first_axes),
            Value::String("Visible".into()),
            Value::String("off".into()),
        ]))
        .unwrap();
        let Value::Num(handle) = h else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Visible".into())]).unwrap(),
            Value::String("off".into())
        );
        let parent = get_builtin(vec![Value::Num(handle), Value::String("Parent".into())]).unwrap();
        assert_eq!(parent, Value::Num(first_axes));
    }

    #[test]
    fn line_set_xdata_ydata_resizes_atomically() {
        let _guard = setup();
        let h = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0, 3.0], 1, 3),
            tensor(&[4.0, 5.0, 6.0], 1, 3),
        ]))
        .unwrap();
        let Value::Num(handle) = h else {
            panic!("expected scalar handle");
        };
        set_builtin(vec![
            Value::Num(handle),
            Value::String("XData".into()),
            tensor(&[7.0, 8.0, 9.0, 10.0], 1, 4),
            Value::String("YData".into()),
            tensor(&[1.0, 2.0, 3.0, 4.0], 1, 4),
        ])
        .unwrap();
        let x = get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap();
        let Value::Tensor(x) = x else {
            panic!("expected XData tensor");
        };
        assert_eq!(x.materialize_f64(), vec![7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn line_set_single_data_property_rejects_length_mismatch() {
        let _guard = setup();
        let h = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0, 3.0], 1, 3),
            tensor(&[4.0, 5.0, 6.0], 1, 3),
        ]))
        .unwrap();
        let Value::Num(handle) = h else {
            panic!("expected scalar handle");
        };
        let err = set_builtin(vec![
            Value::Num(handle),
            Value::String("XData".into()),
            tensor(&[7.0, 8.0, 9.0, 10.0], 1, 4),
        ])
        .expect_err("single mismatched XData update should fail");
        assert!(err.to_string().contains("same number of elements"));
    }

    #[test]
    fn line_matrix_inputs_create_one_handle_per_column() {
        let _guard = setup();
        let handles = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0, 1.0, 2.0], 2, 2),
            tensor(&[3.0, 4.0, 5.0, 6.0], 2, 2),
        ]))
        .unwrap();
        let Value::Tensor(handles) = handles else {
            panic!("expected handle row vector");
        };
        assert_eq!(handles.materialize_f64().len(), 2);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 2);
    }

    #[test]
    fn line_xyz_creates_line3_object() {
        let _guard = setup();
        let _ = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0], 1, 2),
            tensor(&[3.0, 4.0], 1, 2),
            tensor(&[5.0, 6.0], 1, 2),
        ]))
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Line3(_)));
    }

    #[test]
    fn line3_set_xyzdata_resizes_atomically() {
        let _guard = setup();
        let h = block_on(line_builtin(vec![
            tensor(&[1.0, 2.0], 1, 2),
            tensor(&[3.0, 4.0], 1, 2),
            tensor(&[5.0, 6.0], 1, 2),
        ]))
        .unwrap();
        let Value::Num(handle) = h else {
            panic!("expected scalar handle");
        };
        set_builtin(vec![
            Value::Num(handle),
            Value::String("XData".into()),
            tensor(&[1.0, 2.0, 3.0], 1, 3),
            Value::String("YData".into()),
            tensor(&[4.0, 5.0, 6.0], 1, 3),
            Value::String("ZData".into()),
            tensor(&[7.0, 8.0, 9.0], 1, 3),
        ])
        .unwrap();
        let z = get_builtin(vec![Value::Num(handle), Value::String("ZData".into())]).unwrap();
        let Value::Tensor(z) = z else {
            panic!("expected ZData tensor");
        };
        assert_eq!(z.materialize_f64(), vec![7.0, 8.0, 9.0]);
    }
}
