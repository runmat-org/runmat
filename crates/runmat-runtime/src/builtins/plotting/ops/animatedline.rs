//! MATLAB-compatible `animatedline` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{Line3Plot, LinePlot};

use super::line::handles_value;
use super::op_common::line_inputs::NumericInput;
use super::op_common::{apply_axes_target, AxesTarget};
use super::plot::build_line_plot_for_builtin;
use super::plot3::build_line3_plot_for_builtin;
use super::plotting_error;
use super::properties::{resolve_plot_handle, set_properties, PlotHandle};
use super::state::{
    append_active_plot, current_axes_state, current_figure_handle,
    line_color_for_target_axes_series_index, register_animated_line_handle, PlotChildHandleState,
    PlotRenderOptions,
};
use super::style::{
    parse_line_style_args, value_as_f64, value_as_string, LineAppearance, LineStyleParseOptions,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "animatedline";
const DEFAULT_MAXIMUM_NUM_POINTS: usize = 1_000_000;

const ANIMATEDLINE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "an",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Animated line graphics handle.",
}];

const ANIMATEDLINE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional X/Y[/Z] vectors, axes target, and Name/Value properties.",
}];

const ANIMATEDLINE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "an = animatedline()",
        inputs: &[],
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(x, y)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(x, y, z)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(..., Name, Value)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "an = animatedline(ax, ...)",
        inputs: &ANIMATEDLINE_INPUTS,
        outputs: &ANIMATEDLINE_OUTPUTS,
    },
];

const ANIMATEDLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANIMATEDLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:animatedline:InvalidArgument"),
    when: "Coordinates, axes target, or Name/Value pairs are invalid.",
    message: "animatedline: invalid argument",
};

const ANIMATEDLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANIMATEDLINE.INTERNAL",
    identifier: Some("RunMat:animatedline:Internal"),
    when: "Internal plotting state update fails.",
    message: "animatedline: internal operation failed",
};

const ANIMATEDLINE_ERRORS: [BuiltinErrorDescriptor; 2] = [
    ANIMATEDLINE_ERROR_INVALID_ARGUMENT,
    ANIMATEDLINE_ERROR_INTERNAL,
];

pub const ANIMATEDLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANIMATEDLINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ANIMATEDLINE_ERRORS,
};

#[runtime_builtin(
    name = "animatedline",
    category = "plotting",
    summary = "Create an animated line graphics object.",
    keywords = "animatedline,addpoints,plotting,graphics,animation",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::animatedline::ANIMATEDLINE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::animatedline"
)]
pub async fn animatedline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (leading_axes, args) = split_leading_axes_handle_allow_empty(args)?;
    let parsed = parse_animatedline_args(leading_axes, args).await?;
    apply_axes_target(parsed.axes_target, BUILTIN_NAME)?;

    let figure_handle = current_figure_handle();
    let plot_index = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index);
    let axes_index = std::rc::Rc::new(std::cell::RefCell::new(None));
    let axes_index_slot = std::rc::Rc::clone(&axes_index);
    let mut object = Some(parsed.object);
    let render_result = append_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure, active_axes| {
            *axes_index_slot.borrow_mut() = Some(active_axes);
            let object = object
                .take()
                .expect("animatedline plot object consumed exactly once");
            let (index, is_3d) = match object {
                AnimatedLineObject::TwoD(plot) => {
                    (figure.add_line_plot_on_axes(plot, active_axes), false)
                }
                AnimatedLineObject::ThreeD(plot) => {
                    figure.set_axes_z_limits(active_axes, None);
                    (figure.add_line3_plot_on_axes(plot, active_axes), true)
                }
            };
            *plot_index_slot.borrow_mut() = Some((index, is_3d));
            Ok(())
        },
    );

    let axes_index = axes_index.borrow().unwrap_or(0);
    let (plot_index, is_3d) = plot_index.borrow().unwrap_or((0, false));
    let handle = register_animated_line_handle(
        figure_handle,
        axes_index,
        plot_index,
        is_3d,
        parsed.maximum_num_points,
    );

    if !parsed.property_pairs.is_empty() {
        let state = PlotChildHandleState::AnimatedLine(super::state::AnimatedLineHandleState {
            figure: figure_handle,
            axes_index,
            plot_index,
            is_3d,
            maximum_num_points: parsed.maximum_num_points,
        });
        set_properties(
            PlotHandle::PlotChild(handle, Box::new(state)),
            &parsed.property_pairs,
            BUILTIN_NAME,
        )?;
    }

    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(plotting_error(BUILTIN_NAME, err.to_string()));
        }
    }
    Ok(handles_value(vec![handle]))
}

fn split_leading_axes_handle_allow_empty(
    args: Vec<Value>,
) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, BUILTIN_NAME) {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

struct ParsedAnimatedLine {
    axes_target: AxesTarget,
    object: AnimatedLineObject,
    maximum_num_points: Option<usize>,
    property_pairs: Vec<Value>,
}

enum AnimatedLineObject {
    TwoD(LinePlot),
    ThreeD(Line3Plot),
}

async fn parse_animatedline_args(
    mut axes_target: AxesTarget,
    args: Vec<Value>,
) -> BuiltinResult<ParsedAnimatedLine> {
    let mut idx = 0usize;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = None;
    if args.len() >= 2 && is_numeric_value(&args[0]) && is_numeric_value(&args[1]) {
        x = numeric_vector(args[0].clone(), "X").await?;
        y = numeric_vector(args[1].clone(), "Y").await?;
        idx = 2;
        if args.get(idx).is_some_and(is_numeric_value) {
            z = Some(numeric_vector(args[idx].clone(), "Z").await?);
            idx += 1;
        }
    } else if args.first().is_some_and(is_numeric_value) {
        return Err(animatedline_err("expected both X and Y coordinate vectors"));
    }

    let mut property_pairs = Vec::new();
    let mut style_tokens = Vec::new();
    let mut maximum_num_points = Some(DEFAULT_MAXIMUM_NUM_POINTS);
    while idx < args.len() {
        let key = value_as_string(&args[idx])
            .ok_or_else(|| animatedline_err("property names must be strings"))?;
        idx += 1;
        if idx >= args.len() {
            return Err(animatedline_err("name-value arguments must come in pairs"));
        }
        let value = args[idx].clone();
        idx += 1;
        match key.trim().to_ascii_lowercase().as_str() {
            "xdata" => {
                x = numeric_vector(value, "XData").await?;
            }
            "ydata" => {
                y = numeric_vector(value, "YData").await?;
            }
            "zdata" => {
                z = Some(numeric_vector(value, "ZData").await?);
            }
            "parent" => {
                let PlotHandle::Axes(handle, axes_index) =
                    resolve_plot_handle(&value, BUILTIN_NAME)?
                else {
                    return Err(animatedline_err("Parent must be an axes handle"));
                };
                axes_target = Some((handle, axes_index));
            }
            "maximumnumpoints" => {
                maximum_num_points = maximum_from_value(&value)?;
            }
            _ => {
                style_tokens.push(Value::String(key.clone()));
                style_tokens.push(value.clone());
                property_pairs.push(Value::String(key));
                property_pairs.push(value);
            }
        }
    }

    if x.len() != y.len() || z.as_ref().is_some_and(|z| z.len() != x.len()) {
        return Err(animatedline_err(
            "coordinate vectors must have the same length",
        ));
    }

    trim_initial_points(maximum_num_points, &mut x, &mut y, z.as_mut());

    let parsed_style =
        parse_line_style_args(&style_tokens, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    let mut appearance = parsed_style.appearance;
    if !parsed_style.color_explicit {
        let target = axes_target.unwrap_or_else(|| {
            let axes = current_axes_state();
            (axes.handle, axes.active_index)
        });
        appearance.color = line_color_for_target_axes_series_index(target.0, target.1, 0);
    }
    let label = parsed_style.label.unwrap_or_default();
    let object = build_object(x, y, z, &label, &appearance)?;
    Ok(ParsedAnimatedLine {
        axes_target,
        object,
        maximum_num_points,
        property_pairs,
    })
}

fn build_object(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Option<Vec<f64>>,
    label: &str,
    appearance: &LineAppearance,
) -> BuiltinResult<AnimatedLineObject> {
    if let Some(z) = z {
        if !x.is_empty() {
            if appearance.marker.is_some() {
                return Err(animatedline_err(
                    "3-D animated lines do not support marker properties yet",
                ));
            }
            return Ok(AnimatedLineObject::ThreeD(build_line3_plot_for_builtin(
                BUILTIN_NAME,
                x,
                y,
                z,
                label,
                appearance,
            )?));
        }
    }
    Ok(AnimatedLineObject::TwoD(build_line_plot_for_builtin(
        BUILTIN_NAME,
        x,
        y,
        label,
        appearance,
    )?))
}

async fn numeric_vector(value: Value, name: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = NumericInput::from_value(value, BUILTIN_NAME)?
        .into_tensor_async(BUILTIN_NAME)
        .await?;
    if !is_vector_tensor(&tensor) {
        return Err(animatedline_err(format!(
            "{name} must be a scalar or vector"
        )));
    }
    Ok(tensor_utils::tensor_into_values_f64(tensor))
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn is_vector_tensor(tensor: &Tensor) -> bool {
    tensor.rows() == 1 || tensor.cols() == 1
}

fn maximum_from_value(value: &Value) -> BuiltinResult<Option<usize>> {
    let Some(maximum) = value_as_f64(value) else {
        return Err(animatedline_err("MaximumNumPoints must be numeric"));
    };
    if maximum.is_infinite() && maximum.is_sign_positive() {
        return Ok(None);
    }
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err(animatedline_err("MaximumNumPoints must be positive or Inf"));
    }
    let rounded = maximum.round();
    if (rounded - maximum).abs() > f64::EPSILON {
        return Err(animatedline_err(
            "MaximumNumPoints must be a positive integer or Inf",
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(animatedline_err("MaximumNumPoints is too large"));
    }
    Ok(Some(rounded as usize))
}

fn trim_initial_points(
    maximum: Option<usize>,
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    z: Option<&mut Vec<f64>>,
) {
    let Some(maximum) = maximum else {
        return;
    };
    if x.len() <= maximum {
        return;
    }
    let drop = x.len() - maximum;
    x.drain(0..drop);
    y.drain(0..drop);
    if let Some(z) = z {
        z.drain(0..drop);
    }
}

fn animatedline_err(detail: impl AsRef<str>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, detail.as_ref().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use futures::executor::block_on;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn animatedline_descriptor_covers_core_forms() {
        let labels = ANIMATEDLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"an = animatedline()"));
        assert!(labels.contains(&"an = animatedline(x, y, z)"));
        assert!(labels.contains(&"an = animatedline(ax, ...)"));
    }

    #[test]
    fn animatedline_creates_empty_handle_with_default_limit() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(Vec::new())).unwrap();
        assert!(matches!(handle, Value::Num(_)));
        let max = get_builtin(vec![
            handle.clone(),
            Value::String("MaximumNumPoints".into()),
        ])
        .unwrap();
        assert_eq!(max, Value::Num(DEFAULT_MAXIMUM_NUM_POINTS as f64));
        let ty = get_builtin(vec![handle, Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("animatedline".into()));
    }

    #[test]
    fn animatedline_accepts_axes_handle_without_trailing_args() {
        let _guard = setup();
        let axes = crate::builtins::plotting::axes::axes_builtin(Vec::new()).unwrap();
        let handle = block_on(animatedline_builtin(vec![Value::Num(axes)])).unwrap();
        let parent = get_builtin(vec![handle, Value::String("Parent".into())]).unwrap();
        assert_eq!(parent, Value::Num(axes));
    }

    #[test]
    fn animatedline_trims_initial_data_to_maximum() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(vec![
            vector(&[1.0, 2.0, 3.0]),
            vector(&[4.0, 5.0, 6.0]),
            Value::String("MaximumNumPoints".into()),
            Value::Num(2.0),
        ]))
        .unwrap();
        let x = get_builtin(vec![handle, Value::String("XData".into())]).unwrap();
        assert_eq!(tensor_data(x), vec![2.0, 3.0]);
    }

    #[test]
    fn animatedline_rejects_unrepresentable_maximum_before_cast() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(maximum_from_value(&Value::Num(boundary)).is_err());
        assert!(maximum_from_value(&Value::Num(2.5)).is_err());
    }

    #[test]
    fn animatedline_accepts_property_constructor_data() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(vec![
            Value::String("XData".into()),
            vector(&[1.0, 2.0]),
            Value::String("YData".into()),
            vector(&[3.0, 4.0]),
            Value::String("ZData".into()),
            vector(&[5.0, 6.0]),
            Value::String("LineWidth".into()),
            Value::Num(2.0),
        ]))
        .unwrap();
        let z = get_builtin(vec![handle.clone(), Value::String("ZData".into())]).unwrap();
        assert_eq!(tensor_data(z), vec![5.0, 6.0]);
        let width = get_builtin(vec![handle, Value::String("LineWidth".into())]).unwrap();
        assert_eq!(width, Value::Num(2.0));
    }

    #[test]
    fn animatedline_reads_typed_integer_constructor_data_exactly() {
        let _guard = setup();
        let handle = block_on(animatedline_builtin(vec![
            integer_vector(&[1, 2]),
            integer_vector(&[10, 20]),
            integer_vector(&[100, 200]),
        ]))
        .unwrap();
        assert_eq!(
            tensor_data(get_builtin(vec![handle.clone(), Value::String("XData".into())]).unwrap()),
            vec![1.0, 2.0]
        );
        assert_eq!(
            tensor_data(get_builtin(vec![handle.clone(), Value::String("YData".into())]).unwrap()),
            vec![10.0, 20.0]
        );
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::String("ZData".into())]).unwrap()),
            vec![100.0, 200.0]
        );
    }

    #[test]
    fn animatedline_rejects_3d_marker_constructor() {
        let _guard = setup();
        let err = block_on(animatedline_builtin(vec![
            vector(&[1.0]),
            vector(&[2.0]),
            vector(&[3.0]),
            Value::String("Marker".into()),
            Value::String("o".into()),
        ]))
        .unwrap_err();
        assert!(err.message.contains("marker"));
    }

    fn vector(values: &[f64]) -> Value {
        Value::Tensor(
            Tensor::new(values.to_vec(), vec![1, values.len()]).expect("animatedline row vector"),
        )
    }

    fn integer_vector(values: &[i16]) -> Value {
        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(values.to_vec()),
            vec![1, values.len()],
        )
        .unwrap();
        Value::Tensor(tensor)
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
