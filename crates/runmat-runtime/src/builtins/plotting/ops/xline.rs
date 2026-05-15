use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ReferenceLine, ReferenceLineOrientation};

use super::plotting_error;
use super::state::{append_active_plot, register_reference_line_handle, PlotRenderOptions};
use super::style::{
    color_from_token, looks_like_option_name, parse_line_style_args, value_as_bool, value_as_f64,
    value_as_string, LineAppearance, LineStyleParseOptions,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "xline";

#[runtime_builtin(
    name = "xline",
    category = "plotting",
    summary = "Draw vertical reference lines on the current axes.",
    keywords = "xline,reference,line,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::xline"
)]
pub fn xline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    reference_line_builtin(BUILTIN_NAME, ReferenceLineOrientation::Vertical, args)
}

pub(crate) fn reference_line_builtin(
    builtin: &'static str,
    orientation: ReferenceLineOrientation,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(reference_line_error(
            builtin,
            "expected a coordinate argument",
        ));
    }
    let coords = coordinates_from_value(&args[0], builtin)?;
    let options = parse_reference_line_options(builtin, &args[1..])?;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let mut lines = coords
        .iter()
        .map(|&value| {
            let mut line = ReferenceLine::new(orientation, value)
                .map_err(|err| reference_line_error(builtin, err))?
                .with_style(
                    options.appearance.color,
                    options.appearance.line_width,
                    options.appearance.line_style,
                );
            line.label = options.label.clone();
            line.display_name = options.display_name.clone();
            line.label_orientation = options.label_orientation.clone();
            line.visible = options.visible;
            Ok(line)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    let plot_indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices);
    let opts = PlotRenderOptions {
        title: "",
        x_label: "",
        y_label: "",
        ..Default::default()
    };
    let axes_index_slot = std::rc::Rc::new(std::cell::RefCell::new(None));
    let axes_index_out = std::rc::Rc::clone(&axes_index_slot);
    let render_result = append_active_plot(builtin, opts, move |figure, axes_index| {
        *axes_index_out.borrow_mut() = Some(axes_index);
        for line in lines.drain(..) {
            let plot_index = figure.add_reference_line_on_axes(line, axes_index);
            plot_indices_slot.borrow_mut().push(plot_index);
        }
        Ok(())
    });

    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if !(lower.contains("plotting is unavailable") || lower.contains("non-main thread")) {
            return Err(err);
        }
    }

    let axes_index = axes_index_slot.borrow().unwrap_or(0);
    let handles = plot_indices
        .borrow()
        .iter()
        .map(|plot_index| register_reference_line_handle(figure_handle, axes_index, *plot_index))
        .collect::<Vec<_>>();
    if handles.len() == 1 {
        Ok(Value::Num(handles[0]))
    } else {
        Ok(Value::Tensor(Tensor {
            data: handles.clone(),
            rows: 1,
            cols: handles.len(),
            shape: vec![1, handles.len()],
            dtype: runmat_builtins::NumericDType::F64,
        }))
    }
}

#[derive(Clone)]
struct ReferenceLineOptions {
    appearance: LineAppearance,
    label: Option<String>,
    display_name: Option<String>,
    label_orientation: String,
    visible: bool,
}

fn parse_reference_line_options(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<ReferenceLineOptions> {
    let mut style_args = Vec::new();
    let mut label = None;
    let mut label_orientation = "aligned".to_string();
    let mut visible = true;
    let mut color_explicit = false;
    let mut width_explicit = false;

    let mut idx = 0usize;
    while idx < args.len() {
        let Some(text) = value_as_string(&args[idx]) else {
            return Err(reference_line_error(
                builtin,
                "style and option names must be strings",
            ));
        };
        let lower = text.trim().to_ascii_lowercase();
        if lower == "labelorientation" || lower == "visible" {
            if idx + 1 >= args.len() {
                return Err(reference_line_error(
                    builtin,
                    "name-value arguments must come in pairs",
                ));
            }
            match lower.as_str() {
                "labelorientation" => {
                    let Some(value) = value_as_string(&args[idx + 1]) else {
                        return Err(reference_line_error(
                            builtin,
                            "LabelOrientation must be a string",
                        ));
                    };
                    let normalized = value.trim().to_ascii_lowercase();
                    match normalized.as_str() {
                        "aligned" | "horizontal" => label_orientation = normalized,
                        _ => {
                            return Err(reference_line_error(
                                builtin,
                                "LabelOrientation must be 'aligned' or 'horizontal'",
                            ));
                        }
                    }
                }
                "visible" => {
                    visible = value_as_bool(&args[idx + 1])
                        .or_else(|| {
                            value_as_string(&args[idx + 1]).map(|s| {
                                !matches!(s.trim().to_ascii_lowercase().as_str(), "off" | "false")
                            })
                        })
                        .ok_or_else(|| reference_line_error(builtin, "Visible must be boolean"))?;
                }
                _ => unreachable!(),
            }
            idx += 2;
            continue;
        }

        if looks_like_option_name(&lower) {
            if idx + 1 >= args.len() {
                return Err(reference_line_error(
                    builtin,
                    "name-value arguments must come in pairs",
                ));
            }
            if lower == "color" {
                color_explicit = true;
            }
            if lower == "linewidth" {
                width_explicit = true;
            }
            style_args.push(args[idx].clone());
            style_args.push(args[idx + 1].clone());
            idx += 2;
            continue;
        }

        if parse_line_style_args(
            &[args[idx].clone()],
            &LineStyleParseOptions::generic(builtin),
        )
        .is_ok()
        {
            color_explicit |= text.chars().any(|ch| color_from_token(ch).is_some());
            style_args.push(args[idx].clone());
        } else if label.is_none() {
            label = Some(text);
        } else {
            return Err(reference_line_error(
                builtin,
                "unexpected extra label or style argument",
            ));
        }
        idx += 1;
    }

    let mut parsed = parse_line_style_args(&style_args, &LineStyleParseOptions::generic(builtin))?;
    if !color_explicit {
        parsed.appearance.color = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
    }
    if !width_explicit {
        parsed.appearance.line_width = 1.0;
    }
    if parsed.label.is_some() {
        label = parsed.label.clone();
    }

    Ok(ReferenceLineOptions {
        appearance: parsed.appearance,
        label,
        display_name: parsed.label,
        label_orientation,
        visible,
    })
}

fn coordinates_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let value = value_as_f64(value).expect("scalar numeric value");
            if value.is_finite() {
                return Ok(vec![value]);
            }
            return Err(reference_line_error(
                builtin,
                "coordinate values must be finite",
            ));
        }
        _ => {}
    }
    let tensor =
        Tensor::try_from(value).map_err(|err| reference_line_error(builtin, err.to_string()))?;
    if tensor.data.is_empty() {
        return Err(reference_line_error(
            builtin,
            "coordinate vector cannot be empty",
        ));
    }
    if tensor.data.iter().any(|value| !value.is_finite()) {
        return Err(reference_line_error(
            builtin,
            "coordinate values must be finite",
        ));
    }
    Ok(tensor.data)
}

fn reference_line_error(builtin: &'static str, msg: impl Into<String>) -> crate::RuntimeError {
    plotting_error(builtin, format!("{builtin}: {}", msg.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        super::super::state::reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn tensor(data: &[f64]) -> Tensor {
        Tensor::new_2d(data.to_vec(), 1, data.len()).unwrap()
    }

    #[test]
    fn xline_adds_reference_line_without_clearing_existing_plot() {
        let _guard = setup();
        futures::executor::block_on(plot_builtin(vec![
            Value::Tensor(tensor(&[0.0, 1.0])),
            Value::Tensor(tensor(&[2.0, 3.0])),
        ]))
        .unwrap();
        let handle = xline_builtin(vec![Value::Num(0.5), Value::String("--r".into())]).unwrap();
        assert!(matches!(handle, Value::Num(_)));
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 2);
    }

    #[test]
    fn xline_supports_label_and_name_value_style() {
        let _guard = setup();
        let handle = xline_builtin(vec![
            Value::Num(5.0),
            Value::String("--".into()),
            Value::String("Threshold".into()),
            Value::String("LineWidth".into()),
            Value::Num(3.0),
            Value::String("LabelOrientation".into()),
            Value::String("horizontal".into()),
        ])
        .unwrap();
        let Value::Num(handle) = handle else {
            panic!("expected scalar handle");
        };
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Label".into())]).unwrap(),
            Value::String("Threshold".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("LineWidth".into())]).unwrap(),
            Value::Num(3.0)
        );
    }

    #[test]
    fn xline_vector_returns_handle_vector() {
        let _guard = setup();
        let handles = xline_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))]).unwrap();
        let tensor = Tensor::try_from(&handles).unwrap();
        assert_eq!(tensor.data.len(), 3);
    }
}
