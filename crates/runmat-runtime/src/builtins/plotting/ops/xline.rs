use runmat_builtins::{
    BuiltinErrorDescriptor, XLINE_DESCRIPTOR, XLINE_ERROR_INTERNAL, XLINE_ERROR_INVALID_ARGUMENT,
    XLINE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{ReferenceLine, ReferenceLineOrientation};
use runmat_value::NumericScalar;
use runmat_value::{Tensor, Value};

use super::op_common::annotation_error;
use super::plotting_error;
use super::state::{append_active_plot, register_reference_line_handle, PlotRenderOptions};
use super::style::{
    color_from_token, looks_like_option_name, parse_line_style_args, value_as_f64, value_as_string,
    LineAppearance, LineStyleParseOptions,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "xline";

#[runtime_builtin(
    name = "xline",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::xline"
)]
pub fn xline_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &XLINE_DESCRIPTOR,
        &XLINE_ERROR_TOO_MANY_OUTPUTS,
        "xline",
    )?;
    reference_line_builtin(
        BUILTIN_NAME,
        ReferenceLineOrientation::Vertical,
        &XLINE_ERROR_INVALID_ARGUMENT,
        &XLINE_ERROR_INTERNAL,
        args,
    )
}

pub(crate) fn reference_line_builtin(
    builtin: &'static str,
    orientation: ReferenceLineOrientation,
    invalid: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(annotation_error(
            builtin,
            invalid,
            reference_line_error(builtin, "expected a coordinate argument"),
        ));
    }
    let coords = coordinates_from_value(&args[0], builtin)
        .map_err(|error| annotation_error(builtin, invalid, error))?;
    let options = parse_reference_line_options(
        builtin,
        args.view().range(1..).expect("reference-line option tail"),
    )
    .map_err(|error| annotation_error(builtin, invalid, error))?;
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
            line.color_auto = options.color_auto;
            Ok(line)
        })
        .collect::<BuiltinResult<Vec<_>>>()
        .map_err(|error| annotation_error(builtin, invalid, error))?;

    let figure_handle = crate::builtins::plotting::current_figure_handle();

    let plot_indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices);
    let opts = PlotRenderOptions {
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
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(annotation_error(builtin, internal, err));
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
        let count = handles.len();
        Ok(Value::Tensor(
            Tensor::new(handles, vec![1, count]).expect("reference line handle row"),
        ))
    }
}

#[derive(Clone)]
struct ReferenceLineOptions {
    appearance: LineAppearance,
    label: Option<String>,
    display_name: Option<String>,
    label_orientation: String,
    visible: bool,
    /// No color given: the line uses the theme's axes color, as MATLAB's default does.
    color_auto: bool,
}

fn parse_reference_line_options(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ReferenceLineOptions> {
    let mut style_args = Vec::new();
    let mut label = None;
    let mut label_orientation = "aligned".to_string();
    let mut visible = true;
    let mut color_explicit = false;
    let mut width_explicit = false;

    let mut idx = 0usize;
    while idx < args.len() {
        let Some(text) = value_as_string(&args[idx]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                builtin,
                source.to_string(),
                source,
            )
        })?
        else {
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
                    let Some(value) = value_as_string(&args[idx + 1]).map_err(|source| {
                        crate::builtins::plotting::plotting_error_with_source(
                            builtin,
                            source.to_string(),
                            source,
                        )
                    })?
                    else {
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
                    visible = visible_from_value(&args[idx + 1]).ok_or_else(|| {
                        reference_line_error(builtin, "Visible must be 'on', 'off', 0, or 1")
                    })?;
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
        // MATLAB's default; drawn in the theme's axes color (`color_auto`).
        parsed.appearance.color = glam::Vec4::new(0.15, 0.15, 0.15, 1.0);
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
        color_auto: !color_explicit,
    })
}

fn visible_from_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::Num(value) if *value == 0.0 => Some(false),
        Value::Num(value) if *value == 1.0 => Some(true),
        Value::Int(value) => match value.try_to_u64()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        },
        Value::Tensor(tensor) if tensor.len() == 1 => {
            visible_from_numeric_scalar(tensor.numeric_value_at(0)?)
        }
        _ => {
            if crate::builtins::plotting::style::value_text_equals_ascii(value, "on", true) {
                Some(true)
            } else if crate::builtins::plotting::style::value_text_equals_ascii(value, "off", true)
            {
                Some(false)
            } else {
                None
            }
        }
    }
}

fn visible_from_numeric_scalar(value: NumericScalar) -> Option<bool> {
    match value {
        NumericScalar::F64(0.0) => Some(false),
        NumericScalar::F64(1.0) => Some(true),
        NumericScalar::F32(0.0) => Some(false),
        NumericScalar::F32(1.0) => Some(true),
        NumericScalar::I8(value) => visible_from_i128(value as i128),
        NumericScalar::I16(value) => visible_from_i128(value as i128),
        NumericScalar::I32(value) => visible_from_i128(value as i128),
        NumericScalar::I64(value) => visible_from_i128(value as i128),
        NumericScalar::U8(value) => visible_from_u128(value as u128),
        NumericScalar::U16(value) => visible_from_u128(value as u128),
        NumericScalar::U32(value) => visible_from_u128(value as u128),
        NumericScalar::U64(value) => visible_from_u128(value as u128),
        _ => None,
    }
}

fn visible_from_i128(value: i128) -> Option<bool> {
    match value {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

fn visible_from_u128(value: u128) -> Option<bool> {
    match value {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
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
    let tensor = Tensor::try_from(value).map_err(|err| reference_line_error(builtin, err))?;
    if tensor_utils::tensor_element_len(&tensor) == 0 {
        return Err(reference_line_error(
            builtin,
            "coordinate vector cannot be empty",
        ));
    }
    let values = tensor_utils::tensor_into_values_f64(tensor);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(reference_line_error(
            builtin,
            "coordinate values must be finite",
        ));
    }
    Ok(values)
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
    use crate::builtins::plotting::xlabel::xlabel_builtin;
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

    fn integer_tensor(data: &[i16]) -> Tensor {
        Tensor::new_integer(
            runmat_value::IntegerStorage::I16(data.to_vec()),
            vec![1, data.len()],
        )
        .unwrap()
    }

    #[test]
    fn xline_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = XLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = xline(x)"));
        assert!(labels.contains(&"h = xline(x, styleOrLabel)"));
        assert!(labels.contains(&"h = xline(x, Name, Value, ...)"));
    }

    #[test]
    fn xline_adds_reference_line_without_clearing_existing_plot() {
        let _guard = setup();
        futures::executor::block_on(plot_builtin(vec![
            Value::Tensor(tensor(&[0.0, 1.0])),
            Value::Tensor(tensor(&[2.0, 3.0])),
        ]))
        .unwrap();
        let handle = xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(0.5), Value::String("--r".into())],
        ))
        .unwrap();
        assert!(matches!(handle, Value::Num(_)));
        let figure = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(figure.len(), 2);
    }

    #[test]
    fn xline_supports_label_and_name_value_style() {
        let _guard = setup();
        let handle = xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(5.0),
                Value::String("--".into()),
                Value::String("Threshold".into()),
                Value::String("LineWidth".into()),
                Value::Num(3.0),
                Value::String("LabelOrientation".into()),
                Value::String("horizontal".into()),
            ],
        ))
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
        let handles = xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))],
        ))
        .unwrap();
        let tensor = Tensor::try_from(&handles).unwrap();
        assert_eq!(tensor.materialize_f64().len(), 3);
    }

    #[test]
    fn xline_reads_typed_integer_coordinates_exactly() {
        let _guard = setup();
        let handles = xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(integer_tensor(&[1, 2, 3]))],
        ))
        .unwrap();
        let tensor = Tensor::try_from(&handles).unwrap();
        assert_eq!(tensor.materialize_f64().len(), 3);
    }

    #[test]
    fn xline_visible_accepts_only_exact_numeric_zero_or_one() {
        let _guard = setup();
        xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(1.0),
                Value::String("Visible".into()),
                Value::Int(runmat_value::IntValue::U64(1)),
            ],
        ))
        .expect("integer one is visible");
        let error = xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::Num(2.0),
                Value::String("Visible".into()),
                Value::Int(runmat_value::IntValue::I8(2)),
            ],
        ))
        .expect_err("integer two is not a visibility value");
        assert!(error.message.contains("0, or 1"));
    }

    #[test]
    fn xline_preserves_existing_axis_labels() {
        let _guard = setup();
        xlabel_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("Time".into())],
        ))
        .unwrap();
        xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(5.0)],
        ))
        .unwrap();

        let figure = clone_figure(current_figure_handle()).unwrap();
        let meta = figure.axes_metadata(0).unwrap();
        assert_eq!(meta.x_label.as_deref(), Some("Time"));
    }
}

#[cfg(test)]
mod output_contract_tests {
    #[test]
    fn missing_required_input_uses_identity_diagnostic() {
        let error = super::xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .expect_err("missing annotation input");
        assert_eq!(error.identifier(), Some("RunMat:xline:InvalidArgument"));
    }
    #[test]
    fn excessive_outputs_reject_before_argument_or_graphics_effects() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = super::xline_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .expect_err("excess outputs");
        assert_eq!(error.identifier(), Some("RunMat:xline:TooManyOutputs"));
    }
}
