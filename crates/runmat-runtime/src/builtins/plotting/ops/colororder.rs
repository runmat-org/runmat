//! MATLAB-compatible `colororder` builtin.

use glam::Vec4;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    color_order_for_axes, color_order_for_figure, current_figure_handle, set_color_order_for_axes,
    set_color_order_for_figure, FigureHandle,
};
use super::style::{color_from_name_or_token, value_as_string};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::get_type;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "colororder";

const COLORORDER_OUTPUT_COLORS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current color order as an m-by-3 RGB triplet matrix.",
}];

const COLORORDER_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const COLORORDER_INPUTS_COLORS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "colors",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 RGB matrix, color name array, color cellstr, palette name, or default.",
}];

const COLORORDER_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target figure or axes handle.",
}];

const COLORORDER_INPUTS_TARGET_COLORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target figure or axes handle.",
    },
    BuiltinParamDescriptor {
        name: "colors",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "m-by-3 RGB matrix, color name array, color cellstr, palette name, or default.",
    },
];

const COLORORDER_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "C = colororder()",
        inputs: &COLORORDER_INPUTS_NONE,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
    BuiltinSignatureDescriptor {
        label: "C = colororder(target)",
        inputs: &COLORORDER_INPUTS_TARGET,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
    BuiltinSignatureDescriptor {
        label: "colororder(colors)",
        inputs: &COLORORDER_INPUTS_COLORS,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
    BuiltinSignatureDescriptor {
        label: "colororder(palettename)",
        inputs: &COLORORDER_INPUTS_COLORS,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
    BuiltinSignatureDescriptor {
        label: "colororder(target, colors)",
        inputs: &COLORORDER_INPUTS_TARGET_COLORS,
        outputs: &COLORORDER_OUTPUT_COLORS,
    },
];

const COLORORDER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORORDER.INVALID_ARGUMENT",
    identifier: Some("RunMat:colororder:InvalidArgument"),
    when: "Target handle, color array, palette name, or color specification is invalid.",
    message: "colororder: invalid argument",
};

const COLORORDER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORORDER.INTERNAL",
    identifier: Some("RunMat:colororder:Internal"),
    when: "Internal plotting state update fails.",
    message: "colororder: internal operation failed",
};

const COLORORDER_ERRORS: [BuiltinErrorDescriptor; 2] =
    [COLORORDER_ERROR_INVALID_ARGUMENT, COLORORDER_ERROR_INTERNAL];

pub const COLORORDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORORDER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORORDER_ERRORS,
};

#[runtime_builtin(
    name = "colororder",
    category = "plotting",
    summary = "Set or query the plotting color order palette.",
    keywords = "colororder,plotting,axes,figure,color,palette",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::colororder::COLORORDER_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::colororder"
)]
pub fn colororder_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        0 => query_current_figure(),
        1 => {
            if let Ok(target) = resolve_colororder_target(&args[0]) {
                query_target(target)
            } else {
                let colors = parse_color_order(&args[0])?;
                let handle = current_figure_handle();
                set_color_order_for_figure(handle, &colors)
                    .map_err(|err| colororder_state_err(err.to_string()))?;
                Ok(colors_to_tensor(colors))
            }
        }
        2 => {
            let target = resolve_colororder_target(&args[0])?;
            let colors = parse_color_order(&args[1])?;
            match target {
                ColorOrderTarget::Figure(handle) => set_color_order_for_figure(handle, &colors)
                    .map_err(|err| colororder_state_err(err.to_string()))?,
                ColorOrderTarget::Axes(handle, axes_index) => {
                    set_color_order_for_axes(handle, axes_index, &colors)
                        .map_err(|err| colororder_state_err(err.to_string()))?
                }
            }
            Ok(colors_to_tensor(colors))
        }
        _ => Err(colororder_err(
            "expected colororder(), colororder(colors), colororder(target), or colororder(target, colors)",
        )),
    }
}

#[derive(Clone, Copy)]
enum ColorOrderTarget {
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
}

fn query_current_figure() -> BuiltinResult<Value> {
    query_target(ColorOrderTarget::Figure(current_figure_handle()))
}

fn query_target(target: ColorOrderTarget) -> BuiltinResult<Value> {
    let colors = match target {
        ColorOrderTarget::Figure(handle) => color_order_for_figure(handle),
        ColorOrderTarget::Axes(handle, axes_index) => color_order_for_axes(handle, axes_index),
    }
    .map_err(|err| colororder_state_err(err.to_string()))?;
    Ok(colors_to_tensor(colors))
}

fn resolve_colororder_target(value: &Value) -> BuiltinResult<ColorOrderTarget> {
    match resolve_plot_handle(value, BUILTIN_NAME) {
        Ok(PlotHandle::Figure(handle)) => Ok(ColorOrderTarget::Figure(handle)),
        Ok(PlotHandle::Axes(handle, axes_index)) => Ok(ColorOrderTarget::Axes(handle, axes_index)),
        Ok(_) => Err(colororder_err("target must be a figure or axes handle")),
        Err(err) => Err(err),
    }
}

fn parse_color_order(value: &Value) -> BuiltinResult<Vec<Vec4>> {
    if let Some(text) = value_as_string(value) {
        return parse_color_text_or_palette(&text);
    }
    match value {
        Value::Tensor(tensor) => colors_from_tensor(tensor),
        Value::StringArray(strings) => colors_from_strings(&strings.data),
        Value::Cell(cell) => colors_from_cell(cell),
        Value::GpuTensor(_) => Err(colororder_err(
            "color arrays must be host scalars, strings, cell strings, or numeric RGB matrices",
        )),
        _ => Err(colororder_err("unsupported color order value")),
    }
}

fn colors_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec4>> {
    let len = tensor_utils::tensor_element_len(tensor);
    if len == 0 {
        return Err(colororder_err(
            "color array must contain at least one color",
        ));
    }
    if len == 3 && (tensor.rows == 1 || tensor.cols == 1) {
        return Ok(vec![rgb_triplet(
            tensor_utils::tensor_value_f64(tensor, 0),
            tensor_utils::tensor_value_f64(tensor, 1),
            tensor_utils::tensor_value_f64(tensor, 2),
        )?]);
    }
    if tensor.cols != 3 {
        return Err(colororder_err("numeric color array must be m-by-3"));
    }
    let mut colors = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let r = tensor_utils::tensor_value_f64(tensor, row);
        let g = tensor_utils::tensor_value_f64(tensor, tensor.rows + row);
        let b = tensor_utils::tensor_value_f64(tensor, 2 * tensor.rows + row);
        colors.push(rgb_triplet(r, g, b)?);
    }
    Ok(colors)
}

fn colors_from_cell(cell: &CellArray) -> BuiltinResult<Vec<Vec4>> {
    let mut strings = Vec::with_capacity(cell.data.len());
    for value in &cell.data {
        let text = value_as_string(value)
            .ok_or_else(|| colororder_err("cell color arrays must contain strings"))?;
        strings.push(text);
    }
    colors_from_strings(&strings)
}

fn colors_from_strings(strings: &[String]) -> BuiltinResult<Vec<Vec4>> {
    if strings.is_empty() {
        return Err(colororder_err(
            "color array must contain at least one color",
        ));
    }
    let mut colors = Vec::with_capacity(strings.len());
    for text in strings {
        let trimmed = text.trim();
        if let Some(color) = color_from_name_or_token(trimmed) {
            colors.push(color);
        } else {
            return Err(colororder_err(format!(
                "unsupported color specification `{trimmed}`"
            )));
        }
    }
    Ok(colors)
}

fn parse_color_text_or_palette(text: &str) -> BuiltinResult<Vec<Vec4>> {
    let trimmed = text.trim();
    if let Some(palette) = named_palette(trimmed) {
        return Ok(palette);
    }
    if let Some(color) = color_from_name_or_token(trimmed) {
        return Ok(vec![color]);
    }
    Err(colororder_err(format!(
        "unsupported color specification or palette `{trimmed}`"
    )))
}

fn rgb_triplet(r: f64, g: f64, b: f64) -> BuiltinResult<Vec4> {
    for component in [r, g, b] {
        if !component.is_finite() || !(0.0..=1.0).contains(&component) {
            return Err(colororder_err(
                "RGB triplet components must be finite values in [0, 1]",
            ));
        }
    }
    Ok(Vec4::new(r as f32, g as f32, b as f32, 1.0))
}

fn colors_to_tensor(colors: Vec<Vec4>) -> Value {
    let rows = colors.len();
    let mut data = Vec::with_capacity(rows * 3);
    for color in &colors {
        data.push(color.x as f64);
    }
    for color in &colors {
        data.push(color.y as f64);
    }
    for color in &colors {
        data.push(color.z as f64);
    }
    Value::Tensor(Tensor::new(data, vec![rows, 3]).expect("color-order matrix"))
}

fn named_palette(name: &str) -> Option<Vec<Vec4>> {
    match name.to_ascii_lowercase().as_str() {
        "default" | "gem" => Some(vec![
            Vec4::new(0.0660, 0.4430, 0.7450, 1.0),
            Vec4::new(0.8660, 0.3290, 0.0000, 1.0),
            Vec4::new(0.9290, 0.6940, 0.1250, 1.0),
            Vec4::new(0.5210, 0.0860, 0.8190, 1.0),
            Vec4::new(0.2310, 0.6660, 0.1960, 1.0),
            Vec4::new(0.1840, 0.7450, 0.9370, 1.0),
            Vec4::new(0.8190, 0.0150, 0.5450, 1.0),
        ]),
        "gem12" => Some(vec![
            Vec4::new(0.0660, 0.4430, 0.7450, 1.0),
            Vec4::new(0.8660, 0.3290, 0.0000, 1.0),
            Vec4::new(0.9290, 0.6940, 0.1250, 1.0),
            Vec4::new(0.5210, 0.0860, 0.8190, 1.0),
            Vec4::new(0.2310, 0.6660, 0.1960, 1.0),
            Vec4::new(0.1840, 0.7450, 0.9370, 1.0),
            Vec4::new(0.8190, 0.0150, 0.5450, 1.0),
            Vec4::new(0.6500, 0.6500, 0.6500, 1.0),
            Vec4::new(0.1000, 0.6000, 0.5000, 1.0),
            Vec4::new(0.9500, 0.4500, 0.7000, 1.0),
            Vec4::new(0.5500, 0.4000, 0.1000, 1.0),
            Vec4::new(0.3500, 0.3500, 0.9000, 1.0),
        ]),
        "glow" => Some(vec![
            Vec4::new(0.9500, 0.8500, 0.2000, 1.0),
            Vec4::new(0.3500, 0.8500, 1.0000, 1.0),
            Vec4::new(1.0000, 0.4500, 0.7000, 1.0),
            Vec4::new(0.5500, 1.0000, 0.3500, 1.0),
            Vec4::new(0.8500, 0.5500, 1.0000, 1.0),
            Vec4::new(1.0000, 0.6500, 0.2500, 1.0),
            Vec4::new(0.2500, 0.9500, 0.7500, 1.0),
        ]),
        "glow12" => {
            let mut palette = named_palette("glow")?;
            palette.extend([
                Vec4::new(0.7000, 0.9000, 1.0000, 1.0),
                Vec4::new(1.0000, 0.8000, 0.9000, 1.0),
                Vec4::new(0.8500, 1.0000, 0.7000, 1.0),
                Vec4::new(0.9500, 0.7500, 1.0000, 1.0),
                Vec4::new(1.0000, 0.9000, 0.5500, 1.0),
            ]);
            Some(palette)
        }
        "sail" => Some(vec![
            Vec4::new(0.0000, 0.4470, 0.7410, 1.0),
            Vec4::new(0.3000, 0.7450, 0.9330, 1.0),
            Vec4::new(0.4660, 0.6740, 0.1880, 1.0),
            Vec4::new(0.8500, 0.3250, 0.0980, 1.0),
        ]),
        "reef" => Some(vec![
            Vec4::new(0.0000, 0.5000, 0.6500, 1.0),
            Vec4::new(0.0000, 0.7000, 0.4500, 1.0),
            Vec4::new(0.6500, 0.8000, 0.3000, 1.0),
            Vec4::new(0.9500, 0.7000, 0.2500, 1.0),
        ]),
        "meadow" => Some(vec![
            Vec4::new(0.2000, 0.6000, 0.2500, 1.0),
            Vec4::new(0.4500, 0.7500, 0.3000, 1.0),
            Vec4::new(0.7500, 0.6500, 0.2500, 1.0),
            Vec4::new(0.3000, 0.5500, 0.7500, 1.0),
        ]),
        "dye" => Some(vec![
            Vec4::new(0.1500, 0.2500, 0.8500, 1.0),
            Vec4::new(0.8000, 0.1000, 0.6000, 1.0),
            Vec4::new(0.9500, 0.2500, 0.1500, 1.0),
            Vec4::new(0.2000, 0.7500, 0.8500, 1.0),
        ]),
        "earth" => Some(vec![
            Vec4::new(0.4500, 0.3000, 0.1500, 1.0),
            Vec4::new(0.7000, 0.5000, 0.2500, 1.0),
            Vec4::new(0.2500, 0.5000, 0.3500, 1.0),
            Vec4::new(0.5500, 0.6500, 0.7500, 1.0),
        ]),
        _ => None,
    }
}

fn colororder_err(message: impl Into<String>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {}", message.into()))
}

fn colororder_state_err(message: impl Into<String>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_accelerate_api::GpuTensorHandle;
    use runmat_builtins::StringArray;

    fn setup_plot_tests() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn rgb_matrix(data: Vec<f64>, rows: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, 3]).expect("RGB matrix"))
    }

    fn integer_rgb_matrix(data: Vec<u8>, rows: usize) -> Value {
        let tensor =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U8(data), vec![rows, 3]).unwrap();
        Value::Tensor(tensor)
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        Tensor::try_from(&value).unwrap().materialize_f64()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1.0e-6, "value {idx}: expected {e}, got {a}");
        }
    }

    #[test]
    fn colororder_sets_and_queries_current_figure_matrix() {
        let _guard = setup_plot_tests();
        let colors = rgb_matrix(vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0], 2);
        let set = colororder_builtin(vec![colors]).unwrap();
        assert_eq!(tensor_data(set), vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
        let queried = colororder_builtin(Vec::new()).unwrap();
        assert_eq!(tensor_data(queried), vec![1.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn colororder_reads_typed_integer_rgb_storage_exactly() {
        let _guard = setup_plot_tests();
        let colors = integer_rgb_matrix(vec![1, 0, 0, 0, 1, 0], 2);
        let set = colororder_builtin(vec![colors]).unwrap();
        assert_eq!(tensor_data(set), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn colororder_accepts_names_hex_cells_and_palettes() {
        let _guard = setup_plot_tests();
        let strings = Value::StringArray(
            StringArray::new(vec!["red".into(), "#0F0".into(), "b".into()], vec![1, 3]).unwrap(),
        );
        let set = colororder_builtin(vec![strings]).unwrap();
        assert_eq!(
            tensor_data(set),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );

        let cell = Value::Cell(
            CellArray::new(
                vec![
                    Value::String("cyan".into()),
                    Value::String("#ff00ff".into()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let set = colororder_builtin(vec![cell]).unwrap();
        assert_eq!(tensor_data(set), vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

        let gem = colororder_builtin(vec![Value::String("default".into())]).unwrap();
        let tensor = Tensor::try_from(&gem).unwrap();
        assert_eq!(tensor.cols, 3);
        assert!(tensor.rows >= 7);
    }

    #[test]
    fn colororder_rejects_gpu_color_arrays_without_gathering() {
        let _guard = setup_plot_tests();
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: 7,
            buffer_id: 42,
        };
        let err = colororder_builtin(vec![Value::GpuTensor(handle)])
            .expect_err("gpu color arrays must stay unsupported at the API boundary");
        assert!(err
            .to_string()
            .contains("color arrays must be host scalars"));
    }

    #[test]
    fn colororder_target_axes_does_not_change_active_axes() {
        let _guard = setup_plot_tests();
        let ax1 = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(1.0)).unwrap();
        let ax2 = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let _ = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(1.0)).unwrap();

        let colors = rgb_matrix(vec![0.2, 0.4, 0.6], 1);
        let _ = colororder_builtin(vec![Value::Num(ax2), colors]).unwrap();
        let active = crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap();
        assert_eq!(active, Value::Num(ax1));

        let queried = colororder_builtin(vec![Value::Num(ax2)]).unwrap();
        assert_close(&tensor_data(queried), &[0.2, 0.4, 0.6]);
    }

    #[test]
    fn colororder_recolors_existing_and_future_implicit_lines() {
        let _guard = setup_plot_tests();
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("line data"));
        let _ = futures::executor::block_on(plot_builtin(vec![y.clone()])).unwrap();
        let colors = rgb_matrix(vec![1.0, 0.0, 0.0], 1);
        let _ = colororder_builtin(vec![colors]).unwrap();

        let fig = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        let first = fig.plots().next().unwrap();
        let runmat_plot::plots::PlotElement::Line(line) = first else {
            panic!("expected line");
        };
        assert_eq!(line.color, Vec4::new(1.0, 0.0, 0.0, 1.0));

        let _ = futures::executor::block_on(plot_builtin(vec![y])).unwrap();
        let fig = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        let second = fig.plots().next().unwrap();
        let runmat_plot::plots::PlotElement::Line(line) = second else {
            panic!("expected line");
        };
        assert_eq!(line.color, Vec4::new(1.0, 0.0, 0.0, 1.0));
    }
}
