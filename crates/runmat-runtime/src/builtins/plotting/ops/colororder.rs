//! MATLAB-compatible `colororder` builtin.

use glam::Vec4;
use runmat_builtins::{
    COLORORDER_DESCRIPTOR, COLORORDER_ERROR_TOO_MANY_OUTPUTS, COLORORDER_INTEGER_RGB_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Tensor, Value};

use super::op_common::axes_target::{prepare_graphics_target, PreparedGraphicsTarget};
use super::plotting_error;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    color_order_for_axes_for_property, color_order_for_figure_for_property, current_figure_handle,
    set_color_order_for_axes_for_property, set_color_order_for_figure_for_property,
};
use super::style::{color_from_name_or_token, value_as_string};
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerOutputClassRule, COLORORDER_INTEGER_CAPABILITIES, COLORORDER_INTEGER_INPUTS,
};

const BUILTIN_NAME: &str = "colororder";

#[runtime_builtin(
    name = "colororder",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::colororder"
)]
pub fn colororder_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    colororder_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn colororder_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    colororder_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn colororder_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &COLORORDER_DESCRIPTOR,
        &COLORORDER_ERROR_TOO_MANY_OUTPUTS,
        "colororder",
    )?;
    match args.len() {
        0 => query_target(&PreparedGraphicsTarget::figure_alias(runtime, current_figure_handle())),
        1 => {
            if matches!(args[0], Value::Object(_) | Value::HandleObject(_)) {
                // Exact receiver failures are terminal, not palette misses.
                return query_target(&resolve_colororder_target(runtime, &args[0])?);
            }
            if let Ok(target) = resolve_colororder_target(runtime, &args[0]) {
                query_target(&target)
            } else {
                let colors = parse_color_order(&args[0])?;
                super::op_common::axes_output::validate_setter_output(BUILTIN_NAME)?;
                let target = PreparedGraphicsTarget::figure_alias(runtime, current_figure_handle());
                set_target(&target, &colors)?;
                Ok(colors_to_tensor(colors))
            }
        }
        2 => {
            let target = resolve_colororder_target(runtime, &args[0])?;
            let colors = parse_color_order(&args[1])?;
            super::op_common::axes_output::validate_setter_output(BUILTIN_NAME)?;
            set_target(&target, &colors)?;
            Ok(colors_to_tensor(colors))
        }
        _ => Err(colororder_err(
            "expected colororder(), colororder(colors), colororder(target), or colororder(target, colors)",
        )),
    }
}

fn query_target(target: &PreparedGraphicsTarget) -> BuiltinResult<Value> {
    let property = target.authority();
    let colors = match target.handle() {
        PlotHandle::Figure(handle) => color_order_for_figure_for_property(property, *handle),
        PlotHandle::Axes(handle, index) => {
            color_order_for_axes_for_property(property, *handle, *index)
        }
        _ => unreachable!("only Figure and Axes targets are prepared"),
    }
    .map_err(colororder_state_err)?;
    Ok(colors_to_tensor(colors))
}

fn set_target(target: &PreparedGraphicsTarget, colors: &[Vec4]) -> BuiltinResult<()> {
    let property = target.authority();
    match target.handle() {
        PlotHandle::Figure(handle) => {
            set_color_order_for_figure_for_property(property, *handle, colors)
        }
        PlotHandle::Axes(handle, index) => {
            set_color_order_for_axes_for_property(property, *handle, *index, colors)
        }
        _ => unreachable!("only Figure and Axes targets are prepared"),
    }
    .map_err(colororder_state_err)
}

fn resolve_colororder_target(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> BuiltinResult<PreparedGraphicsTarget> {
    if matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
    {
        return Err(colororder_err(
            "target must be a figure or axes graphics handle, not a typed integer",
        ));
    }
    if let Some(target) = prepare_graphics_target(runtime, BUILTIN_NAME, value)? {
        return Ok(target);
    }
    // Preserve owning legacy numeric-role errors; exact receiver failures from
    // the selection above are never swallowed as palette classification.
    match resolve_plot_handle(value, BUILTIN_NAME) {
        Ok(_) => Err(colororder_err("target must be a figure or axes handle")),
        Err(error) => Err(error),
    }
}

fn parse_color_order(value: &Value) -> BuiltinResult<Vec<Vec4>> {
    if let Some(text) = value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source(
            "colororder",
            source.to_string(),
            source,
        )
    })? {
        return parse_color_text_or_palette(&text);
    }
    match value {
        Value::Tensor(tensor) => {
            if tensor.integer_storage().is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLORORDER_INTEGER_RGB_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            colors_from_tensor(tensor)
        }
        Value::StringArray(strings) => colors_from_strings(
            &strings
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .ok_or_else(|| colororder_err("unsupported color order value"))
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
        ),
        Value::Cell(cell) => colors_from_cell(cell),
        Value::GpuTensor(_) => Err(colororder_err(
            "color arrays must be host scalars, strings, cell strings, or numeric RGB matrices",
        )),
        _ => Err(colororder_err("unsupported color order value")),
    }
}

fn colororder_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn colors_from_tensor(tensor: &Tensor) -> BuiltinResult<Vec<Vec4>> {
    let len = tensor_utils::tensor_element_len(tensor);
    if len == 0 {
        return Err(colororder_err(
            "color array must contain at least one color",
        ));
    }
    let (rows, cols) = colororder_matrix_dimensions(tensor);
    if len == 3 && (rows == 1 || cols == 1) {
        return Ok(vec![rgb_triplet(
            tensor_utils::tensor_value_f64(tensor, 0),
            tensor_utils::tensor_value_f64(tensor, 1),
            tensor_utils::tensor_value_f64(tensor, 2),
        )?]);
    }
    if cols != 3 {
        return Err(colororder_err("numeric color array must be m-by-3"));
    }
    let mut colors = Vec::with_capacity(rows);
    for row in 0..rows {
        let r = tensor_utils::tensor_value_f64(tensor, row);
        let g = tensor_utils::tensor_value_f64(tensor, rows + row);
        let b = tensor_utils::tensor_value_f64(tensor, 2 * rows + row);
        colors.push(rgb_triplet(r, g, b)?);
    }
    Ok(colors)
}

fn colors_from_cell(cell: &CellArray) -> BuiltinResult<Vec<Vec4>> {
    let mut strings = Vec::with_capacity(cell.data().len());
    for value in cell.data() {
        let text = value_as_string(value)
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "colororder",
                    source.to_string(),
                    source,
                )
            })?
            .ok_or_else(|| colororder_err("cell color arrays must contain strings"))?;
        strings.push(text);
    }
    colors_from_strings(&strings)
}

fn colors_from_strings<S: AsRef<str>>(strings: &[S]) -> BuiltinResult<Vec<Vec4>> {
    if strings.is_empty() {
        return Err(colororder_err(
            "color array must contain at least one color",
        ));
    }
    let mut colors = Vec::with_capacity(strings.len());
    for text in strings {
        let trimmed = text.as_ref().trim();
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

fn colororder_state_err(source: super::state::FigureError) -> crate::RuntimeError {
    let mut error = plotting_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {source}"));
    error.source = Some(Box::new(source));
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::subplot::subplot_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_value::{IntegerStorage, StringArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn colororder_numeric_rgb_preserves_dense_rank_boundaries() {
        let row = Tensor::new(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
        assert_eq!(
            colors_from_tensor(&row).unwrap(),
            vec![Vec4::new(1.0, 0.0, 0.0, 1.0)]
        );

        let scalar = Tensor::new(vec![1.0], Vec::new()).unwrap();
        assert!(colors_from_tensor(&scalar)
            .expect_err("rank-zero tensor has no RGB columns")
            .message()
            .contains("m-by-3"));
    }

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
            Tensor::new_integer(runmat_value::IntegerStorage::U8(data), vec![rows, 3]).unwrap();
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
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let colors = integer_rgb_matrix(vec![1, 0, 0, 0, 1, 0], 2);
        let set = colororder_builtin(vec![colors]).unwrap();
        assert_eq!(tensor_data(set), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn colororder_runmat_integer_extension_covers_all_eight_classes() {
        let _guard = setup_plot_tests();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![1, 0, 1]),
            IntegerStorage::I16(vec![1, 0, 1]),
            IntegerStorage::I32(vec![1, 0, 1]),
            IntegerStorage::I64(vec![1, 0, 1]),
            IntegerStorage::U8(vec![1, 0, 1]),
            IntegerStorage::U16(vec![1, 0, 1]),
            IntegerStorage::U32(vec![1, 0, 1]),
            IntegerStorage::U64(vec![1, 0, 1]),
        ];
        for storage in storages {
            let colors = Value::Tensor(Tensor::new_integer(storage, vec![1, 3]).unwrap());
            colororder_builtin(vec![colors]).expect("RunMat integer RGB extension");
            let queried = colororder_builtin(Vec::new()).unwrap();
            assert_eq!(tensor_data(queried), vec![1.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn colororder_integer_extension_is_gated_before_conversion() {
        let _guard = setup_plot_tests();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let colors = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![1, 0, 1]), vec![1, 3]).unwrap(),
        );
        let err = colororder_builtin(vec![colors]).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ColororderIntegerRgbExtension")
        );
    }

    #[test]
    fn colororder_integer_extension_rejects_channels_outside_zero_one() {
        let _guard = setup_plot_tests();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let colors = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 0, 1]), vec![1, 3]).unwrap(),
        );
        let err = colororder_builtin(vec![colors]).unwrap_err();
        assert!(err.to_string().contains("values in [0, 1]"));
    }

    #[test]
    fn colororder_rejects_typed_integer_targets_without_f64_aliasing() {
        let _guard = setup_plot_tests();
        let colors = rgb_matrix(vec![1.0, 0.0, 0.0], 1);
        let err = colororder_builtin(vec![
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            colors,
        ])
        .unwrap_err();
        assert!(err.to_string().contains("not a typed integer"));
    }

    #[test]
    fn colororder_descriptor_distinguishes_query_and_setter_outputs() {
        let query = COLORORDER_DESCRIPTOR
            .signatures
            .iter()
            .find(|signature| signature.label == "C = colororder()")
            .unwrap();
        assert_eq!(query.outputs.len(), 1);
        for label in [
            "colororder(colors)",
            "colororder(palettename)",
            "colororder(target, colors)",
        ] {
            let setter = COLORORDER_DESCRIPTOR
                .signatures
                .iter()
                .find(|signature| signature.label == label)
                .unwrap();
            assert!(setter.outputs.is_empty());
        }
        assert_eq!(COLORORDER_INTEGER_INPUTS[0].classes.len(), 8);
        assert_eq!(
            COLORORDER_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::NotApplicable
        );
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
        assert_eq!(tensor.cols(), 3);
        assert!(tensor.rows() >= 7);
    }

    #[test]
    fn colororder_rejects_gpu_color_arrays_without_gathering() {
        let _guard = setup_plot_tests();
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle;
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
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let _outputs =
                crate::output_count::push_output_count(Some(1)).expect("output-count scope");
            let error =
                colororder_builtin(vec![Value::Num(ax2), rgb_matrix(vec![1.0, 0.0, 0.0], 1)])
                    .expect_err("assigned targeted setter rejects before recoloring");
            assert_eq!(
                error.identifier(),
                runmat_builtins::AXES_CONTROL_SETTER_OUTPUT_EXTENSION.error_identifier
            );
            let queried = colororder_builtin(vec![Value::Num(ax2)]).unwrap();
            assert_close(&tensor_data(queried), &[0.2, 0.4, 0.6]);
            assert_eq!(
                crate::builtins::plotting::gca::gca_builtin(Vec::new()).unwrap(),
                Value::Num(ax1)
            );
        }
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
