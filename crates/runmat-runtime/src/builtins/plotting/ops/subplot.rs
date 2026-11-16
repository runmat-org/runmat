//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::{
    BuiltinErrorDescriptor, SUBPLOT_DESCRIPTOR, SUBPLOT_ERROR_INDEX_OUT_OF_RANGE,
    SUBPLOT_ERROR_INTERNAL, SUBPLOT_ERROR_INVALID_ARGUMENT, SUBPLOT_ERROR_TOO_MANY_OUTPUTS,
    SUBPLOT_INTEGER_GRID_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::AxesGridCell;
use runmat_value::{NumericScalar, Value};

use super::op_common::cmd_parsing::scalar_from_value;
use super::state::{encode_axes_handle, FigureError};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "subplot";

fn subplot_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    subplot_error_with_message(error.message, error)
}

fn subplot_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_subplot_figure_error(err: FigureError) -> RuntimeError {
    match err {
        FigureError::InvalidSubplotIndex { .. } => subplot_error(&SUBPLOT_ERROR_INDEX_OUT_OF_RANGE),
        FigureError::InvalidSubplotGrid { .. } => subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT),
        other => {
            let mut error = subplot_error_with_message(
                format!("{}: {}", SUBPLOT_ERROR_INTERNAL.message, other),
                &SUBPLOT_ERROR_INTERNAL,
            );
            error.source = Some(Box::new(other));
            error
        }
    }
}

#[runtime_builtin(
    name = "subplot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::subplot"
)]
pub fn subplot_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    rows: Value,
    cols: Value,
    position: Value,
) -> crate::BuiltinResult<f64> {
    subplot_with_runtime(runtime.as_ref(), rows, cols, position)
}

#[cfg(test)]
pub fn subplot_builtin(rows: Value, cols: Value, position: Value) -> crate::BuiltinResult<f64> {
    subplot_with_runtime(None, rows, cols, position)
}

fn subplot_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    rows: Value,
    cols: Value,
    position: Value,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &SUBPLOT_DESCRIPTOR,
        &SUBPLOT_ERROR_TOO_MANY_OUTPUTS,
        "subplot",
    )?;
    if [&rows, &cols, &position]
        .into_iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SUBPLOT_INTEGER_GRID_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let m = scalar_from_value(&rows, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let n = scalar_from_value(&cols, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let region = parse_subplot_region(m, n, &position)?;
    let property = super::state::PropertyMutationTarget::alias().with_runtime(runtime);
    let axes = super::state::select_subplot_region_for_property(&property, region)
        .map_err(map_subplot_figure_error)?;
    Ok(encode_axes_handle(axes.handle, axes.active_index))
}

fn parse_subplot_region(
    rows: usize,
    columns: usize,
    position: &Value,
) -> crate::BuiltinResult<AxesGridCell> {
    let invalid = || subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT);
    let rows = u32::try_from(rows).map_err(|_| invalid())?;
    let columns = u32::try_from(columns).map_err(|_| invalid())?;
    let count = rows
        .checked_mul(columns)
        .filter(|count| *count > 0)
        .ok_or_else(invalid)?;
    let mut selected: Option<AxesGridCell> = None;
    let mut include = |value: &Value| -> crate::BuiltinResult<()> {
        let index = scalar_from_value(value, BUILTIN_NAME).map_err(|_| invalid())?;
        let slot = u32::try_from(index - 1)
            .ok()
            .filter(|slot| *slot < count)
            .ok_or_else(|| subplot_error(&SUBPLOT_ERROR_INDEX_OUT_OF_RANGE))?;
        selected = Some(match selected {
            Some(region) => region.including_slot(slot).ok_or_else(invalid)?,
            None => AxesGridCell::single(rows, columns, slot),
        });
        Ok(())
    };
    if let Value::Tensor(tensor) = position {
        if crate::builtins::common::shape::effective_rank(tensor.shape()) > 2
            || tensor.shape().iter().filter(|size| **size > 1).count() > 1
        {
            return Err(invalid());
        }
        for index in 0..tensor.len() {
            let value = match tensor.numeric_value_at(index).ok_or_else(invalid)? {
                NumericScalar::F64(value) => Value::Num(value),
                NumericScalar::F32(value) => Value::Num(f64::from(value)),
                integer => Value::Int(integer.into_int_value().ok_or_else(invalid)?),
            };
            include(&value)?;
        }
    } else {
        include(position)?;
    }
    selected.ok_or_else(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::IntValue;

    fn select(rows: f64, columns: f64, positions: Value) -> super::super::state::FigureAxesState {
        subplot_builtin(Value::Num(rows), Value::Num(columns), positions).unwrap();
        super::super::state::current_axes_state()
    }

    fn active_line() -> f64 {
        use super::super::state;
        let axes = state::current_axes_state();
        state::append_active_plot(
            "subplot-test",
            state::PlotRenderOptions::default(),
            |figure, index| {
                figure.add_line_plot_on_axes(
                    runmat_plot::plots::LinePlot::new(vec![0., 1.], vec![1., 2.]).unwrap(),
                    index,
                );
                Ok(())
            },
        )
        .unwrap();
        let index = state::clone_figure(axes.handle)
            .unwrap()
            .plot_axes_indices()
            .len()
            - 1;
        state::register_line_handle(axes.handle, axes.active_index, index)
    }

    #[test]
    fn spans_and_mixed_grids_preserve_existing_layout_and_receivers() {
        use super::super::state;
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        for mixed in [false, true] {
            state::reset_plot_state();
            let positions = if mixed {
                Value::Num(1.)
            } else {
                Value::Tensor(runmat_value::Tensor::new(vec![1., 2., 4., 5.], vec![1, 4]).unwrap())
            };
            let (rows, columns) = if mixed { (1., 2.) } else { (2., 3.) };
            let left = select(rows, columns, positions.clone());
            let child = active_line();
            let receiver = state::axes_graphics_receiver(left.handle, left.active_index)
                .unwrap()
                .unwrap();
            let original = state::clone_figure(left.handle)
                .unwrap()
                .resolved_axes_grid_cell(left.active_index)
                .unwrap();
            let upper = select(
                2.,
                if mixed { 2. } else { 3. },
                Value::Num(if mixed { 2. } else { 3. }),
            );
            let lower = select(
                2.,
                if mixed { 2. } else { 3. },
                Value::Num(if mixed { 4. } else { 6. }),
            );
            assert_ne!(left.active_index, upper.active_index);
            assert_ne!(left.active_index, lower.active_index);
            assert_ne!(upper.active_index, lower.active_index);
            let figure = state::clone_figure(left.handle).unwrap();
            assert_eq!(
                figure.resolved_axes_grid_cell(left.active_index),
                Some(original)
            );
            assert_eq!(
                figure
                    .axes_metadata
                    .iter()
                    .filter(|axes| !axes.retired)
                    .count(),
                3
            );
            assert!(state::graphics_receiver_valid(&receiver));
            assert!(state::plot_child_handle_snapshot(child).is_ok());
            assert_eq!(
                select(rows, columns, positions).active_index,
                left.active_index
            );
            assert!(state::graphics_receiver_valid(&receiver));
            assert!(state::plot_child_handle_snapshot(child).is_ok());
        }
    }

    #[test]
    fn overlapping_subplot_retires_exact_handles_and_reuses_slots_without_rebasing_survivors() {
        use super::super::state;
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        state::reset_plot_state();
        let upper = select(2., 2., Value::Num(1.));
        let upper_receiver = state::axes_graphics_receiver(upper.handle, upper.active_index)
            .unwrap()
            .unwrap();
        let upper_child = active_line();
        let lower = select(2., 2., Value::Num(3.));
        let lower_receiver = state::axes_graphics_receiver(lower.handle, lower.active_index)
            .unwrap()
            .unwrap();
        let lower_child = active_line();
        let right = select(1., 2., Value::Num(2.));
        let right_receiver = state::axes_graphics_receiver(right.handle, right.active_index)
            .unwrap()
            .unwrap();
        let right_child = active_line();
        let merged = select(1., 2., Value::Num(1.));
        assert_eq!(merged.active_index, upper.active_index);
        assert!(!state::graphics_receiver_valid(&upper_receiver));
        assert!(!state::graphics_receiver_valid(&lower_receiver));
        assert!(state::plot_child_handle_snapshot(upper_child).is_err());
        assert!(state::plot_child_handle_snapshot(lower_child).is_err());
        assert!(state::graphics_receiver_valid(&right_receiver));
        assert_eq!(
            state::plot_child_handle_snapshot(right_child)
                .unwrap()
                .plot_index(),
            Some(0)
        );
        assert!(
            state::axes_graphics_receiver(lower.handle, lower.active_index)
                .unwrap()
                .is_none()
        );
        assert!(state::select_axes_for_figure(lower.handle, lower.active_index).is_err());
        assert!(!state::axes_handle_exists(lower.handle, lower.active_index));
        let handles = state::axes_handles_for_figure(merged.handle).unwrap();
        assert_eq!(handles.len(), 2);
        assert!(!handles.contains(&state::encode_axes_handle(lower.handle, lower.active_index)));
        state::reset_zoom_baseline_for_figure(merged.handle).unwrap();
        state::restore_zoom_baseline_for_figure(merged.handle).unwrap();
        assert!(!state::axes_handle_exists(lower.handle, lower.active_index));
        assert!(state::color_order_for_axes_for_property(
            &state::PropertyMutationTarget::alias(),
            lower.handle,
            lower.active_index,
        )
        .is_err());
        let figure = state::clone_figure(merged.handle).unwrap();
        let restored = runmat_plot::event::FigureScene::capture(&figure)
            .into_figure()
            .unwrap();
        assert!(!restored.axes_is_live(lower.active_index));
        assert!(restored.axes_is_live(right.active_index));
        select(2., 2., Value::Num(2.));
        let reused = select(2., 2., Value::Num(4.));
        assert_eq!(reused.active_index, lower.active_index);
        assert_eq!(state::clone_figure(reused.handle).unwrap().axes_count(), 3);
        assert!(!state::graphics_receiver_valid(&lower_receiver));
    }

    #[test]
    fn subplot_region_parser_keeps_exact_integer_slots_and_rejects_invalid_lists() {
        use runmat_value::{IntegerStorage, Tensor};
        let integer = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 4, 5]), vec![1, 4]).unwrap(),
        );
        assert_eq!(
            parse_subplot_region(2, 3, &integer).unwrap(),
            AxesGridCell::from_slots(2, 3, [0, 1, 3, 4]).unwrap()
        );
        for invalid in [
            Value::Tensor(Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
            Value::Tensor(Tensor::new(vec![1., 7.], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1., 1.5], vec![1, 2]).unwrap()),
            Value::Int(IntValue::U64(u64::MAX)),
        ] {
            assert!(parse_subplot_region(2, 3, &invalid).is_err());
        }
    }

    #[test]
    fn subplot_returns_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        assert!(matches!(props, Value::Struct(_)));
    }

    #[test]
    fn subplot_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUBPLOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ax = subplot(rows, cols, position)"));
    }

    #[test]
    fn subplot_typed_integer_grid_controls_are_explicitly_gated() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = subplot_builtin(
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(2)),
            Value::Int(IntValue::I32(2)),
        )
        .expect_err("strict mode rejects typed integer grid controls");
        assert_eq!(
            error.identifier(),
            SUBPLOT_INTEGER_GRID_EXTENSION.error_identifier
        );
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = subplot_builtin(
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(2)),
            Value::Int(IntValue::I32(2)),
        )
        .expect("extension mode accepts typed integer grid controls");
        assert!(handle.is_finite());
    }
}
