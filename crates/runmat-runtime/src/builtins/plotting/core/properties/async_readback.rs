//! Async plot-source materialization at the graphics property boundary.
//! Snapshotting releases the registry guard before a provider readback awaits.
use super::super::state::{plot_child_element_snapshot, PlotChildHandleState};
use super::*;
use runmat_plot::plots::figure::PlotElement;

async fn line_xy(
    line: &runmat_plot::plots::LinePlot,
    builtin: &'static str,
) -> BuiltinResult<PlotXY> {
    line.export_numeric_xy_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: line has no source data")))
}

async fn scatter_xy(
    scatter: &runmat_plot::plots::ScatterPlot,
    builtin: &'static str,
) -> BuiltinResult<PlotXY> {
    scatter
        .export_numeric_xy_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: scatter has no source data")))
}

async fn line3_xyz(
    line: &runmat_plot::plots::Line3Plot,
    builtin: &'static str,
) -> BuiltinResult<PlotXYZ> {
    line.export_numeric_xyz_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: plot3 has no source data")))
}

pub async fn get_properties_async(
    handle: PlotHandle,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return get_properties(handle, property, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return get_properties(handle, property, builtin);
    }
    let (child, plot, _) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    match (child, plot) {
        (PlotChildHandleState::Line(state), PlotElement::Line(line)) => {
            let (x, y) = line_xy(&line, builtin).await?;
            line_property_with_data(&state, property, builtin, line, x, y)
        }
        (PlotChildHandleState::Scatter(state), PlotElement::Scatter(scatter)) => {
            let (x, y) = scatter_xy(&scatter, builtin).await?;
            scatter_property_with_data(&state, property, builtin, scatter, x, y)
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line))
            if line3_property_needs_data(property) =>
        {
            let data = line3_xyz(&line, builtin).await?;
            line3_property_with_data(&state, property, builtin, line, Some(data))
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line)) => {
            line3_property_with_data(&state, property, builtin, line, None)
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => get_properties(handle, property, builtin),
    }
}

pub async fn set_properties_async(
    handle: PlotHandle,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return set_properties(handle, args, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return set_properties(handle, args, builtin);
    }
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    let mut pairs = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        pairs.push((property_name(&pair[0], builtin)?, &pair[1]));
    }
    let (child, plot, revision) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    let guard = PlotUpdateGuard {
        child_handle: *id,
        revision,
    };
    match (child, plot) {
        (PlotChildHandleState::Line(state), PlotElement::Line(line)) => {
            let data = line_xy(&line, builtin).await?;
            apply_line_properties_with_data(&state, &pairs, builtin, data, Some(guard))
        }
        (PlotChildHandleState::Scatter(state), PlotElement::Scatter(scatter)) => {
            let data = scatter_xy(&scatter, builtin).await?;
            apply_scatter_properties_with_data(&state, &pairs, builtin, scatter, data, Some(guard))
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line)) => {
            let data = line3_xyz(&line, builtin).await?;
            apply_line3_properties_with_data(&state, &pairs, builtin, data, Some(guard))
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => set_properties(handle, args, builtin),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::line::line_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use futures::executor::block_on;

    fn host_line() -> f64 {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("x"));
        let y = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).expect("y"));
        let Value::Num(handle) = block_on(line_builtin(vec![x, y])).expect("line") else {
            panic!("expected graphics handle");
        };
        handle
    }

    fn x_data(handle: f64) -> Value {
        let plot = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
        block_on(get_properties_async(plot, Some("XData"), "get")).expect("XData")
    }

    #[test]
    fn async_get_preserves_host_numeric_source_and_unpolled_set_has_no_effect() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let before = x_data(handle);
        let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let args = [
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("replacement")),
        ];
        let future = set_properties_async(plot, &args, "set");
        drop(future);
        assert_eq!(x_data(handle), before);
    }

    #[test]
    fn stale_snapshot_refuses_atomic_property_update() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let (child, _, revision) = plot_child_element_snapshot(handle).expect("snapshot");
        let PlotChildHandleState::Line(state) = child else {
            panic!("expected line child");
        };
        let figure = state.figure;
        let plot_index = state.plot_index;
        super::super::super::state::update_plot_element(figure, plot_index, |_| {})
            .expect("intervening mutation");
        let mut called = false;
        let error = super::super::super::state::update_plot_element_if_current(
            handle,
            figure,
            plot_index,
            revision,
            |_| called = true,
        )
        .expect_err("stale revision");
        assert!(matches!(
            error,
            super::super::super::state::FigureError::StalePlotSnapshot
        ));
        assert!(!called);
    }

    #[test]
    fn invalid_async_set_does_not_mutate_host_line() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let before = x_data(handle);
        let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let args = [
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0, 10.0], vec![1, 3]).expect("mismatch")),
        ];
        let error =
            block_on(set_properties_async(plot, &args, "set")).expect_err("mismatched lengths");
        assert!(error.message.contains("same number of elements"));
        assert_eq!(x_data(handle), before);
    }
}
