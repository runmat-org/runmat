use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::{map_figure_error, value_as_text_string};
use crate::builtins::plotting::properties::parse_text_style_pairs;
use crate::builtins::plotting::style::value_as_f64;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::builtins::plotting::{plotting_error, state::add_text_annotation_for_axes};

#[runtime_builtin(
    name = "text",
    category = "plotting",
    summary = "Add text annotation at a 2-D or 3-D plot position.",
    keywords = "text,annotation,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::text"
)]
pub fn text_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target, rest) = super::op_common::text::split_axes_target("text", &args)?;
    if rest.len() < 3 {
        return Err(plotting_error(
            "text",
            "text: expected text(x, y, label) or text(x, y, z, label)",
        ));
    }
    let x =
        value_as_f64(&rest[0]).ok_or_else(|| plotting_error("text", "text: x must be numeric"))?;
    let y =
        value_as_f64(&rest[1]).ok_or_else(|| plotting_error("text", "text: y must be numeric"))?;

    let (z, text_idx) = if let Some(text) = value_as_text_string(&rest[2]) {
        let style = parse_text_style_pairs("text", &rest[3..])?;
        return add_text_annotation_for_axes(
            target.0,
            target.1,
            glam::Vec3::new(x as f32, y as f32, 0.0),
            &text,
            style,
        )
        .map_err(|err| map_figure_error("text", err));
    } else {
        let z = value_as_f64(&rest[2])
            .ok_or_else(|| plotting_error("text", "text: z must be numeric for the 3-D form"))?;
        (z, 3usize)
    };

    if rest.len() <= text_idx {
        return Err(plotting_error("text", "text: expected annotation string"));
    }
    let text = value_as_text_string(&rest[text_idx])
        .ok_or_else(|| plotting_error("text", "text: label must be text"))?;
    let style = parse_text_style_pairs("text", &rest[text_idx + 1..])?;
    add_text_annotation_for_axes(
        target.0,
        target.1,
        glam::Vec3::new(x as f32, y as f32, z as f32),
        &text,
        style,
    )
    .map_err(|err| map_figure_error("text", err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::PlotTestLockGuard;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_builtins::Tensor;

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn text_creates_world_annotation_handle() {
        let _guard = setup();
        let handle = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::String("Hello".into()),
        ])
        .unwrap();
        let position =
            get_builtin(vec![Value::Num(handle), Value::String("Position".into())]).unwrap();
        let tensor = Tensor::try_from(&position).unwrap();
        assert_eq!(tensor.data, vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn text_supports_3d_form() {
        let _guard = setup();
        let handle = text_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("Hello".into()),
        ])
        .unwrap();
        let fig = crate::builtins::plotting::clone_figure(current_figure_handle()).unwrap();
        let annotation = fig.axes_text_annotations(0).first().unwrap();
        assert_eq!(annotation.text, "Hello");
        assert_eq!(annotation.position, glam::Vec3::new(1.0, 2.0, 3.0));
        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string, Value::String("Hello".into()));
    }

    #[test]
    fn text_annotations_clear_on_fresh_axes_replot() {
        let _guard = setup();
        let _ = text_builtin(vec![
            Value::Num(0.5),
            Value::Num(0.0),
            Value::String("midpoint".into()),
        ])
        .unwrap();
        futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
            Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 1, 2).unwrap()),
            Value::Tensor(Tensor::new_2d(vec![0.0, 1.0], 1, 2).unwrap()),
        ]))
        .unwrap();
        let fig = crate::builtins::plotting::clone_figure(current_figure_handle()).unwrap();
        assert!(fig.axes_text_annotations(0).is_empty());
    }
}
