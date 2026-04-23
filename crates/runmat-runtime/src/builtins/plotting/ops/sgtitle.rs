use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::map_figure_error;
use crate::builtins::plotting::properties::parse_text_style_pairs;
use crate::builtins::plotting::style::value_as_f64;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::builtins::plotting::{
    plotting_error,
    state::{current_figure_handle, figure_handle_exists, set_sg_title_for_figure, FigureHandle},
};

#[runtime_builtin(
    name = "sgtitle",
    category = "plotting",
    summary = "Set a title centered above the entire figure.",
    keywords = "sgtitle,subplot,title,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::sgtitle"
)]
pub fn sgtitle_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target, rest) = split_figure_target("sgtitle", &args)?;
    if rest.is_empty() {
        return Err(plotting_error("sgtitle", "sgtitle: expected text input"));
    }
    let text = super::op_common::value_as_text_string(&rest[0])
        .or_else(|| format_num_as_title_text(&rest[0]))
        .ok_or_else(|| {
            plotting_error(
                "sgtitle",
                "sgtitle: expected text as char array, string, string array, or cell array of strings",
            )
        })?;
    let style = parse_text_style_pairs("sgtitle", &rest[1..])?;
    set_sg_title_for_figure(target, &text, style).map_err(|err| map_figure_error("sgtitle", err))
}

fn split_figure_target<'a>(
    builtin: &'static str,
    args: &'a [Value],
) -> crate::BuiltinResult<(FigureHandle, &'a [Value])> {
    // Only attempt handle detection when there is a subsequent argument to serve as title text.
    // Without this guard, `sgtitle(42)` would be misread as targeting figure 42 with no text,
    // making it impossible to use a numeric value as title text when that figure exists.
    if args.len() >= 2 {
        if let Some(first) = args.first() {
            if let Some(handle) = try_parse_figure_target(first) {
                if figure_handle_exists(handle) {
                    return Ok((handle, &args[1..]));
                }
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: invalid figure handle"),
                ));
            }
        }
    }
    Ok((current_figure_handle(), args))
}

/// Converts a finite scalar number to a title string, matching MATLAB's behaviour of
/// accepting numeric values wherever text is expected in annotation builtins.
fn format_num_as_title_text(value: &Value) -> Option<String> {
    let n = match value {
        Value::Num(n) => *n,
        _ => return None,
    };
    if !n.is_finite() {
        return None;
    }
    if n.fract() == 0.0 && n.abs() < 1e15 {
        Some(format!("{}", n as i64))
    } else {
        Some(format!("{n}"))
    }
}

fn try_parse_figure_target(value: &Value) -> Option<FigureHandle> {
    let scalar = value_as_f64(value)?;
    if !scalar.is_finite() || scalar <= 0.0 || scalar.fract().abs() > f64::EPSILON {
        return None;
    }
    Some(FigureHandle::from(scalar.round() as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::state::{decode_plot_object_handle, PlotObjectKind};
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, figure::figure_builtin,
        reset_hold_state_for_run,
    };

    fn figure_children_count(handle: f64) -> usize {
        let children =
            get_builtin(vec![Value::Num(handle), Value::String("Children".into())]).unwrap();
        match children {
            Value::Num(_) => 1,
            Value::Tensor(t) => t.data.len(),
            _ => 0,
        }
    }

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn sgtitle_returns_handle_and_updates_current_figure() {
        let _guard = setup();
        let handle = sgtitle_builtin(vec![Value::String("Overview".into())]).unwrap();
        let (figure, axes, kind) = decode_plot_object_handle(handle).unwrap();
        assert_eq!(figure, current_figure_handle());
        assert_eq!(axes, 0);
        assert_eq!(kind, PlotObjectKind::SuperTitle);

        let fig = clone_figure(figure).unwrap();
        assert_eq!(fig.sg_title.as_deref(), Some("Overview"));
    }

    #[test]
    fn sgtitle_accepts_explicit_figure_targets_and_properties() {
        let _guard = setup();
        let fig = figure_builtin(vec![Value::Num(321.0)]).unwrap();
        sgtitle_builtin(vec![
            Value::Num(fig),
            Value::String("Figure Overview".into()),
            Value::String("FontSize".into()),
            Value::Num(18.0),
            Value::String("FontWeight".into()),
            Value::String("bold".into()),
        ])
        .unwrap();

        let figure = clone_figure(FigureHandle::from(321)).unwrap();
        assert_eq!(figure.sg_title.as_deref(), Some("Figure Overview"));
        assert_eq!(figure.sg_title_style.font_size, Some(18.0));
        assert_eq!(figure.sg_title_style.font_weight.as_deref(), Some("bold"));
    }

    #[test]
    fn sgtitle_handle_supports_get() {
        let _guard = setup();
        let handle = sgtitle_builtin(vec![Value::String("Top".into())]).unwrap();
        let string = get_builtin(vec![Value::Num(handle), Value::String("String".into())]).unwrap();
        assert_eq!(string, Value::String("Top".into()));
        let ty = get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap();
        assert_eq!(ty, Value::String("text".into()));
    }

    #[test]
    fn sgtitle_rejects_invalid_inputs() {
        let _guard = setup();
        let err = sgtitle_builtin(vec![]).unwrap_err();
        assert!(err.message.contains("expected text input"));

        let err = sgtitle_builtin(vec![
            Value::String("Top".into()),
            Value::String("Bogus".into()),
            Value::Num(1.0),
        ])
        .unwrap_err();
        assert!(err.message.contains("unsupported property"));
    }

    #[test]
    fn sgtitle_numeric_title_when_figure_with_same_number_exists() {
        // Regression: sgtitle(42) must set the title to "42" on the current figure even when
        // figure 42 exists, rather than consuming 42 as a handle and erroring with "expected text".
        let _guard = setup();
        let _ = figure_builtin(vec![Value::Num(42.0)]).unwrap();
        let current = current_figure_handle();
        sgtitle_builtin(vec![Value::Num(42.0)]).unwrap();
        let fig = clone_figure(current).unwrap();
        assert_eq!(fig.sg_title.as_deref(), Some("42"));
    }

    #[test]
    fn sgtitle_numeric_title_when_no_figure_with_that_number() {
        // sgtitle(99) should set the title to "99" on the current figure when figure 99 does not
        // exist (previously errored with "invalid figure handle").
        let _guard = setup();
        let current = current_figure_handle();
        sgtitle_builtin(vec![Value::Num(99.0)]).unwrap();
        let fig = clone_figure(current).unwrap();
        assert_eq!(fig.sg_title.as_deref(), Some("99"));
    }

    #[test]
    fn sgtitle_float_numeric_title() {
        let _guard = setup();
        let current = current_figure_handle();
        let pi = std::f64::consts::PI;
        sgtitle_builtin(vec![Value::Num(pi)]).unwrap();
        let fig = clone_figure(current).unwrap();
        let expected = format!("{pi}");
        assert_eq!(fig.sg_title.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn sgtitle_explicit_handle_with_text_still_works() {
        // When there are 2+ args and the first is a valid figure handle, it is still treated as
        // the target figure — the pre-existing behaviour must be preserved.
        let _guard = setup();
        let _ = figure_builtin(vec![Value::Num(7.0)]).unwrap();
        sgtitle_builtin(vec![Value::Num(7.0), Value::String("Seven".into())]).unwrap();
        let fig = clone_figure(FigureHandle::from(7)).unwrap();
        assert_eq!(fig.sg_title.as_deref(), Some("Seven"));
    }

    #[test]
    fn figure_children_excludes_sg_title_handle_before_sgtitle_called() {
        // Before sgtitle is invoked the Children array must contain only the axes handles,
        // not a spurious super-title text object.
        let _guard = setup();
        let fig_handle = figure_builtin(vec![Value::Num(50.0)]).unwrap();
        let children_before = figure_children_count(fig_handle);
        // Default figure has exactly one axes.
        assert_eq!(
            children_before, 1,
            "expected 1 child (axes) before sgtitle, got {children_before}"
        );

        sgtitle_builtin(vec![
            Value::Num(fig_handle),
            Value::String("Now it exists".into()),
        ])
        .unwrap();
        let children_after = figure_children_count(fig_handle);
        assert_eq!(
            children_after, 2,
            "expected 2 children (axes + super-title) after sgtitle, got {children_after}"
        );
    }
}
