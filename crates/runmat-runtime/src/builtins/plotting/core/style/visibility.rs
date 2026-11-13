use super::*;
use crate::builtins::plotting::{state, tests::lock_plot_registry};
use futures::executor::block_on;
use std::sync::{Arc, Mutex};

#[test]
fn line_visibility_preserves_unset_and_explicit_overrides() {
    let opts = LineStyleParseOptions::generic("visibility-test");
    assert_eq!(
        parse_line_style_args(&[], &opts)
            .unwrap()
            .appearance
            .visible,
        None
    );
    for (value, expected) in [
        (Value::from("off"), false),
        (Value::Bool(true), true),
        (Value::Num(0.0), false),
        (Value::Int(runmat_value::IntValue::U64(u64::MAX)), true),
    ] {
        let parsed = parse_line_style_args(&[Value::from("Visible"), value], &opts).unwrap();
        assert_eq!(parsed.appearance.visible, Some(expected));
    }
    let parsed = parse_line_style_args(
        &[
            Value::from("Visible"),
            Value::from("off"),
            Value::from("Color"),
            Value::from("r"),
            Value::from("Visible"),
            Value::from("on"),
        ],
        &opts,
    )
    .unwrap();
    assert_eq!(parsed.appearance.visible, Some(true));
    assert!(
        parse_line_style_args(&[Value::from("Visible"), Value::from("invalid")], &opts).is_err()
    );
    assert!(parse_line_style_args(&[Value::from("Visible")], &opts).is_err());
    let mut options = LineStyleOptions {
        visible: Some(false),
        ..Default::default()
    };
    options.merge(LineStyleOptions::default());
    assert_eq!(options.resolve().visible, Some(false));
    options.merge(LineStyleOptions {
        visible: Some(true),
        ..Default::default()
    });
    assert_eq!(options.resolve().visible, Some(true));
}

fn vector(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).unwrap())
}

#[test]
fn visibility_reaches_2d_and_3d_host_and_typed_numeric_builders() {
    use crate::builtins::plotting::{plot, plot3};
    use runmat_plot::plots::NumericPlotData;
    for visible in [None, Some(false), Some(true)] {
        let appearance = LineAppearance {
            visible,
            ..Default::default()
        };
        let expected = visible.unwrap_or(true);
        let values = vec![1., 2.];
        assert_eq!(
            plot::build_line_plot_for_builtin(
                "test",
                values.clone(),
                values.clone(),
                "",
                &appearance
            )
            .unwrap()
            .visible,
            expected
        );
        assert_eq!(
            plot3::build_line3_plot_for_builtin(
                "test",
                values.clone(),
                values.clone(),
                values.clone(),
                "",
                &appearance
            )
            .unwrap()
            .visible,
            expected
        );
        let numeric = NumericPlotData::from_f64(values, vec![1, 2]).unwrap();
        assert_eq!(
            plot::build_line_plot_from_numeric_data_for_builtin(
                "test",
                numeric.clone(),
                numeric.clone(),
                "",
                &appearance
            )
            .unwrap()
            .visible,
            expected
        );
        assert_eq!(
            plot3::build_line3_plot_from_numeric_data_for_builtin(
                "test",
                numeric.clone(),
                numeric.clone(),
                numeric,
                "",
                &appearance
            )
            .unwrap()
            .visible,
            expected
        );
    }
}

fn visibility_cases() -> Vec<(&'static str, Vec<Value>)> {
    let x = vector(&[1., 2.]);
    let y = vector(&[3., 4.]);
    let z = vector(&[5., 6.]);
    vec![
        ("animatedline", vec![]),
        ("animatedline", vec![x.clone(), y.clone(), z.clone()]),
        ("plot", vec![x.clone(), y.clone()]),
        ("plot3", vec![x.clone(), y.clone(), z.clone()]),
        ("polarplot", vec![x.clone(), y.clone()]),
        ("stem", vec![x.clone(), y.clone()]),
        ("stairs", vec![x.clone(), y.clone()]),
        ("area", vec![x.clone(), y.clone()]),
        ("errorbar", vec![x.clone(), y.clone(), vector(&[0.1, 0.2])]),
        ("quiver", vec![x.clone(), y.clone(), x.clone(), y.clone()]),
        (
            "quiver3",
            vec![
                x.clone(),
                y.clone(),
                z.clone(),
                x.clone(),
                y.clone(),
                z.clone(),
            ],
        ),
        ("scatter", vec![x.clone(), y.clone()]),
        ("scatter3", vec![x.clone(), y.clone(), z]),
        ("refline", vec![Value::Num(1.), Value::Num(0.)]),
        // Existing identity-owned interceptors must keep their authority.
        ("line", vec![x, y]),
        ("xline", vec![Value::Num(1.)]),
        ("yline", vec![Value::Num(1.)]),
        (
            "triplot",
            vec![
                vector(&[1., 2., 3.]),
                vector(&[0., 1., 0.]),
                vector(&[0., 0., 1.]),
            ],
        ),
        (
            "stackedplot",
            vec![Value::Tensor(
                Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap(),
            )],
        ),
    ]
}

#[test]
fn shared_visibility_is_applied_before_first_publication_for_all_consumers() {
    let _lock = lock_plot_registry();
    state::disable_rendering_for_tests();
    for (name, data) in visibility_cases() {
        for requested in [None, Some(false), Some(true)] {
            state::reset_plot_state();
            let snapshots = Arc::new(Mutex::new(Vec::<Vec<bool>>::new()));
            let weak = Arc::downgrade(&snapshots);
            state::install_figure_observer(Arc::new(move |view| {
                if let (Some(snapshots), Some(figure)) = (weak.upgrade(), view.figure) {
                    let visible: Vec<_> = figure.plots().map(|plot| plot.is_visible()).collect();
                    if !visible.is_empty() {
                        snapshots.lock().unwrap().push(visible);
                    }
                }
            }))
            .unwrap();
            let mut args = data.clone();
            if let Some(visible) = requested {
                args.extend([
                    Value::from("Visible"),
                    Value::from(if visible { "on" } else { "off" }),
                ]);
            }
            block_on(crate::call_builtin_async(name, &args))
                .unwrap_or_else(|error| panic!("{name} {requested:?}: {error}"));
            let expected = requested.unwrap_or(true);
            let snapshots = snapshots.lock().unwrap();
            assert!(!snapshots.is_empty(), "{name} did not publish a plot");
            assert!(
                snapshots
                    .iter()
                    .flatten()
                    .all(|visible| *visible == expected),
                "{name} published incorrect visibility: {snapshots:?}"
            );
        }
    }
}

#[test]
fn shared_visibility_rejects_invalid_value_without_publishing_a_plot() {
    let _lock = lock_plot_registry();
    state::disable_rendering_for_tests();
    for (name, mut data) in visibility_cases() {
        state::reset_plot_state();
        data.extend([Value::from("Visible"), Value::from("invalid")]);
        assert!(
            block_on(crate::call_builtin_async(name, &data)).is_err(),
            "{name}"
        );
        if let Some(figure) = state::clone_figure(state::current_figure_handle()) {
            assert_eq!(
                figure.plots().count(),
                0,
                "{name} published before rejecting visibility"
            );
        }
    }
}

#[test]
fn shared_visibility_is_preserved_for_resident_coordinate_inputs() {
    use crate::builtins::common::{gpu_helpers, test_support};
    let _lock = lock_plot_registry();
    state::disable_rendering_for_tests();
    test_support::with_test_provider(|provider| {
        for (name, mut args) in visibility_cases().into_iter().filter(|(name, _)| {
            matches!(
                *name,
                "animatedline"
                    | "plot"
                    | "plot3"
                    | "polarplot"
                    | "stem"
                    | "stairs"
                    | "area"
                    | "errorbar"
                    | "quiver"
                    | "quiver3"
                    | "scatter"
                    | "scatter3"
            )
        }) {
            state::reset_plot_state();
            for value in &mut args {
                if let Value::Tensor(tensor) = value {
                    *value =
                        Value::GpuTensor(gpu_helpers::upload_tensor(provider, tensor).unwrap());
                }
            }
            args.extend([Value::from("Visible"), Value::from("off")]);
            block_on(crate::call_builtin_async(name, &args))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let figure = state::clone_figure(state::current_figure_handle()).unwrap();
            assert!(figure.plots().count() > 0, "{name}");
            assert!(figure.plots().all(|plot| !plot.is_visible()), "{name}");
        }
    });
}
