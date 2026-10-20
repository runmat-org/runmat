use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::set::set_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{
    clear_figure, clone_figure, configure_subplot, current_figure_handle, reset_hold_state_for_run,
};
use runmat_value::IntegerStorage;

#[test]
fn view_rejects_nonfinite_and_overflowed_angles_before_axes_creation() {
    use crate::builtins::plotting::state;
    let _guard = lock_plot_registry();
    state::reset_plot_state();
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        for args in [
            vec![Value::Num(invalid), Value::Num(0.)],
            vec![Value::Tensor(
                Tensor::new(vec![0., invalid], vec![1, 2]).unwrap(),
            )],
        ] {
            assert_eq!(
                view_builtin_source(args).unwrap_err().identifier(),
                Some("RunMat:view:InvalidArgument")
            );
            assert!(state::current_figure_handle_if_exists().is_none());
        }
    }
    assert!(view_builtin_source(vec![Value::Num(f32::MAX as f64), Value::Num(0.)]).is_ok());
}

#[test]
fn view_property_rejects_camera_overflow_without_changing_existing_state() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    let ax = crate::builtins::plotting::axes::axes_builtin(vec![]).unwrap();
    let values = |data| Value::Tensor(Tensor::new(data, vec![1, 2]).unwrap());
    set_builtin(vec![
        Value::Num(ax),
        Value::from("View"),
        values(vec![45., 20.]),
    ])
    .unwrap();
    for invalid in [f64::MAX, f64::NAN, f64::INFINITY] {
        assert!(set_builtin(vec![
            Value::Num(ax),
            Value::from("View"),
            values(vec![invalid, 10.])
        ])
        .is_err());
        assert_eq!(
            get_builtin(vec![Value::Num(ax), Value::from("View")]).unwrap(),
            values(vec![45., 20.])
        );
    }
}

#[test]
fn view_two_outputs_share_query_and_set_state_in_matlab_mode() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert_eq!(
        view_builtin_source(vec![Value::Num(45.), Value::Num(20.)]).unwrap(),
        Value::output_list(vec![Value::Num(45.), Value::Num(20.)])
    );
    assert_eq!(
        view_builtin_source(vec![]).unwrap(),
        Value::output_list(vec![Value::Num(45.), Value::Num(20.)])
    );
}

#[test]
fn view_output_and_numeric_admission_precedes_default_axes_creation() {
    use crate::builtins::plotting::state;
    let _guard = lock_plot_registry();
    state::reset_plot_state();
    {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        assert_eq!(
            view_builtin_source(vec![]).unwrap_err().identifier(),
            Some("RunMat:view:TooManyOutputs")
        );
        assert!(state::current_figure_handle_if_exists().is_none());
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        assert_eq!(
            view_builtin_source(vec![]).unwrap_err().identifier(),
            Some("RunMat:compatibility:ViewVectorOutputExtension")
        );
        assert!(state::current_figure_handle_if_exists().is_none());
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        assert!(view_builtin_source(vec![Value::Num(4.)]).is_err());
        assert!(state::current_figure_handle_if_exists().is_none());
        assert_eq!(
            view_builtin_source(vec![Value::Num(2.)]).unwrap(),
            Value::output_list(vec![])
        );
        assert!(state::current_figure_handle_if_exists().is_some());
    }
}

#[test]
fn view_sets_axes_local_angles() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    let value = view_builtin_source(vec![Value::Num(45.0), Value::Num(20.0)]).unwrap();
    let t = Tensor::try_from(&value).unwrap();
    assert_eq!(t.materialize_f64(), vec![45.0, 20.0]);
    let fig = clone_figure(current_figure_handle()).unwrap();
    let meta = fig.axes_metadata(0).unwrap();
    assert_eq!(meta.view_azimuth_deg, Some(45.0));
    assert_eq!(meta.view_elevation_deg, Some(20.0));
}

#[test]
fn view_is_subplot_local() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    configure_subplot(1, 2, 1).unwrap();
    let _ = view_builtin_source(vec![Value::Num(10.0), Value::Num(15.0)]).unwrap();
    let fig = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(fig.axes_metadata(0).unwrap().view_azimuth_deg, None);
    assert_eq!(fig.axes_metadata(1).unwrap().view_azimuth_deg, Some(10.0));
}

#[test]
fn view_supports_query_forms_and_presets() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let ax = Value::Num(crate::builtins::plotting::axes::axes_builtin(vec![]).unwrap());
    let _ = view_builtin_source(vec![ax.clone(), Value::Num(2.0)]).unwrap();
    let v = view_builtin_source(vec![ax.clone()]).unwrap();
    let t = Tensor::try_from(&v).unwrap();
    assert_eq!(t.materialize_f64(), vec![0.0, 90.0]);

    let _ = view_builtin_source(vec![ax.clone(), Value::Num(3.0)]).unwrap();
    let v = view_builtin_source(vec![ax]).unwrap();
    let t = Tensor::try_from(&v).unwrap();
    assert_eq!(t.materialize_f64(), vec![-37.5, 30.0]);
}

#[test]
fn view_handles_large_angles_and_property_roundtrip() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let ax_handle = crate::builtins::plotting::axes::axes_builtin(vec![]).unwrap();
    let (figure_handle, axes_index) =
        crate::builtins::plotting::state::decode_axes_handle(ax_handle).unwrap();
    let ax = Value::Num(ax_handle);
    let _ = view_builtin_source(vec![ax.clone(), Value::Num(405.0), Value::Num(89.0)]).unwrap();
    let queried = get_builtin(vec![ax.clone(), Value::String("View".into())]).unwrap();
    let t = Tensor::try_from(&queried).unwrap();
    assert_eq!(t.materialize_f64(), vec![405.0, 89.0]);

    set_builtin(vec![
        ax,
        Value::String("View".into()),
        Value::Tensor(Tensor::new(vec![180.0, -30.0], vec![1, 2]).expect("view vector")),
    ])
    .unwrap();
    let fig = clone_figure(figure_handle).unwrap();
    assert_eq!(
        fig.axes_metadata(axes_index).unwrap().view_azimuth_deg,
        Some(180.0)
    );
    assert_eq!(
        fig.axes_metadata(axes_index).unwrap().view_elevation_deg,
        Some(-30.0)
    );
}

#[test]
fn view_accepts_typed_integer_angle_vector() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![45, 20]), vec![1, 2]).expect("angles");
    let angles = Value::Tensor(tensor);
    let value = view_builtin_source(vec![angles]).unwrap();
    let t = Tensor::try_from(&value).unwrap();

    assert_eq!(t.materialize_f64(), vec![45.0, 20.0]);
}

#[test]
fn view_accepts_separate_typed_integer_angle_scalars() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let az = Tensor::new_integer(IntegerStorage::I16(vec![45]), vec![1, 1]).expect("azimuth");
    let el = Tensor::new_integer(IntegerStorage::I16(vec![20]), vec![1, 1]).expect("elevation");

    let value = view_builtin_source(vec![Value::Tensor(az), Value::Tensor(el)]).unwrap();
    let t = Tensor::try_from(&value).unwrap();

    assert_eq!(t.materialize_f64(), vec![45.0, 20.0]);
}

#[test]
fn view_integer_angles_are_gated_and_exact_at_camera_boundary() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let angles = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I32(vec![45, 20]), vec![1, 2]).expect("integer angles"),
    );
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = view_builtin_source(vec![angles.clone()]).expect_err("compatibility gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ViewIntegerAnglesExtension")
        );
    }
    {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lossy = Value::Int(IntValue::I32(16_777_217));
        let error = view_builtin_source(vec![lossy, Value::Int(IntValue::I8(0))])
            .expect_err("lossy binary32 angle");
        assert!(error.message().contains("exactly representable as single"));
    }
}

#[test]
fn view_descriptor_signatures_cover_query_and_set_forms() {
    let labels: Vec<&str> = runmat_builtins::VIEW_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"angles = view()"));
    assert!(labels.contains(&"angles = view(ax)"));
    assert!(labels.contains(&"angles = view([az el] | preset)"));
    assert!(labels.contains(&"angles = view(az, el)"));
    assert!(labels.contains(&"angles = view(ax, [az el] | preset)"));
    assert!(labels.contains(&"angles = view(ax, az, el)"));
}
