use super::super::test_support::setup;
use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::state::color_limits_snapshot;
use runmat_value::IntegerStorage;

#[test]
fn automatic_color_endpoints_use_scaled_color_data_not_spatial_coordinates() {
    let _guard = setup();
    let mut figure = runmat_plot::plots::Figure::new();
    figure.add_surface_plot(
        runmat_plot::plots::SurfacePlot::new(
            vec![100., 200.],
            vec![300.],
            vec![vec![20.], vec![40.]],
        )
        .unwrap()
        .with_image_mode(true)
        .with_flatten_z(true),
    );
    let handle = crate::builtins::plotting::state::import_figure(figure);
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(
                vec![1., 2., 3., 4., 5., 6., f64::NEG_INFINITY, f64::INFINITY],
                vec![1, 8],
            )
            .unwrap(),
        )],
    ))
    .unwrap();
    assert_eq!(
        axes_metadata_snapshot(handle, 0).unwrap().color_limits,
        Some((20., 40.))
    );
}

#[test]
fn unavailable_resident_color_extent_rejects_before_any_limit_mutation() {
    let _guard = setup();
    let mut figure = runmat_plot::plots::Figure::new();
    let mut surface =
        runmat_plot::plots::SurfacePlot::new(vec![1.], vec![1.], vec![vec![20.]]).unwrap();
    // Missing retained extent is explicit; resolving limits cannot trigger readback.
    surface.z_data = None;
    figure.add_surface_plot(surface);
    let handle = crate::builtins::plotting::state::import_figure(figure);
    let before = axes_metadata_snapshot(handle, 0).unwrap();
    assert!(
        axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(
                Tensor::new(
                    vec![1., 2., 3., 4., 5., 6., f64::NEG_INFINITY, f64::INFINITY],
                    vec![1, 8]
                )
                .unwrap()
            )]
        ))
        .is_err()
    );
    let after = axes_metadata_snapshot(handle, 0).unwrap();
    assert_eq!(after.x_limits, before.x_limits);
    assert_eq!(after.y_limits, before.y_limits);
    assert_eq!(after.z_limits, before.z_limits);
    assert_eq!(after.color_limits, before.color_limits);
}

#[test]
fn automatic_three_dimensional_queries_and_infinite_endpoints_share_z_bounds() {
    let _guard = setup();
    let mut figure = runmat_plot::plots::Figure::new();
    figure.add_line3_plot(
        runmat_plot::plots::Line3Plot::new(vec![1., 4.], vec![2., 8.], vec![20., 40.]).unwrap(),
    );
    let handle = crate::builtins::plotting::state::import_figure(figure);
    let query = || {
        Tensor::try_from(
            &axis_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![],
            ))
            .unwrap(),
        )
        .unwrap()
        .materialize_f64()
    };
    assert_eq!(query(), vec![1., 4., 2., 8., 20., 40.]);
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(
                vec![
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                ],
                vec![1, 6],
            )
            .unwrap(),
        )],
    ))
    .unwrap();
    assert_eq!(query(), vec![1., 4., 2., 8., 20., 40.]);
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("auto".into())],
    ))
    .unwrap();
    assert!(axes_metadata_snapshot(handle, 0)
        .unwrap()
        .z_limits
        .is_none());
    assert_eq!(query().len(), 6);
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("manual".into())],
    ))
    .unwrap();
    assert_eq!(
        axes_metadata_snapshot(handle, 0).unwrap().z_limits,
        Some((20., 40.))
    );
    assert_eq!(query(), vec![1., 4., 2., 8., 20., 40.]);
}

#[test]
fn explicit_z_limits_do_not_require_three_dimensional_content() {
    let _guard = setup();
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(vec![1., 4., 2., 8., 20., 40.], vec![1, 6]).unwrap(),
        )],
    ))
    .unwrap();
    let result = Tensor::try_from(
        &axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![],
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(result.materialize_f64(), vec![1., 4., 2., 8., 20., 40.]);
}
#[test]
fn invalid_later_limits_do_not_create_figure_or_mutate_earlier_pairs() {
    let _guard = setup();
    crate::builtins::plotting::state::reset_plot_state();
    let bad = || {
        Value::Tensor(
            Tensor::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 8.0], vec![1, 8]).unwrap(),
        )
    };
    assert!(
        axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![bad()]
        ))
        .is_err()
    );
    assert!(current_figure_handle_if_exists().is_none());
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], vec![1, 8]).unwrap(),
        )],
    ))
    .unwrap();
    let state = current_axes_state();
    let before = axes_metadata_snapshot(state.handle, state.active_index).unwrap();
    assert!(
        axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![bad()]
        ))
        .is_err()
    );
    let after = axes_metadata_snapshot(state.handle, state.active_index).unwrap();
    assert_eq!(before.x_limits, after.x_limits);
    assert_eq!(before.y_limits, after.y_limits);
    assert_eq!(before.z_limits, after.z_limits);
    assert_eq!(before.color_limits, after.color_limits);
}
#[test]
fn axis_accepts_six_element_3d_limits() {
    let _guard = setup();
    let ax = crate::builtins::plotting::subplot::subplot_builtin(
        Value::Num(1.0),
        Value::Num(1.0),
        Value::Num(1.0),
    )
    .unwrap();

    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 6])
                .expect("six-element axis limits"),
        )],
    ))
    .unwrap();
    let zlim = get_builtin(vec![Value::Num(ax), Value::String("ZLim".into())]).unwrap();
    let zlim = Tensor::try_from(&zlim).unwrap();
    assert_eq!(zlim.materialize_f64(), vec![4.0, 5.0]);
}

#[test]
fn axis_limits_read_typed_integer_storage_exactly() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();
    let limits = Tensor::new_integer(
        runmat_value::IntegerStorage::I16(vec![0, 10, -2, 2, 4, 5]),
        vec![1, 6],
    )
    .expect("typed limits");

    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(limits)],
    ))
    .unwrap();

    let xlim =
        Tensor::try_from(&get_builtin(vec![ax.clone(), Value::String("XLim".into())]).unwrap())
            .unwrap();
    let ylim =
        Tensor::try_from(&get_builtin(vec![ax.clone(), Value::String("YLim".into())]).unwrap())
            .unwrap();
    let zlim =
        Tensor::try_from(&get_builtin(vec![ax, Value::String("ZLim".into())]).unwrap()).unwrap();
    assert_eq!(xlim.materialize_f64(), vec![0.0, 10.0]);
    assert_eq!(ylim.materialize_f64(), vec![-2.0, 2.0]);
    assert_eq!(zlim.materialize_f64(), vec![4.0, 5.0]);
}

#[test]
fn axis_limits_accept_all_integer_classes_and_query_double_graphics_state() {
    let _guard = setup();
    let storages = [
        IntegerStorage::I8(vec![0, 10, 1, 2]),
        IntegerStorage::I16(vec![0, 10, 1, 2]),
        IntegerStorage::I32(vec![0, 10, 1, 2]),
        IntegerStorage::I64(vec![0, 10, 1, 2]),
        IntegerStorage::U8(vec![0, 10, 1, 2]),
        IntegerStorage::U16(vec![0, 10, 1, 2]),
        IntegerStorage::U32(vec![0, 10, 1, 2]),
        IntegerStorage::U64(vec![0, 10, 1, 2]),
    ];
    for storage in storages {
        let limits = Tensor::new_integer(storage, vec![1, 4]).expect("limits");
        assert_eq!(
            axis_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::Tensor(limits)]
            ))
            .expect("set limits"),
            Value::Bool(true)
        );
        let Value::Tensor(actual) = axis_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("query limits") else {
            panic!("expected limit tensor");
        };
        assert_eq!(actual.materialize_f64(), vec![0.0, 10.0, 1.0, 2.0]);
        assert_eq!(actual.numeric_dtype(), runmat_value::NumericDType::F64);
    }
}

#[test]
fn axis_supports_eight_limits_semiautomatic_endpoints_and_strict_ordering() {
    let _guard = setup();
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(
                vec![f64::NEG_INFINITY, 10.0, 0.0, f64::INFINITY],
                vec![1, 4],
            )
            .expect("semiautomatic limits"),
        )],
    ))
    .expect("semiautomatic axis");
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 2.0, 8.0], vec![1, 8])
                .expect("eight limits"),
        )],
    ))
    .expect("eight-element axis");
    assert_eq!(color_limits_snapshot(), Some((2.0, 8.0)));

    let equal = Tensor::new(vec![0.0, 0.0, 0.0, 1.0], vec![1, 4]).expect("equal");
    assert!(
        axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(equal)]
        ))
        .is_err()
    );
}

#[test]
fn axis_visibility_accepts_logical_and_all_integer_scalar_classes() {
    let _guard = setup();
    let state = current_axes_state();
    let zeros = [
        IntegerStorage::I8(vec![0]),
        IntegerStorage::I16(vec![0]),
        IntegerStorage::I32(vec![0]),
        IntegerStorage::I64(vec![0]),
        IntegerStorage::U8(vec![0]),
        IntegerStorage::U16(vec![0]),
        IntegerStorage::U32(vec![0]),
        IntegerStorage::U64(vec![0]),
    ];
    for storage in zeros {
        let scalar = Tensor::new_integer(storage, vec![1, 1]).expect("visibility");
        axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(scalar)],
        ))
        .expect("axis off");
        assert!(
            !axes_metadata_snapshot(state.handle, state.active_index)
                .expect("metadata")
                .axes_style
                .visible
        );
    }
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Bool(true)],
    ))
    .expect("axis on");
    assert!(
        axes_metadata_snapshot(state.handle, state.active_index)
            .expect("metadata")
            .axes_style
            .visible
    );
}

#[test]
fn axis_rejects_resident_limits_before_provider_access() {
    let _guard = setup();
    let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 4],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap()
        .handle;
    let error = axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::GpuTensor(resident)],
    ))
    .expect_err("resident graphics limits must reject");
    assert_eq!(error.identifier(), Some("RunMat:axis:InvalidArgument"));
}

#[test]
fn axis_accepts_common_command_modes() {
    let _guard = setup();
    for mode in [
        "equal", "auto", "tight", "image", "manual", "ij", "xy", "on", "off",
    ] {
        axis_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String(mode.into())],
        ))
        .unwrap_or_else(|err| panic!("axis {mode} should be accepted: {err:?}"));
    }
}

#[test]
fn axis_image_enables_equal_aspect_and_data_fitted_limits() {
    let _guard = setup();
    let ax = crate::builtins::plotting::gca::gca_builtin(vec![]).unwrap();

    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(vec![0.0, 10.0, -2.0, 2.0], vec![1, 4]).expect("four-element axis limits"),
        )],
    ))
    .unwrap();
    axis_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("image".into())],
    ))
    .unwrap();

    assert_eq!(
        get_builtin(vec![ax.clone(), Value::String("AxisEqual".into())]).unwrap(),
        Value::Bool(true)
    );
    let xlim = get_builtin(vec![ax.clone(), Value::String("XLim".into())]).unwrap();
    let ylim = get_builtin(vec![ax, Value::String("YLim".into())]).unwrap();
    let xlim = Tensor::try_from(&xlim).unwrap();
    let ylim = Tensor::try_from(&ylim).unwrap();
    assert!(xlim.materialize_f64().iter().all(|value| value.is_nan()));
    assert!(ylim.materialize_f64().iter().all(|value| value.is_nan()));
}
