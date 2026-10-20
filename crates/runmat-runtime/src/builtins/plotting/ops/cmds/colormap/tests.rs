use super::super::test_support::setup;
use super::*;
use crate::builtins::plotting::clone_figure;
use crate::builtins::plotting::current_figure_handle;
use runmat_value::{IntegerStorage, Tensor};

fn map_length(value: Value) -> usize {
    let Value::Tensor(map) = value else {
        panic!("RGB matrix");
    };
    assert_eq!(map.shape()[1], 3);
    map.shape()[0]
}

#[test]
fn colormap_preserves_rank_one_and_trailing_singleton_rgb_representations() {
    let _guard = setup();
    for (shape, data, rows) in [
        (vec![3], vec![1., 0., 0.], 1),
        (vec![2, 3, 1], vec![1., 0., 0., 1., 0., 0.], 2),
    ] {
        assert_eq!(
            map_length(
                colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                    vec![Value::Tensor(Tensor::new(data, shape).unwrap())]
                ))
                .unwrap()
            ),
            rows
        );
    }
    assert!(
        colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(
                Tensor::new(vec![0.; 12], vec![2, 3, 2]).unwrap()
            )]
        ))
        .is_err()
    );
}

#[test]
fn colormap_strict_setter_sets_figure_default_and_returns_new_palette() {
    let _guard = setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    crate::builtins::plotting::state::reset_plot_state();
    let figure = current_axes_state().handle;
    let palette = colormap_tensor(runmat_plot::plots::surface::ColorMap::Hot, 7);
    let returned = colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(palette.clone())],
    ))
    .unwrap();
    assert_eq!(
        Tensor::try_from(&returned).unwrap().materialize_f64(),
        palette.materialize_f64()
    );
    crate::builtins::plotting::state::configure_subplot(1, 2, 1).unwrap();
    assert_eq!(colormap_length_for_axes(figure, 1), 7);
    let map = colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("jet".into())],
    ))
    .unwrap();
    assert_eq!(map_length(map), 7);
    assert_eq!(colormap_length_for_axes(figure, 0), 7);
    assert_eq!(
        map_length(
            colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![]
            ))
            .unwrap()
        ),
        7
    );
    assert_eq!(
        map_length(
            colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("default".into())]
            ))
            .unwrap()
        ),
        256
    );
}

#[test]
fn colormap_runmat_axes_local_and_named_explicit_target_lengths_preserve_selection() {
    let _guard = setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::plotting::state::reset_plot_state();
    let figure = current_axes_state().handle;
    crate::builtins::plotting::state::configure_subplot(1, 2, 0).unwrap();
    colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(colormap_tensor(
            runmat_plot::plots::surface::ColorMap::Hot,
            7,
        ))],
    ))
    .unwrap();
    crate::builtins::plotting::state::configure_subplot(1, 2, 1).unwrap();
    assert_eq!(
        map_length(
            colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![]
            ))
            .unwrap()
        ),
        256
    );
    let target = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
        figure, 0,
    ));
    assert_eq!(
        map_length(
            colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![target, Value::String("jet".into())]
            ))
            .unwrap()
        ),
        7
    );
    assert_eq!(current_axes_state().active_index, 1);
    assert_eq!(
        map_length(
            colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::Num(figure.as_u32() as f64)]
            ))
            .unwrap()
        ),
        256
    );
}

#[test]
fn colormap_invalid_named_map_does_not_create_figure() {
    let _guard = setup();
    crate::builtins::plotting::state::reset_plot_state();
    assert!(
        colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("not-a-map".into())]
        ))
        .is_err()
    );
    assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
}
#[test]
fn colormap_accepts_rgb_matrix_lookup_tables() {
    let _guard = setup();
    colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(
            Tensor::new(vec![0.2, 0.8, 0.4, 0.1, 0.6, 0.0], vec![2, 3]).expect("RGB colormap"),
        )],
    ))
    .unwrap();

    let figure = clone_figure(current_figure_handle()).expect("current figure");
    let meta = figure
        .axes_metadata(figure.active_axes_index)
        .expect("axes");
    let runmat_plot::plots::surface::ColorMap::Listed(colors) = &meta.colormap else {
        panic!("expected listed colormap");
    };
    assert_eq!(colors.as_ref(), &[[0.2, 0.4, 0.6], [0.8, 0.1, 0.0]]);
}

#[test]
fn colormap_preserves_generated_parula_rows_as_listed_matrix() {
    let _guard = setup();
    let generated = crate::builtins::plotting::colormap_arrays::colormap_tensor(
        runmat_plot::plots::surface::ColorMap::Parula,
        8,
    );

    colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(generated)],
    ))
    .unwrap();

    let figure = clone_figure(current_figure_handle()).expect("current figure");
    let meta = figure
        .axes_metadata(figure.active_axes_index)
        .expect("axes");
    let runmat_plot::plots::surface::ColorMap::Listed(colors) = &meta.colormap else {
        panic!("expected listed colormap");
    };
    assert_eq!(colors.len(), 8);
}

#[test]
fn colormap_integer_admission_covers_all_classes_and_uint8_scaling() {
    let _guard = setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let uint8 =
        Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 128]), vec![1, 3]).expect("uint8 map");
    let returned = colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(uint8)],
    ))
    .expect("public uint8 map");
    let values = Tensor::try_from(&returned).unwrap().materialize_f64();
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 0.0);
    assert!((values[2] - 128.0 / 255.0).abs() < 1.0e-6);

    let extensions = [
        IntegerStorage::I8(vec![1, 0, 1]),
        IntegerStorage::I16(vec![1, 0, 1]),
        IntegerStorage::I32(vec![1, 0, 1]),
        IntegerStorage::I64(vec![1, 0, 1]),
        IntegerStorage::U16(vec![1, 0, 1]),
        IntegerStorage::U32(vec![1, 0, 1]),
        IntegerStorage::U64(vec![1, 0, 1]),
    ];
    for storage in extensions {
        let map = Tensor::new_integer(storage, vec![1, 3]).unwrap();
        let returned = colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Tensor(map)],
        ))
        .unwrap();
        assert_eq!(
            Tensor::try_from(&returned).unwrap().materialize_f64(),
            vec![1.0, 0.0, 1.0]
        );
    }
}

#[test]
fn colormap_strict_mode_keeps_uint8_public_and_gates_other_integers() {
    let _guard = setup();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let public = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 3]).unwrap();
    colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(public)],
    ))
    .expect("uint8 is public");
    let extension = Tensor::new_integer(IntegerStorage::I64(vec![1, 0, 0]), vec![1, 3]).unwrap();
    let err = colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Tensor(extension)],
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:ColormapNonUint8IntegerMapExtension")
    );
}

#[test]
fn colormap_rejects_typed_integer_targets_without_f64_aliasing() {
    let _guard = setup();
    let err = colormap_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            Value::String("parula".into()),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:colormap:InvalidArgument"));
    assert!(err.to_string().contains("not a typed integer"));
}
