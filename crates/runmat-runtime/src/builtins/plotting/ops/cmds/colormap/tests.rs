use super::super::test_support::setup;
use super::*;
use crate::builtins::plotting::clone_figure;
use crate::builtins::plotting::current_figure_handle;
use runmat_value::{IntegerStorage, Tensor};
#[test]
fn colormap_accepts_rgb_matrix_lookup_tables() {
    let _guard = setup();
    colormap_builtin(vec![Value::Tensor(
        Tensor::new(vec![0.2, 0.8, 0.4, 0.1, 0.6, 0.0], vec![2, 3]).expect("RGB colormap"),
    )])
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

    colormap_builtin(vec![Value::Tensor(generated)]).unwrap();

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
    let returned = colormap_builtin(vec![Value::Tensor(uint8)]).expect("public uint8 map");
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
        let returned = colormap_builtin(vec![Value::Tensor(map)]).unwrap();
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
    colormap_builtin(vec![Value::Tensor(public)]).expect("uint8 is public");
    let extension = Tensor::new_integer(IntegerStorage::I64(vec![1, 0, 0]), vec![1, 3]).unwrap();
    let err = colormap_builtin(vec![Value::Tensor(extension)]).unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:ColormapNonUint8IntegerMapExtension")
    );
}

#[test]
fn colormap_rejects_typed_integer_targets_without_f64_aliasing() {
    let _guard = setup();
    let err = colormap_builtin(vec![
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        Value::String("parula".into()),
    ])
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:colormap:InvalidArgument"));
    assert!(err.to_string().contains("not a typed integer"));
}
