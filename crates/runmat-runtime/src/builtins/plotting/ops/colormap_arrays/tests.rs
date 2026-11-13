use super::*;
use crate::builtins::plotting::figure::figure_builtin;
use crate::builtins::plotting::state::set_colormap_with_length;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

#[test]
fn named_colormaps_return_m_by_3_rgb_arrays() {
    for (name, builtin) in [
        (
            "parula",
            parula_builtin as fn(crate::call::BuiltinArguments) -> BuiltinResult<Value>,
        ),
        ("colorcube", colorcube_builtin),
        ("viridis", viridis_builtin),
        ("plasma", plasma_builtin),
        ("inferno", inferno_builtin),
        ("magma", magma_builtin),
        ("turbo", turbo_builtin),
        ("jet", jet_builtin),
        ("hot", hot_builtin),
        ("cool", cool_builtin),
        ("spring", spring_builtin),
        ("summer", summer_builtin),
        ("autumn", autumn_builtin),
        ("winter", winter_builtin),
        ("gray", gray_builtin),
        ("bone", bone_builtin),
        ("copper", copper_builtin),
        ("pink", pink_builtin),
        ("lines", lines_builtin),
    ] {
        let Value::Tensor(cmap) = builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(6.0)]),
        )
        .expect(name) else {
            panic!("expected tensor");
        };
        assert_eq!(
            (cmap.rows(), cmap.cols(), cmap.materialize_f64().len()),
            (6, 3, 18)
        );
        assert!(
            cmap.materialize_f64()
                .iter()
                .all(|v| (0.0..=1.0).contains(v)),
            "{name} produced out-of-range RGB entries"
        );
    }
}

#[test]
fn every_named_palette_preserves_zero_rows_double_class_and_typed_errors() {
    for (name, builtin) in [
        (
            "autumn",
            autumn_builtin as fn(crate::call::BuiltinArguments) -> BuiltinResult<Value>,
        ),
        ("bone", bone_builtin),
        ("colorcube", colorcube_builtin),
        ("cool", cool_builtin),
        ("copper", copper_builtin),
        ("gray", gray_builtin),
        ("hot", hot_builtin),
        ("inferno", inferno_builtin),
        ("jet", jet_builtin),
        ("lines", lines_builtin),
        ("magma", magma_builtin),
        ("parula", parula_builtin),
        ("pink", pink_builtin),
        ("plasma", plasma_builtin),
        ("spring", spring_builtin),
        ("summer", summer_builtin),
        ("turbo", turbo_builtin),
        ("viridis", viridis_builtin),
        ("winter", winter_builtin),
    ] {
        let Value::Tensor(empty) = builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        )
        .expect(name) else {
            panic!("{name} must return a numeric tensor");
        };
        assert_eq!(
            (empty.rows(), empty.cols(), empty.len()),
            (0, 3, 0),
            "{name}"
        );
        assert_eq!(empty.numeric_dtype(), runmat_value::NumericDType::F64);
        for bad in [
            vec![Value::Num(f64::INFINITY)],
            vec![Value::Num(-1.0)],
            vec![Value::Num(1.5)],
            vec![Value::Int(IntValue::U64(u64::MAX))],
            vec![Value::String("8".into())],
            vec![Value::Num(1.0), Value::Num(2.0)],
        ] {
            let error = builtin(crate::builtins::common::test_support::builtin_arguments(
                bad,
            ))
            .expect_err(name);
            assert_eq!(
                error.identifier(),
                PALETTE_INVALID_ARGUMENT.identifier,
                "{name}"
            );
        }
    }
}

#[test]
fn default_length_tracks_current_colormap_length() {
    let _guard = setup();
    let _ = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");
    set_colormap_with_length(ColorMap::ColorCube, 17);
    let Value::Tensor(cmap) = parula_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("parula default") else {
        panic!("expected tensor");
    };
    assert_eq!((cmap.rows(), cmap.cols()), (17, 3));
}

#[test]
fn length_argument_validates_nonnegative_integer() {
    let err = parula_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Num(1.5)],
    ))
    .expect_err("fractional length");
    assert!(err.message().contains("nonnegative integer"));
    let err = colorcube_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Num(-1.0)],
    ))
    .expect_err("negative length");
    assert!(err.message().contains("nonnegative integer"));
}

#[test]
fn length_argument_reads_typed_integer_tensor_exactly() {
    let length = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![MAX_PALETTE_LENGTH as u64]),
        vec![1, 1],
    )
    .expect("typed length");
    assert_eq!(
        colormap_length(&Value::Tensor(length), "parula").unwrap(),
        MAX_PALETTE_LENGTH
    );

    let too_large = runmat_value::Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![MAX_PALETTE_LENGTH as u64 + 1]),
        vec![1, 1],
    )
    .expect("too large");
    assert!(colormap_length(&Value::Tensor(too_large), "parula").is_err());

    let negative =
        runmat_value::Tensor::new_integer(runmat_value::IntegerStorage::I64(vec![-1]), vec![1, 1])
            .expect("negative");
    assert!(colormap_length(&Value::Tensor(negative), "parula").is_err());
}

#[test]
fn colorcube_accepts_all_integer_length_classes_and_returns_double() {
    let storages = [
        runmat_value::IntegerStorage::I8(vec![6]),
        runmat_value::IntegerStorage::I16(vec![6]),
        runmat_value::IntegerStorage::I32(vec![6]),
        runmat_value::IntegerStorage::I64(vec![6]),
        runmat_value::IntegerStorage::U8(vec![6]),
        runmat_value::IntegerStorage::U16(vec![6]),
        runmat_value::IntegerStorage::U32(vec![6]),
        runmat_value::IntegerStorage::U64(vec![6]),
    ];

    for storage in storages {
        let length = Tensor::new_integer(storage, vec![1, 1]).expect("length");
        let Value::Tensor(cmap) = colorcube_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(length)]),
        )
        .expect("colorcube") else {
            panic!("expected colormap tensor");
        };
        assert_eq!((cmap.rows(), cmap.cols()), (6, 3));
        assert_eq!(cmap.numeric_dtype(), runmat_value::NumericDType::F64);
    }
}

#[test]
fn parula_accepts_all_integer_length_classes_and_returns_double() {
    let storages = [
        runmat_value::IntegerStorage::I8(vec![6]),
        runmat_value::IntegerStorage::I16(vec![6]),
        runmat_value::IntegerStorage::I32(vec![6]),
        runmat_value::IntegerStorage::I64(vec![6]),
        runmat_value::IntegerStorage::U8(vec![6]),
        runmat_value::IntegerStorage::U16(vec![6]),
        runmat_value::IntegerStorage::U32(vec![6]),
        runmat_value::IntegerStorage::U64(vec![6]),
    ];

    for storage in storages {
        let length = Tensor::new_integer(storage, vec![1, 1]).expect("length");
        let Value::Tensor(cmap) = parula_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(length)]),
        )
        .expect("parula") else {
            panic!("expected colormap tensor");
        };
        assert_eq!((cmap.rows(), cmap.cols()), (6, 3));
        assert_eq!(cmap.numeric_dtype(), runmat_value::NumericDType::F64);
    }
}

#[test]
fn colorcube_rejects_resident_length_without_provider_dispatch() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let err = colorcube_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![resident],
    ))
    .expect_err("resident length");
    assert!(err.message().contains("numeric scalar"));
}

#[test]
fn parse_rgb_colormap_tensor_recognizes_generated_maps_and_custom_gradient() {
    let parula = colormap_tensor(ColorMap::Parula, 8);
    let (map, len) = parse_rgb_colormap_tensor(&parula, "colormap").expect("parse parula");
    assert!(matches!(map, ColorMap::Listed(_)));
    assert_eq!(len, 8);

    let custom =
        Tensor::new(vec![0.0, 1.0, 0.0, 0.5, 1.0, 0.0], vec![2, 3]).expect("custom colormap");
    let (map, len) = parse_rgb_colormap_tensor(&custom, "colormap").expect("parse custom");
    assert_eq!(len, 2);
    assert!(matches!(map, ColorMap::Listed(_)));

    let typed = Tensor::new_integer(
        runmat_value::IntegerStorage::U8(vec![0, 255, 0, 128, 255, 0]),
        vec![2, 3],
    )
    .expect("typed RGB colormap");
    let (map, len) = parse_rgb_colormap_tensor(&typed, "colormap").expect("parse typed");
    assert_eq!(len, 2);
    let ColorMap::Listed(rows) = map else {
        panic!("expected listed map");
    };
    assert_eq!(rows.as_ref()[0], [0.0, 0.0, 1.0]);
    assert_eq!(rows.as_ref()[1][0], 1.0);
    assert!((rows.as_ref()[1][1] - 128.0_f32 / 255.0_f32).abs() < f32::EPSILON);
    assert_eq!(rows.as_ref()[1][2], 0.0);
}
