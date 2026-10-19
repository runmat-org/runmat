use super::input::{numeric_data_from_value, numeric_selector_index};
use super::options::option_scalar;
use super::*;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{
    clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
};
use crate::builtins::table::table_from_columns;
use runmat_plot::plots::figure::PlotElement;
use runmat_value::IntegerStorage;

fn vec_tensor(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn first_unrepresentable_usize_double() -> f64 {
    if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    }
}

#[test]
fn binscatter_parses_typed_integer_options_exactly() {
    assert_eq!(
        parse_num_bins(&int_tensor(IntegerStorage::U16(vec![12, 8]), vec![1, 2])).unwrap(),
        [12, 8]
    );
    assert_eq!(
        parse_num_bins(&Value::Int(IntValue::U8(9))).unwrap(),
        [9, 9]
    );
    assert_eq!(
        numeric_selector_index(&int_tensor(IntegerStorage::I32(vec![2]), vec![1, 1])).unwrap(),
        Some(1)
    );
    assert!(option_bool(
        &int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
        "ShowEmptyBins"
    )
    .unwrap());
    assert_eq!(
        option_scalar(
            &int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]),
            "FaceAlpha",
        )
        .unwrap(),
        IntValue::U64(9_007_199_254_740_993).to_f64()
    );
}

#[test]
fn binscatter_structural_integer_parsers_ignore_all_typed_mirrors() {
    let storages = [
        IntegerStorage::I8(vec![4]),
        IntegerStorage::I16(vec![4]),
        IntegerStorage::I32(vec![4]),
        IntegerStorage::I64(vec![4]),
        IntegerStorage::U8(vec![4]),
        IntegerStorage::U16(vec![4]),
        IntegerStorage::U32(vec![4]),
        IntegerStorage::U64(vec![4]),
    ];

    for storage in storages {
        let value = int_tensor(storage, vec![1, 1]);
        assert_eq!(parse_num_bins(&value).unwrap(), [4, 4]);
        assert_eq!(numeric_selector_index(&value).unwrap(), Some(3));
    }
}

#[test]
fn binscatter_preserves_every_documented_integer_data_class() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let storages = [
        IntegerStorage::I8(vec![0, 1]),
        IntegerStorage::I16(vec![0, 1]),
        IntegerStorage::I32(vec![0, 1]),
        IntegerStorage::I64(vec![0, 1]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 1]),
        IntegerStorage::U32(vec![0, 1]),
        IntegerStorage::U64(vec![0, 1]),
    ];

    for storage in storages {
        let _ = clear_figure(None);
        let expected = storage.clone();
        let x = int_tensor(storage.clone(), vec![2, 1]);
        let y = int_tensor(storage, vec![2, 1]);
        let handle = futures::executor::block_on(binscatter_builtin(vec![
            x,
            y,
            int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
        ]))
        .expect("documented integer data");
        let resolved =
            crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
                .unwrap();
        for property in ["XData", "YData"] {
            let Value::Tensor(data) = crate::builtins::plotting::properties::get_properties(
                resolved.clone(),
                Some(property),
                NAME,
            )
            .unwrap() else {
                panic!("expected typed {property}");
            };
            assert_eq!(data.integer_storage(), Some(&expected));
            assert_eq!(data.shape(), &[2, 1]);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn binscatter_public_path_accepts_rank_zero_and_rank_one_vectors() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![1.0, 2.0])] {
        let _ = clear_figure(None);
        let x = Value::Tensor(Tensor::new(values.clone(), shape.clone()).unwrap());
        let y = Value::Tensor(Tensor::new(values, shape).unwrap());
        let handle = futures::executor::block_on(binscatter_builtin(vec![
            x,
            y,
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        ]))
        .expect("numeric vector geometry");
        assert!(handle.is_finite());
    }
}

#[test]
fn binscatter_wide_integer_binning_uses_exact_relative_coordinates() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    let base = 9_007_199_254_740_992_u64;
    let x = IntegerStorage::U64(vec![base + 1, base + 2]);
    let x_limits = IntegerStorage::U64(vec![base + 1, base + 3]);
    let handle = futures::executor::block_on(binscatter_builtin(vec![
        int_tensor(x.clone(), vec![2, 1]),
        int_tensor(IntegerStorage::U64(vec![0, 1]), vec![2, 1]),
        int_tensor(IntegerStorage::U8(vec![2, 2]), vec![1, 2]),
        Value::String("XLimits".into()),
        int_tensor(x_limits.clone(), vec![1, 2]),
        Value::String("YLimits".into()),
        int_tensor(IntegerStorage::U8(vec![0, 2]), vec![1, 2]),
    ]))
    .expect("wide integer binscatter");
    let resolved =
        crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
            .unwrap();
    let Value::Tensor(values) = crate::builtins::plotting::properties::get_properties(
        resolved.clone(),
        Some("Values"),
        NAME,
    )
    .unwrap() else {
        panic!("expected Values tensor");
    };
    assert_eq!(values.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
    let Value::Tensor(x_data) = crate::builtins::plotting::properties::get_properties(
        resolved.clone(),
        Some("XData"),
        NAME,
    )
    .unwrap() else {
        panic!("expected XData tensor");
    };
    assert_eq!(x_data.integer_storage(), Some(&x));
    let Value::Tensor(limits) =
        crate::builtins::plotting::properties::get_properties(resolved, Some("XLimits"), NAME)
            .unwrap()
    else {
        panic!("expected XLimits tensor");
    };
    assert_eq!(limits.integer_storage(), Some(&x_limits));

    let spanning_x = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 63) - 1, 1_u64 << 63]),
        vec![2, 1],
    )
    .unwrap();
    let spanning_limits =
        Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2]).unwrap();
    let y = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![2, 1]).unwrap();
    let y_limits = Tensor::new_integer(IntegerStorage::U8(vec![0, 2]), vec![1, 2]).unwrap();
    let chart = build_binscatter_chart(
        &spanning_x,
        &y,
        [2, 2],
        Some(&spanning_limits),
        Some(&y_limits),
        false,
        1.0,
        None,
        None,
    )
    .expect("full uint64 span");
    assert_eq!(chart.values.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn binscatter_rejects_matrix_data_and_gates_resident_input_before_gather() {
    let matrix = Value::Tensor(Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap());
    let err = numeric_data_from_value(matrix, "x").unwrap_err();
    assert!(err.message().contains("vector"), "{}", err.message());

    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let err =
        futures::executor::block_on(binscatter_builtin(vec![resident, vec_tensor(&[0.0, 1.0])]))
            .expect_err("MATLAB mode rejects resident input before provider access");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:BinscatterGpuInputExtension")
    );
}

#[test]
fn binscatter_runmat_gpu_gather_preserves_every_integer_class() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let storages = [
            IntegerStorage::I8(vec![0, 1]),
            IntegerStorage::I16(vec![0, 1]),
            IntegerStorage::I32(vec![0, 1]),
            IntegerStorage::I64(vec![0, 1]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 1]),
        ];
        for storage in storages {
            let _ = clear_figure(None);
            let source = Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap();
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
                .expect("typed integer upload");
            let result = {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                futures::executor::block_on(binscatter_builtin(vec![
                    Value::GpuTensor(handle.clone()),
                    vec_tensor(&[0.0, 1.0]),
                    Value::Num(2.0),
                ]))
                .expect("RunMat resident gather")
            };
            let resolved = crate::builtins::plotting::properties::resolve_plot_handle(
                &Value::Num(result),
                NAME,
            )
            .unwrap();
            let Value::Tensor(x_data) = crate::builtins::plotting::properties::get_properties(
                resolved,
                Some("XData"),
                NAME,
            )
            .unwrap() else {
                panic!("expected gathered XData");
            };
            assert_eq!(x_data.integer_storage(), Some(&storage));
            drop(handle);
        }
    });
}

#[test]
fn binscatter_rejects_invalid_typed_integer_options() {
    let err = parse_num_bins(&int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1])).unwrap_err();
    assert!(
        err.message().contains("positive integers"),
        "{}",
        err.message()
    );

    let err = parse_num_bins(&int_tensor(IntegerStorage::U16(vec![251]), vec![1, 1])).unwrap_err();
    assert!(
        err.message().contains("no greater than 250"),
        "{}",
        err.message()
    );

    let err = option_bool(
        &int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
        "ShowEmptyBins",
    )
    .unwrap_err();
    assert!(err.message().contains("0 or 1"), "{}", err.message());
}

#[test]
fn binscatter_rejects_unrepresentable_double_indices_before_casting() {
    let err =
        numeric_selector_index(&Value::Num(first_unrepresentable_usize_double())).unwrap_err();
    assert!(
        err.message().contains("platform limits"),
        "{}",
        err.message()
    );

    let err = parse_num_bins(&vec_tensor(&[first_unrepresentable_usize_double()])).unwrap_err();
    assert!(
        err.message().contains("no greater than 250"),
        "{}",
        err.message()
    );
}

#[test]
fn binscatter_counts_and_registers_chart_state() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let handle = futures::executor::block_on(binscatter_builtin(vec![
        vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
        vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    let props = crate::builtins::plotting::properties::get_properties(
        crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
            .unwrap(),
        Some("Values"),
        NAME,
    )
    .unwrap();
    let Value::Tensor(values) = props else {
        panic!("expected Values tensor");
    };
    assert_eq!(values.shape(), &[2, 2]);
    assert_eq!(values.materialize_f64(), vec![2.0, 0.0, 0.0, 2.0]);

    let figure = clone_figure(current_figure_handle()).unwrap();
    let plot = figure.plots().next().unwrap();
    let PlotElement::Surface(surface) = plot else {
        panic!("expected surface plot");
    };
    assert!(surface.image_mode);
    assert!(surface.flatten_z);
    assert_eq!(surface.x_data, vec![0.25, 0.75]);
    assert_eq!(surface.y_data, vec![0.25, 0.75]);
}

#[test]
fn binscatter_accepts_limits_show_empty_bins_and_face_alpha() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let handle = futures::executor::block_on(binscatter_builtin(vec![
        vec_tensor(&[0.0, 1.0, 2.0]),
        vec_tensor(&[0.0, 1.0, 2.0]),
        Value::String("NumBins".into()),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
        Value::String("XLimits".into()),
        Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![1, 2]).unwrap()),
        Value::String("YLimits".into()),
        Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![1, 2]).unwrap()),
        Value::String("ShowEmptyBins".into()),
        Value::String("on".into()),
        Value::String("FaceAlpha".into()),
        Value::Num(0.5),
        Value::String("DisplayName".into()),
        Value::String("density".into()),
    ]))
    .unwrap();
    let face_alpha = crate::builtins::plotting::properties::get_properties(
        crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
            .unwrap(),
        Some("FaceAlpha"),
        NAME,
    )
    .unwrap();
    assert!(matches!(face_alpha, Value::Num(value) if (value - 0.5).abs() < 1.0e-12));
}

#[test]
fn binscatter_supports_table_variable_inputs() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);

    let table = table_from_columns(
        vec!["x".to_string(), "y".to_string()],
        vec![
            Tensor::new(vec![0.0, 0.2, 0.8, 1.0], vec![4, 1])
                .map(Value::Tensor)
                .unwrap(),
            Tensor::new(vec![0.0, 0.1, 0.9, 1.0], vec![4, 1])
                .map(Value::Tensor)
                .unwrap(),
        ],
    )
    .unwrap();
    let handle = futures::executor::block_on(binscatter_builtin(vec![
        table,
        Value::String("x".into()),
        Value::String("y".into()),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    let props = crate::builtins::plotting::properties::get_properties(
        crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), NAME)
            .unwrap(),
        Some("Values"),
        NAME,
    )
    .unwrap();
    let Value::Tensor(values) = props else {
        panic!("expected Values tensor");
    };
    assert_eq!(values.materialize_f64(), vec![2.0, 0.0, 0.0, 2.0]);
}
