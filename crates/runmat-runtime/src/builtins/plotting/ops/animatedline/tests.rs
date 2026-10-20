use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
use futures::executor::block_on;
use runmat_builtins::{BuiltinIntegerOverloadKind, ANIMATEDLINE_INTEGER_CAPABILITIES};
use runmat_value::IntegerStorage;

#[path = "lifecycle.rs"]
mod lifecycle;
#[path = "resident_controls.rs"]
mod resident_controls;

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}

fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

#[test]
fn animatedline_descriptor_covers_core_forms() {
    let labels = ANIMATEDLINE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"an = animatedline()"));
    assert!(labels.contains(&"an = animatedline(x, y, z)"));
    assert!(labels.contains(&"an = animatedline(ax, ...)"));
    assert_eq!(ANIMATEDLINE_INTEGER_CAPABILITIES.len(), 3);
    assert_eq!(
        ANIMATEDLINE_INTEGER_CAPABILITIES[2].overload,
        BuiltinIntegerOverloadKind::StructuralParameter
    );
}

#[test]
fn animatedline_creates_empty_handle_with_default_limit() {
    let _guard = setup();
    let handle = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(matches!(handle, Value::Num(_)));
    let max = get_builtin(vec![
        handle.clone(),
        Value::String("MaximumNumPoints".into()),
    ])
    .unwrap();
    assert_eq!(max, Value::Num(DEFAULT_MAXIMUM_NUM_POINTS as f64));
    let ty = get_builtin(vec![handle, Value::String("Type".into())]).unwrap();
    assert_eq!(ty, Value::String("animatedline".into()));
}

#[test]
fn animatedline_accepts_axes_handle_without_trailing_args() {
    let _guard = setup();
    let axes = crate::builtins::plotting::axes::axes_builtin(Vec::new()).unwrap();
    let handle = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(axes)]),
    ))
    .unwrap();
    let parent = get_builtin(vec![handle, Value::String("Parent".into())]).unwrap();
    assert_eq!(parent, Value::Num(axes));
}

#[test]
fn animatedline_trims_initial_data_to_maximum() {
    let _guard = setup();
    let handle = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            vector(&[1.0, 2.0, 3.0]),
            vector(&[4.0, 5.0, 6.0]),
            Value::String("MaximumNumPoints".into()),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let x = get_builtin(vec![handle, Value::String("XData".into())]).unwrap();
    assert_eq!(tensor_data(x), vec![2.0, 3.0]);
}

#[test]
fn animatedline_rejects_unrepresentable_maximum_before_cast() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(block_on(maximum_from_value(Value::Num(boundary))).is_err());
    assert!(block_on(maximum_from_value(Value::Num(2.5))).is_err());
}

#[test]
fn animatedline_accepts_property_constructor_data() {
    let _guard = setup();
    let handle = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("XData".into()),
            vector(&[1.0, 2.0]),
            Value::String("YData".into()),
            vector(&[3.0, 4.0]),
            Value::String("ZData".into()),
            vector(&[5.0, 6.0]),
            Value::String("LineWidth".into()),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let z = get_builtin(vec![handle.clone(), Value::String("ZData".into())]).unwrap();
    assert_eq!(tensor_data(z), vec![5.0, 6.0]);
    let width = get_builtin(vec![handle, Value::String("LineWidth".into())]).unwrap();
    assert_eq!(width, Value::Num(2.0));
}

#[test]
fn animatedline_reads_typed_integer_constructor_data_exactly() {
    let _guard = setup();
    let handle = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            integer_vector(&[1, 2]),
            integer_vector(&[10, 20]),
            integer_vector(&[100, 200]),
        ]),
    ))
    .unwrap();
    assert_eq!(
        tensor_data(get_builtin(vec![handle.clone(), Value::String("XData".into())]).unwrap()),
        vec![1.0, 2.0]
    );
    assert_eq!(
        tensor_data(get_builtin(vec![handle.clone(), Value::String("YData".into())]).unwrap()),
        vec![10.0, 20.0]
    );
    assert_eq!(
        tensor_data(get_builtin(vec![handle, Value::String("ZData".into())]).unwrap()),
        vec![100.0, 200.0]
    );
}

#[test]
fn animatedline_accepts_all_integer_coordinate_classes_and_mixed_double() {
    let _guard = setup();
    for storage in all_integer_storages() {
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                integer_value(storage.clone()),
                vector(&[10.0, 20.0]),
                integer_value(storage),
            ]),
        ))
        .expect("all documented integer coordinates must construct");
        assert_eq!(
            tensor_data(get_builtin(vec![handle.clone(), Value::String("XData".into())]).unwrap()),
            vec![1.0, 2.0]
        );
        assert_eq!(
            tensor_data(get_builtin(vec![handle.clone(), Value::String("YData".into())]).unwrap()),
            vec![10.0, 20.0]
        );
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::String("ZData".into())]).unwrap()),
            vec![1.0, 2.0]
        );
    }
}

#[test]
fn animatedline_reads_all_integer_maximum_classes_exactly() {
    let _guard = setup();
    for storage in all_integer_scalar_storages(2) {
        let maximum = storage.value_at(0).expect("integer scalar");
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                vector(&[1.0, 2.0, 3.0]),
                vector(&[4.0, 5.0, 6.0]),
                Value::String("MaximumNumPoints".into()),
                Value::Int(maximum),
            ]),
        ))
        .expect("documented integer maximum must construct");
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::String("XData".into())]).unwrap()),
            vec![2.0, 3.0]
        );
    }
    if usize::BITS == 64 {
        let wide = 9_007_199_254_740_993_u64;
        assert_eq!(
            block_on(maximum_from_value(Value::Int(runmat_value::IntValue::U64(
                wide
            ))))
            .expect("wide exact maximum"),
            Some(wide as usize)
        );
        assert_eq!(
            block_on(maximum_from_value(Value::Int(runmat_value::IntValue::U64(
                u64::MAX
            ))))
            .expect("full-width exact maximum"),
            Some(usize::MAX)
        );
    }
    for invalid in [
        Value::Int(runmat_value::IntValue::I8(-1)),
        Value::Int(runmat_value::IntValue::U8(0)),
    ] {
        assert!(block_on(maximum_from_value(invalid)).is_err());
    }
}

#[test]
fn animatedline_set_accepts_all_integer_maximum_classes() {
    let _guard = setup();
    for storage in all_integer_scalar_storages(2) {
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                vector(&[1.0, 2.0, 3.0]),
                vector(&[4.0, 5.0, 6.0]),
            ]),
        ))
        .expect("animated line");
        let plot_handle = resolve_plot_handle(&handle, BUILTIN_NAME).expect("plot handle");
        set_properties(
            plot_handle,
            &[
                Value::String("MaximumNumPoints".into()),
                integer_value(storage),
            ],
            BUILTIN_NAME,
        )
        .expect("typed integer MaximumNumPoints property");
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::String("XData".into())]).unwrap()),
            vec![2.0, 3.0]
        );
    }
}

#[test]
fn animatedline_gathers_all_integer_coordinate_and_maximum_classes() {
    let _guard = setup();
    test_support::with_test_provider(|provider| {
        for (coordinate, maximum) in all_integer_storages()
            .into_iter()
            .zip(all_integer_scalar_storages(2))
        {
            let coordinate_tensor =
                Tensor::new_integer(coordinate, vec![1, 2]).expect("coordinate tensor");
            let maximum_tensor = Tensor::new_integer(maximum, vec![1, 1]).expect("maximum tensor");
            let coordinate_handle = gpu_helpers::upload_tensor(provider, &coordinate_tensor)
                .expect("coordinate upload");
            let maximum_handle =
                gpu_helpers::upload_tensor(provider, &maximum_tensor).expect("maximum upload");
            let handle = block_on(animatedline_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(coordinate_handle),
                    vector(&[10.0, 20.0]),
                    Value::String("MaximumNumPoints".into()),
                    Value::GpuTensor(maximum_handle),
                ]),
            ))
            .expect("resident documented integer inputs");
            assert_eq!(
                tensor_data(get_builtin(vec![handle, Value::String("XData".into())]).unwrap()),
                vec![1.0, 2.0]
            );
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn animatedline_wgpu_gathers_all_integer_coordinate_and_maximum_classes() {
    check_animatedline_wgpu_integer_gathers(false);
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a compatible WGPU device; run explicitly for hardware qualification"]
fn animatedline_required_wgpu_gathers_all_integer_coordinate_and_maximum_classes() {
    check_animatedline_wgpu_integer_gathers(true);
}

#[cfg(feature = "wgpu")]
fn check_animatedline_wgpu_integer_gathers(require_device: bool) {
    let _plot_guard = setup();
    let _accel_guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        assert!(!require_device, "required WGPU device is unavailable");
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    for (coordinate, maximum) in all_integer_storages()
        .into_iter()
        .zip(all_integer_scalar_storages(2))
    {
        let coordinate_tensor =
            Tensor::new_integer(coordinate, vec![1, 2]).expect("coordinate tensor");
        let maximum_tensor = Tensor::new_integer(maximum, vec![1, 1]).expect("maximum tensor");
        let coordinate_handle =
            gpu_helpers::upload_tensor(provider, &coordinate_tensor).expect("coordinate upload");
        let maximum_handle =
            gpu_helpers::upload_tensor(provider, &maximum_tensor).expect("maximum upload");
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(coordinate_handle),
                vector(&[10.0, 20.0]),
                Value::String("MaximumNumPoints".into()),
                Value::GpuTensor(maximum_handle),
            ]),
        ))
        .expect("resident WGPU integer inputs");
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::String("XData".into())]).unwrap()),
            vec![1.0, 2.0]
        );
    }
}

#[test]
fn animatedline_rejects_undocumented_logical_coordinates() {
    let _guard = setup();
    let error = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("XData".into()),
            Value::Bool(true),
            Value::String("YData".into()),
            Value::Num(2.0),
        ]),
    ))
    .expect_err("logical coordinates are not documented");
    assert!(error.message().contains("must not be logical"));
}

#[test]
fn animatedline_rejects_3d_marker_constructor() {
    let _guard = setup();
    let err = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            vector(&[1.0]),
            vector(&[2.0]),
            vector(&[3.0]),
            Value::String("Marker".into()),
            Value::String("o".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("marker"));
}

fn vector(values: &[f64]) -> Value {
    Value::Tensor(
        Tensor::new(values.to_vec(), vec![1, values.len()]).expect("animatedline row vector"),
    )
}

fn integer_vector(values: &[i16]) -> Value {
    let tensor = Tensor::new_integer(
        runmat_value::IntegerStorage::I16(values.to_vec()),
        vec![1, values.len()],
    )
    .unwrap();
    Value::Tensor(tensor)
}

fn integer_value(storage: IntegerStorage) -> Value {
    let len = storage.len();
    Value::Tensor(Tensor::new_integer(storage, vec![1, len]).expect("integer vector"))
}

fn all_integer_storages() -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![1, 2]),
        IntegerStorage::I16(vec![1, 2]),
        IntegerStorage::I32(vec![1, 2]),
        IntegerStorage::I64(vec![1, 2]),
        IntegerStorage::U8(vec![1, 2]),
        IntegerStorage::U16(vec![1, 2]),
        IntegerStorage::U32(vec![1, 2]),
        IntegerStorage::U64(vec![1, 2]),
    ]
}

fn all_integer_scalar_storages(value: u8) -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![value as i8]),
        IntegerStorage::I16(vec![value as i16]),
        IntegerStorage::I32(vec![value as i32]),
        IntegerStorage::I64(vec![value as i64]),
        IntegerStorage::U8(vec![value]),
        IntegerStorage::U16(vec![value as u16]),
        IntegerStorage::U32(vec![value as u32]),
        IntegerStorage::U64(vec![value as u64]),
    ]
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        other => panic!("expected tensor, got {other:?}"),
    }
}
