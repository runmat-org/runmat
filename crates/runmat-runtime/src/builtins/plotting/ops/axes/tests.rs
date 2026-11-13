use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{current_figure_handle, reset_plot_state};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn axes_output_and_pair_syntax_admission_preserve_current_selection() {
    use crate::builtins::plotting::state;
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert_eq!(
            axes_builtin(vec![]).unwrap_err().identifier(),
            Some("RunMat:axes:TooManyOutputs")
        );
        assert!(state::current_figure_handle_if_exists().is_none());
    }
    let first = axes_builtin(vec![]).unwrap();
    axes_builtin(vec![]).unwrap();
    let before = state::current_axes_state();
    assert!(axes_builtin(vec![Value::Num(first), Value::from("Position")]).is_err());
    assert_eq!(
        state::current_axes_state().active_index,
        before.active_index
    );
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    axes_builtin(vec![Value::Num(first)]).unwrap();
    assert_ne!(
        state::current_axes_state().active_index,
        before.active_index
    );
}

#[test]
fn axes_records_shared_ordered_property_application_boundary() {
    use crate::builtins::plotting::state;
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    assert!(axes_builtin(vec![
        Value::from("FontSize"),
        Value::Num(16.),
        Value::from("UnsupportedAxesProperty"),
        Value::Num(1.)
    ])
    .is_err());
    // Existing property application is ordered, not transactional. A future
    // shared property-owner transaction must deliberately update this witness.
    let current = state::current_axes_state();
    let handle = state::encode_axes_handle(current.handle, current.active_index);
    assert_eq!(
        get_builtin(vec![Value::Num(handle), Value::from("FontSize")]).unwrap(),
        Value::Num(16.)
    );
}

#[test]
fn axes_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = AXES_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"ax = axes()"));
    assert!(labels.contains(&"ax = axes(target)"));
    assert!(labels.contains(&"ax = axes(property, value, ...)"));
}

#[test]
fn axes_creates_handle_and_sets_properties() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();

    let handle = axes_builtin(vec![
        Value::String("Position".into()),
        Value::Tensor(runmat_value::Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4]).unwrap()),
        Value::String("Units".into()),
        Value::String("normalized".into()),
    ])
    .expect("axes");
    let props = get_builtin(vec![Value::Num(handle)]).expect("get axes");
    assert!(matches!(props, Value::Struct(_)));
}

#[test]
fn axes_selects_existing_axes() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();

    let first = axes_builtin(Vec::new()).expect("first axes");
    let second = axes_builtin(Vec::new()).expect("second axes");
    assert_ne!(first, second);
    let selected = axes_builtin(vec![Value::Num(first)]).expect("select first");
    assert_eq!(selected, first);
}

#[test]
fn axes_creation_and_selection_publish_figure_changes() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();

    let first = axes_builtin(Vec::new()).expect("first axes");
    let handle = current_figure_handle();
    assert_eq!(
        crate::builtins::plotting::state::take_recent_figures(),
        vec![handle]
    );

    axes_builtin(Vec::new()).expect("second axes");
    assert_eq!(
        crate::builtins::plotting::state::take_recent_figures(),
        vec![handle]
    );

    axes_builtin(vec![Value::Num(first)]).expect("select first axes");
    assert_eq!(
        crate::builtins::plotting::state::take_recent_figures(),
        vec![handle]
    );
}

#[test]
fn positioned_axes_can_render_without_a_plot() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();

    axes_builtin(vec![
        Value::from("Position"),
        Value::Tensor(Tensor::new(vec![0.2, 0.2, 0.6, 0.6], vec![1, 4]).expect("position")),
    ])
    .expect("positioned axes");
    let figure = current_figure_handle();
    let image = futures::executor::block_on(crate::builtins::plotting::render_figure_snapshot(
        figure, 640, 360, None,
    ))
    .expect("render positioned axes");
    assert!(image.starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[test]
fn axes_parent_figure_selects_target_figure() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();

    let fig = crate::builtins::plotting::figure::figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(9.0)]),
    )
    .expect("figure");
    let ax =
        axes_builtin(vec![Value::String("Parent".into()), Value::Num(fig)]).expect("axes parent");
    assert!(ax > fig);
    assert_eq!(current_figure_handle().as_u32(), 9);
}

#[test]
fn axes_data_aspect_ratio_accepts_all_integer_classes_at_graphics_boundary() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let storages = [
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ];
    for storage in storages {
        let ratio = Tensor::new_integer(storage, vec![1, 3]).expect("ratio");
        let handle = axes_builtin(vec![
            Value::String("DataAspectRatio".into()),
            Value::Tensor(ratio),
        ])
        .expect("axes");
        let actual = get_builtin(vec![
            Value::Num(handle),
            Value::String("DataAspectRatio".into()),
        ])
        .expect("get ratio");
        let Value::Tensor(actual) = actual else {
            panic!("expected ratio tensor");
        };
        assert_eq!(actual.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }
}

#[test]
fn axes_rejects_resident_property_values_before_provider_access() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let error = axes_builtin(vec![Value::String("DataAspectRatio".into()), resident])
        .expect_err("resident property must reject");
    assert!(error.message().contains("cannot convert"));
}
