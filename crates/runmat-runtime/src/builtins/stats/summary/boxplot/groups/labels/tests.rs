use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::{CharArray, ObjectInstance, StringArray};
use runmat_value::{IntValue, IntegerStorage};
use std::num::NonZeroU64;
use std::rc::Rc;

#[test]
fn string_group_presence_uses_tags_not_literal_spellings_in_every_carrier() {
    use runmat_value::StringElement;
    for (element, present) in [
        (StringElement::Missing, false),
        (StringElement::text(""), false),
        (StringElement::text("  "), true),
        (StringElement::text("<missing>"), true),
        (StringElement::text("<undefined>"), true),
        (StringElement::text("missing"), true),
    ] {
        let scalar = Value::String(element.clone());
        let array = Value::StringArray(
            StringArray::from_elements_unbounded(vec![element.clone()], vec![1, 1]).unwrap(),
        );
        let cell = Value::Cell(CellArray::new(vec![Value::String(element.clone())], 1, 1).unwrap());
        for value in [scalar, array, cell] {
            let labels = group_labels(&value, 1).unwrap();
            assert_eq!(labels[0].is_some(), present, "{value:?}");
            if present {
                assert!(
                    matches!(labels[0].as_deref(), Some([KeyAtom::Text(text)]) if Some(text.as_str()) == element.as_str())
                );
            }
        }
    }
}

#[test]
fn scalar_group_label_reads_typed_integer_storage_without_double_mirror() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![42]), vec![1, 1]).unwrap();
    let Some(label) = scalar_group_label(&Value::Tensor(tensor)).unwrap() else {
        panic!("expected exact integer label");
    };
    assert!(matches!(
        label.as_slice(),
        [KeyAtom::Integer(IntValue::U16(42))]
    ));
}

#[test]
fn nested_factor_labels_have_identical_sync_and_cooperative_identity() {
    let first = Value::StringArray(
        runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
    );
    let second = Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![(1u64 << 53) + 1, (1u64 << 53) + 2]),
            vec![2, 1],
        )
        .unwrap(),
    );
    let group = Value::Cell(CellArray::new(vec![first, second], 1, 2).unwrap());
    let expected = group_labels(&group, 2).unwrap();
    let mut cooperation = super::super::super::cooperation::BoxplotCooperation::new(None);
    let actual =
        futures::executor::block_on(group_labels_cooperative(&group, 2, &mut cooperation)).unwrap();
    assert_eq!(actual, expected);
    assert!(matches!(
        actual[0].as_deref(),
        Some([KeyAtom::Text(text), KeyAtom::Integer(IntValue::U64(value))])
            if text == "a" && *value == (1u64 << 53) + 1
    ));
}

#[test]
fn scalar_cell_precedes_multifactor_and_late_fallback_preserves_error() {
    let scalar_cell = Value::Cell(
        CellArray::new(
            vec![
                Value::String("first".into()),
                Value::String("second".into()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let sync = group_labels(&scalar_cell, 2).unwrap();
    let mut cooperation = super::super::super::cooperation::BoxplotCooperation::new(None);
    let cooperative =
        futures::executor::block_on(group_labels_cooperative(&scalar_cell, 2, &mut cooperation))
            .unwrap();
    assert_eq!(cooperative, sync);
    assert!(matches!(sync[0].as_deref(), Some([KeyAtom::Text(text)]) if text == "first"));

    let late_failure = Value::Cell(
        CellArray::new(
            vec![
                Value::String("first".into()),
                Value::StringArray(
                    StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
                ),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let sync_error = group_labels(&late_failure, 2).unwrap_err();
    let cooperative_error =
        futures::executor::block_on(group_labels_cooperative(&late_failure, 2, &mut cooperation))
            .unwrap_err()
            .into_runtime();
    assert_eq!(cooperative_error.message, sync_error.message);
}

#[test]
fn long_single_text_label_yields_before_it_is_published() {
    let group = Value::String("💠".repeat(2048).into());
    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(256).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = super::super::super::cooperation::BoxplotCooperation::with_budget(budget);
    let error = futures::executor::block_on(group_labels_cooperative(&group, 1, &mut cooperation))
        .expect_err("a long label must yield before the finished label is published")
        .into_runtime();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn cooperative_label_decode_cancels_within_nested_cell_before_completion() {
    let group = Value::Cell(
        CellArray::new(
            vec![Value::StringArray(
                runmat_value::StringArray::new(
                    (0..128).map(|index| format!("group{index}")).collect(),
                    vec![128, 1],
                )
                .unwrap(),
            )],
            1,
            1,
        )
        .unwrap(),
    );
    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = super::super::super::cooperation::BoxplotCooperation::with_budget(budget);
    let error =
        futures::executor::block_on(group_labels_cooperative(&group, 128, &mut cooperation))
            .expect_err("cancelled recursive labels cannot publish a vector")
            .into_runtime();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn character_and_categorical_labels_keep_sync_async_policy_and_errors() {
    let chars =
        Value::CharArray(CharArray::from_unicode_scalars(vec!['A', '1', 'B', '2'], 2, 2).unwrap());
    let expected = group_labels(&chars, 2).unwrap();
    let mut cooperation = super::super::super::cooperation::BoxplotCooperation::new(None);
    let actual =
        futures::executor::block_on(group_labels_cooperative(&chars, 2, &mut cooperation)).unwrap();
    assert_eq!(actual, expected);
    assert!(matches!(actual[0].as_deref(), Some([KeyAtom::Text(text)]) if text == "A1"));
    assert!(matches!(actual[1].as_deref(), Some([KeyAtom::Text(text)]) if text == "B2"));

    let mut categorical = ObjectInstance::new(runmat_types::standard::CATEGORICAL.to_string());
    categorical.replace_property_unbounded(
        "Categories".into(),
        Value::StringArray(StringArray::new(vec!["one".into(), "two".into()], vec![2, 1]).unwrap()),
    );
    categorical.replace_property_unbounded(
        "Codes".into(),
        Value::Tensor(Tensor::new(vec![f64::NAN, 2.0, 3.0], vec![3, 1]).unwrap()),
    );
    let categorical = Value::Object(categorical);
    let sync = group_labels(&categorical, 2).unwrap_err();
    let async_error =
        futures::executor::block_on(group_labels_cooperative(&categorical, 2, &mut cooperation))
            .unwrap_err()
            .into_runtime();
    assert_eq!(sync.message, async_error.message);
    assert!(sync.source.is_some());
    assert!(async_error.source.is_some());
}
