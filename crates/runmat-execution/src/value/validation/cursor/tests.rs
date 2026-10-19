use super::*;
use crate::value::{DurableDynamicProperty, DurableStructArrayField};
use runmat_types::{ClassIdentity, MemberAccess};

fn inline(value: InlineValue) -> ValuePayload {
    ValuePayload::Inline(Box::new(value))
}

#[test]
fn scalar_batch_is_incremental_and_shares_limits() {
    let values = (0..64)
        .map(|index| inline(InlineValue::I32(index)))
        .collect::<Vec<_>>();
    let mut cursor = ValuePayloadValidationCursor::new(
        &values,
        ValueLimits {
            max_inline_bytes: 255,
            ..ValueLimits::default()
        },
    );
    assert_eq!(
        cursor.advance(NonZeroUsize::new(7).unwrap()).unwrap(),
        ValueValidationProgress::Pending { completed_work: 7 }
    );
    let error = loop {
        match cursor.advance(NonZeroUsize::new(7).unwrap()) {
            Ok(ValueValidationProgress::Pending { completed_work }) => {
                assert_eq!(completed_work, 7)
            }
            Ok(ValueValidationProgress::Complete { .. }) => panic!("batch must exceed limit"),
            Err(error) => break error,
        }
    };
    assert!(matches!(
        error,
        ContractError::Limit {
            field: "inline value bytes",
            limit: 255
        }
    ));
}

#[test]
fn sparse_pointer_phase_precedes_row_and_data_failures_across_turns() {
    let payload = inline(InlineValue::Sparse(SparseValue {
        element_type: ElementType::F64,
        rows: 1,
        columns: 2,
        column_offsets: vec![0, 2, 1],
        row_indices: vec![7],
        little_endian_data: Vec::new(),
    }));
    let mut cursor =
        ValuePayloadValidationCursor::new(std::slice::from_ref(&payload), ValueLimits::default());
    let error = loop {
        match cursor.advance(NonZeroUsize::new(1).unwrap()) {
            Ok(ValueValidationProgress::Pending { completed_work: 1 }) => {}
            Ok(progress) => panic!("unexpected progress: {progress:?}"),
            Err(error) => break error,
        }
    };
    assert_eq!(
        error.to_string(),
        ContractError::invalid(
            "sparse payload",
            "column offsets are not canonical CSC offsets"
        )
        .to_string()
    );
}

#[test]
fn synchronous_api_drains_the_same_deep_cursor() {
    let mut payload = inline(InlineValue::I32(1));
    for _ in 0..32 {
        payload = inline(InlineValue::Cell {
            shape: vec![1, 1],
            values: vec![payload],
        });
    }
    let limits = ValueLimits {
        max_depth: 31,
        ..ValueLimits::default()
    };
    let direct = payload.validate(limits).unwrap_err();
    let mut cursor = ValuePayloadValidationCursor::new(std::slice::from_ref(&payload), limits);
    let stepped = loop {
        match cursor.advance(NonZeroUsize::new(3).unwrap()) {
            Ok(ValueValidationProgress::Pending { .. }) => {}
            Ok(ValueValidationProgress::Complete { .. }) => panic!("depth must fail"),
            Err(error) => break error,
        }
    };
    assert_eq!(direct.to_string(), stepped.to_string());
}

#[test]
fn durable_struct_array_shape_error_precedes_field_limit() {
    let payload = inline(InlineValue::DurableStructArray(DurableStructArray {
        shape: vec![u64::MAX, 2],
        fields: vec![DurableStructArrayField {
            name: "field".into(),
            values: Vec::new(),
        }],
    }));
    let error = payload
        .validate_for_durable_storage(ValueLimits {
            max_fields: 0,
            ..ValueLimits::default()
        })
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        ContractError::invalid("shape", "element count overflow").to_string()
    );
}

#[test]
fn durable_plain_object_property_error_precedes_dynamic_limit() {
    let class_identity = ClassIdentity::new("ExampleClass").unwrap();
    let dynamic_properties = {
        let dynamic = |name: &str| DurableDynamicProperty {
            name: name.into(),
            definition_name: name.into(),
            defining_class: class_identity.clone(),
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            dependent: false,
            hidden: false,
            transient: false,
            non_copyable: false,
            abort_set: false,
            set_observable: false,
            get_observable: false,
            description: String::new(),
        };
        vec![dynamic("first"), dynamic("second")]
    };
    let payload = inline(InlineValue::DurablePlainObject(DurablePlainObject {
        class_identity,
        properties: vec![StructField {
            name: "broken".into(),
            value: inline(InlineValue::Null),
        }],
        dynamic_properties,
    }));
    let error = payload
        .validate_for_durable_storage(ValueLimits {
            max_fields: 1,
            ..ValueLimits::default()
        })
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        ContractError::invalid(
            "durable storage value",
            "context-bound or unregistered value is not durable"
        )
        .to_string()
    );
}
