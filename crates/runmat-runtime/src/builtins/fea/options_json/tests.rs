use super::*;
use runmat_value::{CellArray, IntValue, IntegerStorage, StructArray, StructValue, Tensor};
use serde_json::json;

const NAME: &str = "fea.authorStudy";

fn record(id: u64) -> StructValue {
    let mut record = StructValue::new();
    record.insert("RegionId".into(), Value::Int(IntValue::U64(id)));
    record.insert("PressurePa".into(), Value::Num(2.5));
    record
}

#[test]
fn typed_record_arrays_follow_fea_column_major_sequence_layout() {
    let ids = [u64::MAX, (1 << 53) + 1, 7, 9];
    let records = ids.into_iter().map(record).collect::<Vec<_>>();
    let expected = serde_json::Value::Array(
        ids.into_iter()
            .map(|id| {
                json!({
                    "region_id": id, "pressure_pa": 2.5,
                })
            })
            .collect(),
    );
    for shape in [vec![4, 1], vec![1, 4], vec![2, 2], vec![2, 1, 2]] {
        let array = Value::StructArray(StructArray::new(records.clone(), shape.clone()).unwrap());
        assert_eq!(value_to_json(NAME, &array).unwrap(), expected);
        let cells = Value::Cell(
            CellArray::new_with_shape(
                records.iter().cloned().map(Value::Struct).collect(),
                shape.clone(),
            )
            .unwrap(),
        );
        assert_eq!(value_to_json(NAME, &cells).unwrap(), expected);
        let numbers =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U64(ids.to_vec()), shape).unwrap());
        assert_eq!(value_to_json(NAME, &numbers).unwrap(), json!(ids));
    }
}

#[test]
fn scalar_empty_and_explicit_single_cell_keep_distinct_json_cardinality() {
    let scalar = Value::Struct(record(u64::MAX));
    let object = json!({"region_id":u64::MAX, "pressure_pa":2.5});
    assert_eq!(value_to_json(NAME, &scalar).unwrap(), object);
    let cell = Value::Cell(CellArray::new(vec![scalar], 1, 1).unwrap());
    assert_eq!(value_to_json(NAME, &cell).unwrap(), json!([object]));
    for shape in [vec![0, 2], vec![3, 0, 2]] {
        let array = Value::StructArray(StructArray::empty(vec!["RegionId".into()], shape).unwrap());
        assert_eq!(value_to_json(NAME, &array).unwrap(), json!([]));
    }
}

#[test]
fn decoded_record_arrays_restore_exact_nested_integer_payloads() {
    let input = json!([
        {"region_id":u64::MAX, "nested":{"nodes":[u64::MAX, (1u64<<53)+1]}, "records":[{"index":u64::MAX}]},
        {"region_id":(1u64<<53)+1, "nested":{"nodes":[1,2]}, "records":[{"index":17}]}
    ]);
    let decoded =
        crate::builtins::fea::integer_serialization::value_from_json_preserving_integer_kinds(
            NAME,
            &ERROR_INPUT,
            &input,
        )
        .unwrap();
    assert!(matches!(&decoded, Value::StructArray(array) if array.shape() == [2,1]));
    // Singleton record sequences are canonically scalar language structures;
    // only their schema owner can restore the absent JSON array tag.
    let expected = json!([
        {"region_id":u64::MAX, "nested":{"nodes":[u64::MAX, (1u64<<53)+1]}, "records":{"index":u64::MAX}},
        {"region_id":(1u64<<53)+1, "nested":{"nodes":[1,2]}, "records":{"index":17}}
    ]);
    assert_eq!(value_to_json(NAME, &decoded).unwrap(), expected);
    let matrix = json!([[{"id":u64::MAX},{"id":2}], [{"id":(1u64<<53)+1},{"id":4}]]);
    let decoded =
        crate::builtins::fea::integer_serialization::value_from_json_preserving_integer_kinds(
            NAME,
            &ERROR_INPUT,
            &matrix,
        )
        .unwrap();
    assert_eq!(
        value_to_json(NAME, &decoded).unwrap(),
        json!([{"id":u64::MAX},{"id":(1u64<<53)+1},{"id":2},{"id":4}])
    );
}
