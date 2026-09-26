use super::*;

#[derive(Serialize)]
#[serde(rename = "AnalysisFieldValues", rename_all = "snake_case")]
enum LegacyValues<'a> {
    HostF64(&'a Vec<f64>),
    DeviceRef(&'a DeviceFieldRef),
}

#[derive(Serialize)]
#[serde(rename = "AnalysisField")]
struct LegacyField<'a> {
    field_id: &'a String,
    shape: &'a Vec<usize>,
    values: LegacyValues<'a>,
}

struct RepeatedScalar {
    count: usize,
    value: f64,
}
impl Serialize for RepeatedScalar {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(self.value, self.count))
    }
}

#[test]
fn borrowed_host_sequence_preserves_owned_and_legacy_wire_shape() {
    for (shape, count) in [(vec![0], 0), (vec![3], 3), (vec![2, 3], 6)] {
        let field = AnalysisField::host_f64("thermal.heat_flux.0", shape, vec![1.25; count]);
        let values = RepeatedScalar { count, value: 1.25 };
        let view = AnalysisField::host_f64_serialization(&field.field_id, &field.shape, &values);
        let AnalysisFieldValues::HostF64(ref owned) = field.values else {
            unreachable!()
        };
        let legacy = LegacyField {
            field_id: &field.field_id,
            shape: &field.shape,
            values: LegacyValues::HostF64(owned),
        };
        let bytes = serde_json::to_vec(&field).unwrap();
        assert_eq!(serde_json::to_vec(&view).unwrap(), bytes);
        assert_eq!(serde_json::to_vec(&legacy).unwrap(), bytes);
        assert_eq!(
            serde_json::to_value(&view).unwrap(),
            serde_json::json!({
                "field_id": field.field_id, "shape": field.shape, "values": { "host_f64": owned }
            })
        );
        assert_eq!(
            serde_json::from_slice::<AnalysisField>(&bytes).unwrap(),
            field
        );
        assert_eq!(
            serde_json::to_vec(&field.values).unwrap(),
            serde_json::to_vec(&legacy.values).unwrap()
        );
    }
}

#[test]
fn device_reference_retains_legacy_wire_shape_and_roundtrip() {
    let field = AnalysisField {
        field_id: "device.temperature".into(),
        shape: vec![5],
        values: AnalysisFieldValues::DeviceRef(DeviceFieldRef {
            backend: "gpu".into(),
            token: "resident-buffer".into(),
            element_count: 5,
        }),
    };
    let AnalysisFieldValues::DeviceRef(ref reference) = field.values else {
        unreachable!()
    };
    let legacy = LegacyField {
        field_id: &field.field_id,
        shape: &field.shape,
        values: LegacyValues::DeviceRef(reference),
    };
    let bytes = serde_json::to_vec(&field).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&legacy).unwrap());
    assert_eq!(
        serde_json::to_value(&field).unwrap(),
        serde_json::json!({
            "field_id": "device.temperature", "shape": [5],
            "values": { "device_ref": { "backend": "gpu", "token": "resident-buffer", "element_count": 5 } }
        })
    );
    assert_eq!(
        serde_json::from_slice::<AnalysisField>(&bytes).unwrap(),
        field
    );
    assert_eq!(
        serde_json::to_vec(&field.values).unwrap(),
        serde_json::to_vec(&legacy.values).unwrap()
    );
}
