use super::*;
use crate::study::output::{FieldLocation, ObservationQuantity};

fn global_scalar() -> AnalysisFieldLayout {
    AnalysisFieldLayout::scalar(ObservationQuantity::HeatFlux, FieldLocation::Global)
}

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
    layout: &'a AnalysisFieldLayout,
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
        let field = AnalysisField::host_f64(
            "thermal.heat_flux.0",
            global_scalar(),
            shape,
            vec![1.25; count],
        );
        let values = RepeatedScalar { count, value: 1.25 };
        let view = AnalysisField::host_f64_serialization(
            &field.field_id,
            &field.layout,
            &field.shape,
            &values,
        );
        let AnalysisFieldValues::HostF64(ref owned) = field.values else {
            unreachable!()
        };
        let legacy = LegacyField {
            field_id: &field.field_id,
            layout: &field.layout,
            shape: &field.shape,
            values: LegacyValues::HostF64(owned),
        };
        let bytes = serde_json::to_vec(&field).unwrap();
        assert_eq!(serde_json::to_vec(&view).unwrap(), bytes);
        assert_eq!(serde_json::to_vec(&legacy).unwrap(), bytes);
        assert_eq!(
            serde_json::to_value(&view).unwrap(),
            serde_json::json!({
                "field_id": field.field_id,
                "layout": { "quantity": "heat_flux", "location": "global", "value_shape": "scalar", "ordinal": { "kind": "single" } },
                "shape": field.shape, "values": { "host_f64": owned }
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
        layout: AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node),
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
        layout: &field.layout,
        shape: &field.shape,
        values: LegacyValues::DeviceRef(reference),
    };
    let bytes = serde_json::to_vec(&field).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&legacy).unwrap());
    assert_eq!(
        serde_json::to_value(&field).unwrap(),
        serde_json::json!({
            "field_id": "device.temperature",
            "layout": { "quantity": "temperature", "location": "node", "value_shape": "scalar", "ordinal": { "kind": "single" } },
            "shape": [5],
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

#[test]
fn layout_entity_counts_follow_declared_location_and_components() {
    let node_vector = AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node);
    assert_eq!(node_vector.entity_count(&[4, 3]), Ok(4));
    assert!(node_vector.entity_count(&[4]).is_err());
    let element_scalar = AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Element);
    assert_eq!(element_scalar.entity_count(&[2692]), Ok(2692));
    assert!(element_scalar.entity_count(&[2692, 3]).is_err());
    let points = AnalysisFieldLayout::symmetric_tensor3(
        ObservationQuantity::Stress,
        FieldLocation::IntegrationPoint,
    );
    assert_eq!(points.entity_count(&[28, 6]), Ok(28));
    assert!(points.entity_count(&[7, 4, 6]).is_err());
    let complex = AnalysisFieldLayout::scalar(ObservationQuantity::AcousticPressure, FieldLocation::Node)
        .representation(crate::study::output::ComplexRepresentation::Complex);
    assert_eq!(complex.entity_count(&[5, 2]), Ok(5));
    assert!(complex.entity_count(&[5]).is_err());
    let modes = AnalysisFieldLayout::scalar(ObservationQuantity::ModalFrequency, FieldLocation::Mode);
    assert_eq!(modes.entity_count(&[6]), Ok(6));
}

#[test]
fn layout_roundtrips_with_ordinal_and_representation() {
    let layout = AnalysisFieldLayout::scalar(ObservationQuantity::AcousticPressure, FieldLocation::Node)
        .representation(crate::study::output::ComplexRepresentation::Magnitude)
        .mode(2);
    let json = serde_json::to_value(&layout).unwrap();
    assert_eq!(json["ordinal"], serde_json::json!({ "kind": "mode", "index": 2 }));
    assert_eq!(json["representation"], "magnitude");
    assert_eq!(serde_json::from_value::<AnalysisFieldLayout>(json).unwrap(), layout);
}
