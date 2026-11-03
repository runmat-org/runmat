use super::*;
use crate::results::field::{
    canonical_field_partition_descriptors, canonical_field_partitions, reconstruct_recovered_field,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
};
use runmat_analysis_core::{
    AnalysisFieldValues, DeviceFieldRef, FieldLocation, ObservationQuantity, ObservationValueShape,
};

fn definition(shape: ObservationValueShape) -> RecoveredFieldDefinition {
    let quantity = match shape {
        ObservationValueShape::Scalar => ObservationQuantity::Temperature,
        ObservationValueShape::Vector3 => ObservationQuantity::HeatFlux,
        ObservationValueShape::SymmetricTensor3 => ObservationQuantity::Stress,
        _ => unreachable!(),
    };
    RecoveredFieldDefinition::new(
        "borrowed.multipart",
        quantity,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::VolumeElements,
        shape,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

#[test]
fn borrowed_multipart_content_matches_owned_identity_bytes_and_reconstruction() {
    for shape in [
        ObservationValueShape::Scalar,
        ObservationValueShape::Vector3,
        ObservationValueShape::SymmetricTensor3,
    ] {
        let definition = definition(shape);
        let count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3;
        let storage = AnalysisField::host_f64(
            definition.field_id(), definition.layout(),
            definition.storage_shape(count).unwrap(),
            (0..count * definition.storage_component_count().unwrap())
                .map(|i| i as f64 - 0.25)
                .collect(),
        );
        let sample = RecoveredFieldSample::TimeSeconds(0.5);
        let owned = RecoveredField::new(definition.clone(), sample, storage.clone()).unwrap();
        let view = RecoveredFieldView::new(&definition, sample, &storage).unwrap();
        assert_eq!(view.identity(), owned.identity());
        assert_eq!(
            RecoveredFieldView::from_field(&owned).unwrap().identity(),
            view.identity()
        );
        assert!(std::ptr::eq(view.storage(), &storage));
        assert!(std::ptr::eq(view.definition(), &definition));
        let mut iterator = view.canonical_partitions().unwrap();
        assert_eq!(iterator.len(), 2);
        let first = iterator.next().unwrap().unwrap();
        assert_eq!(iterator.len(), 1);
        let second = iterator.next().unwrap().unwrap();
        assert_eq!(iterator.len(), 0);
        assert!(iterator.next().is_none());
        assert!(iterator.next().is_none());
        let borrowed = vec![first, second];
        let collected = canonical_field_partitions(&owned).unwrap();
        assert_eq!(borrowed.len(), 2);
        assert_eq!(borrowed[1].source_indices.len(), 3);
        for (actual, expected) in borrowed.iter().zip(&collected) {
            assert_eq!(actual, expected);
            assert_eq!(
                actual.canonical_bytes().unwrap(),
                expected.canonical_bytes().unwrap()
            );
            let independently_selected =
                RecoveredFieldPartition::select(&owned, actual.source_indices.clone()).unwrap();
            assert_eq!(actual, &independently_selected);
        }
        assert_eq!(
            view.canonical_partition_descriptors().unwrap(),
            canonical_field_partition_descriptors(&owned).unwrap()
        );
        assert_eq!(
            reconstruct_recovered_field(
                view.identity(),
                definition.clone(),
                sample,
                borrowed,
                count
            )
            .unwrap(),
            owned
        );
    }
}

#[test]
fn borrowed_partition_boundary_rejects_invalid_storage_and_indices() {
    let definition = definition(ObservationValueShape::Scalar);
    let sample = RecoveredFieldSample::Final;
    let storage = AnalysisField::host_f64(definition.field_id(), definition.layout(), vec![3], vec![1.0, 2.0, 3.0]);
    let view = RecoveredFieldView::new(&definition, sample, &storage).unwrap();
    for indices in [vec![], vec![0, 0], vec![1, 0], vec![3], vec![usize::MAX]] {
        assert!(RecoveredFieldPartition::select_view(&view, indices).is_err());
    }
    let selected = RecoveredFieldPartition::select_view(&view, vec![0, 2]).unwrap();
    assert_eq!(selected.storage.as_host_f64().unwrap(), &[1.0, 3.0]);
    let mut malformed = storage.clone();
    malformed.shape = vec![usize::MAX, 2];
    assert!(RecoveredFieldView::new(&definition, sample, &malformed).is_err());
    malformed = storage.clone();
    malformed.field_id = "different".into();
    assert!(RecoveredFieldView::new(&definition, sample, &malformed).is_err());
    malformed = storage;
    malformed.values = AnalysisFieldValues::DeviceRef(DeviceFieldRef {
        backend: "device".into(),
        token: "resident".into(),
        element_count: 3,
    });
    let device = RecoveredFieldView::new(&definition, sample, &malformed).unwrap();
    assert!(device.canonical_partitions().is_err());
    assert!(RecoveredFieldPartition::select_view(&device, vec![0]).is_err());
}
