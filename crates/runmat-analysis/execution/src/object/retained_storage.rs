//! Requested owned transport storage; excludes allocator overhead and process RSS.
use super::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution_artifact::LogicalObject;

pub(crate) fn logical_objects_storage(objects: &Vec<LogicalObject>) -> AnalysisBridgeResult<u64> {
    let slots = objects
        .capacity()
        .checked_mul(std::mem::size_of::<LogicalObject>())
        .ok_or_else(overflow)?;
    let mut bytes = u64::try_from(slots).map_err(|_| overflow())?;
    for object in objects {
        for count in [
            object.bytes.capacity(),
            object.descriptor.logical_name.capacity(),
            object.descriptor.media_type.capacity(),
        ] {
            bytes = bytes
                .checked_add(u64::try_from(count).map_err(|_| overflow())?)
                .ok_or_else(overflow)?;
        }
    }
    Ok(bytes)
}
fn overflow() -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid("retained analysis transport storage overflow".into())
}

impl PreparedAnalysisObjects {
    pub(crate) fn retained_storage_bytes(&self) -> AnalysisBridgeResult<u64> {
        let metadata = profile_serialized(
            &(&self.manifest, &self.root),
            CanonicalLimits::platform_content(65_536, 64),
        )?;
        logical_objects_storage(&self.objects)?
            .checked_add(metadata.materialization_resident_bytes_estimate)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(overflow)
    }
}

impl ImportedAnalysisPartition {
    pub(crate) fn retained_storage_bytes(&self) -> AnalysisBridgeResult<u64> {
        let metadata = profile_serialized(
            &(&self.manifest, &self.partition, &self.root),
            CanonicalLimits::platform_content(65_536, 64),
        )?;
        logical_objects_storage(&self.objects)?
            .checked_add(metadata.materialization_resident_bytes_estimate)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(overflow)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn retained_transport_charges_spare_capacity_without_changing_immutable_bytes() {
        let mut objects = crate::prepare_analysis_objects(
            runmat_analysis_fea::execution::AnalysisNumericalObjectKind::NumericalReceipt,
            runmat_execution::Digest::sha256(b"retained-storage-fixture"),
            &[1, 2, 3],
            4096,
            runmat_execution_artifact::ObjectNamespace::ResultValue,
            runmat_execution_artifact::object::ObjectInventoryLimits::default(),
        )
        .unwrap();
        let original = objects.objects[0].bytes.clone();
        let before = objects.retained_storage_bytes().unwrap();
        let original_capacity = objects.objects[0].bytes.capacity();
        objects.objects[0].bytes.reserve_exact(4096);
        let delta = objects.objects[0].bytes.capacity() - original_capacity;
        assert_eq!(objects.objects[0].bytes, original);
        assert!(delta > 0);
        assert_eq!(
            objects.retained_storage_bytes().unwrap() - before,
            delta as u64
        );
    }
}
