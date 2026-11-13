//! Capture one borrowed quality input per connected component. Persistence stays
//! outside the numerical owner; no diagnostic is consumed as an accepted stage.
use crate::codec_control::CodecControl;
use crate::subtask_cache::diagnostic_chunks;
use crate::{MeshingSubtaskCacheError, MeshingSubtaskCacheStore};
use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryEvaluationControl, GeometryEvaluationErrorKind,
};
use runmat_meshing_core::StableDigest;
use runmat_meshing_tetrahedron::cdt::{
    prepare_volume_quality_snapshot, volume_quality_snapshot_preparation_storage_bytes,
    DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage,
    VolumeQualityBinding, VolumeQualityObserver, VolumeQualitySnapshot,
};
use std::sync::Mutex;

pub(super) struct Observer<'a> {
    pub store: Option<&'a dyn MeshingSubtaskCacheStore>,
    pub binding: VolumeQualityBinding,
    pub control: &'a dyn GeometryEvaluationControl,
    pub failure: Mutex<Option<MeshingSubtaskCacheError>>,
}

impl VolumeQualityObserver for Observer<'_> {
    fn observe_stalled(
        &self,
        snapshot: VolumeQualitySnapshot<'_>,
    ) -> Result<(), DelaunayVolumeMeshError> {
        tracing::info!(
            node_count = snapshot.topology.nodes.len(),
            tetrahedron_count = snapshot.topology.tetrahedra.len(),
            "retaining terminal unaccepted quality state for explicit diagnostic replay"
        );
        self.capture(snapshot, false)
    }
    fn observe(&self, snapshot: VolumeQualitySnapshot<'_>) -> Result<(), DelaunayVolumeMeshError> {
        self.capture(snapshot, true)
    }
}
impl Observer<'_> {
    fn capture(
        &self,
        snapshot: VolumeQualitySnapshot<'_>,
        record_failure: bool,
    ) -> Result<(), DelaunayVolumeMeshError> {
        let Some(store) = self.store else {
            return Ok(());
        };
        let Some(implementation) = store
            .volume_implementation_identity()
            .filter(|id| *id != StableDigest::ZERO)
        else {
            return Ok(());
        };
        let binding = VolumeQualityBinding {
            implementation,
            ..self.binding
        };
        match self.retain(store, &snapshot, binding) {
            Ok(()) => Ok(()),
            // Optional diagnostics cannot spend the solver's required memory.
            Err(MeshingSubtaskCacheError::Control(error))
                if error.kind == GeometryEvaluationErrorKind::AllocationBudgetExceeded =>
            {
                tracing::info!(reason=%error.reason, "volume quality diagnostic not admitted; original solve continues");
                Ok(())
            }
            Err(failure) => {
                if record_failure {
                    *self.failure.lock().expect("diagnostic failure lock") = Some(failure);
                }
                Err(DelaunayVolumeMeshError {
                    stage: DelaunayVolumeMeshStage::Quality,
                    kind: DelaunayVolumeMeshErrorKind::Cancelled,
                    reason: "quality diagnostic interrupted; execution retains original failure"
                        .into(),
                })
            }
        }
    }
}
impl Observer<'_> {
    fn retain(
        &self,
        store: &dyn MeshingSubtaskCacheStore,
        snapshot: &VolumeQualitySnapshot<'_>,
        binding: VolumeQualityBinding,
    ) -> Result<(), MeshingSubtaskCacheError> {
        if !store.admit(self.control)? {
            return Ok(());
        }
        let codec = CodecControl::new(self.control);
        let limits = runmat_canonical_codec::CanonicalLimits::new(
            1024 * 1024 * 1024,
            50_000_000,
            1_048_576,
            64,
        );
        let preparation_bytes = volume_quality_snapshot_preparation_storage_bytes(snapshot)
            .map_err(|error| codec.error(error))?;
        let _preparation = GeometryAllocationReservation::new(self.control, preparation_bytes)
            .map_err(MeshingSubtaskCacheError::Control)?;
        let prepared = prepare_volume_quality_snapshot(snapshot, binding, &codec)
            .map_err(|error| codec.error(error))?;
        let (profile, materialization) = prepared
            .profile(limits, &codec)
            .map_err(|error| codec.error(error))?;
        let _encoding = GeometryAllocationReservation::new(
            self.control,
            profile.resident_workspace_bytes_estimate,
        )
        .map_err(MeshingSubtaskCacheError::Control)?;
        let bytes = prepared
            .encode(limits, &codec)
            .map_err(|error| codec.error(error))?;
        drop(prepared);
        drop(_preparation);
        if let Some(key) = diagnostic_chunks::retain(store, &bytes, materialization, self.control)?
        {
            tracing::info!(diagnostic_key=?key, ?binding, encoded_bytes=bytes.len(),
                node_count=snapshot.topology.nodes.len(), tetrahedron_count=snapshot.topology.tetrahedra.len(),
                "volume quality input retained for explicit diagnostic replay; not an accepted mesh");
        } else {
            tracing::info!(
                encoded_bytes = bytes.len(),
                "volume quality diagnostic exceeds cache retention quota; original solve continues"
            );
        }
        Ok(())
    }
}
