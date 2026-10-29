use std::collections::BTreeSet;

use runmat_analysis_fea::results::observation::{
    BoundedObservationHistory, ObservationHistory, ObservationHistoryCheckpoint,
    ObservationHistoryCheckpointIdentity,
};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};

use crate::{
    prepare_observation_history_checkpoint_objects, prepare_observation_history_objects,
    AnalysisBridgeError, AnalysisBridgeResult, ImportedObservationHistoryCheckpoint,
    PreparedAnalysisObjects,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PublishedObservationHistoryCheckpoint {
    pub checkpoint: ObservationHistoryCheckpoint,
    pub objects: PreparedAnalysisObjects,
    pub root: ValueRef,
    pub written_object_count: usize,
    pub written_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublishedObservationHistory {
    pub history: ObservationHistory,
    pub objects: PreparedAnalysisObjects,
    pub root: ValueRef,
    pub written_object_count: usize,
    pub written_bytes: u64,
}

/// Transactional adapter from an analysis-owned bounded history to the shared content-addressed
/// artifact cache.
///
/// The publisher advances its immutable lineage only after every object has been admitted by the
/// cache. Reused retained samples keep identical chunk digests, so an idempotent cache writes only
/// the new sample chunks and checkpoint manifest physically.
pub struct ObservationHistoryCheckpointPublisher {
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
    previous_closure_digests: BTreeSet<runmat_execution::Digest>,
}

impl ObservationHistoryCheckpointPublisher {
    pub fn new(chunk_bytes: usize, limits: ObjectInventoryLimits) -> AnalysisBridgeResult<Self> {
        if chunk_bytes == 0 {
            return Err(AnalysisBridgeError::Invalid(
                "history checkpoint chunk size must be nonzero".into(),
            ));
        }
        Ok(Self {
            predecessor: None,
            chunk_bytes,
            limits,
            previous_closure_digests: BTreeSet::new(),
        })
    }

    pub fn resume_from(
        imported: &ImportedObservationHistoryCheckpoint,
        chunk_bytes: usize,
        limits: ObjectInventoryLimits,
    ) -> AnalysisBridgeResult<Self> {
        let mut publisher = Self::new(chunk_bytes, limits)?;
        publisher.predecessor = Some(imported.checkpoint.manifest().identity);
        publisher
            .previous_closure_digests
            .insert(imported.root.logical_digest);
        publisher.previous_closure_digests.extend(
            imported
                .object_manifest
                .chunks
                .iter()
                .map(|chunk| chunk.digest),
        );
        Ok(publisher)
    }

    pub const fn predecessor(&self) -> Option<ObservationHistoryCheckpointIdentity> {
        self.predecessor
    }

    pub fn publish(
        &mut self,
        store: &mut impl CacheExport,
        history: &BoundedObservationHistory,
        access: &ArtifactAccess,
    ) -> AnalysisBridgeResult<PublishedObservationHistoryCheckpoint> {
        let checkpoint = history
            .checkpoint(self.predecessor)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let objects = prepare_observation_history_checkpoint_objects(
            &checkpoint,
            self.chunk_bytes,
            ObjectNamespace::ResultValue,
            self.limits,
        )?;
        let root = objects.root_reference(access, ValueRefKind::ResultObject)?;
        let (written_object_count, written_bytes) = self.write_new(store, &objects)?;
        self.predecessor = Some(checkpoint.manifest().identity);
        Ok(PublishedObservationHistoryCheckpoint {
            checkpoint,
            objects,
            root,
            written_object_count,
            written_bytes,
        })
    }

    pub fn finish_and_publish(
        &mut self,
        store: &mut impl CacheExport,
        history: BoundedObservationHistory,
        access: &ArtifactAccess,
    ) -> AnalysisBridgeResult<PublishedObservationHistory> {
        let history = history
            .finish()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let objects = prepare_observation_history_objects(
            &history,
            self.chunk_bytes,
            ObjectNamespace::ResultValue,
            self.limits,
        )?;
        let root = objects.root_reference(access, ValueRefKind::ResultObject)?;
        let (written_object_count, written_bytes) = self.write_new(store, &objects)?;
        Ok(PublishedObservationHistory {
            history,
            objects,
            root,
            written_object_count,
            written_bytes,
        })
    }

    fn write_new(
        &mut self,
        store: &mut impl CacheExport,
        objects: &PreparedAnalysisObjects,
    ) -> AnalysisBridgeResult<(usize, u64)> {
        let pending = objects
            .objects
            .iter()
            .filter(|object| {
                !self
                    .previous_closure_digests
                    .contains(&object.descriptor.digest)
            })
            .collect::<Vec<_>>();
        let written_bytes = pending.iter().try_fold(0_u64, |total, object| {
            total
                .checked_add(object.descriptor.encoded_length)
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(
                        "history checkpoint written byte count overflows".into(),
                    )
                })
        })?;
        store.write_verified_batch(pending.iter().copied())?;
        let written_object_count = pending.len();
        drop(pending);
        self.previous_closure_digests = objects
            .objects
            .iter()
            .map(|object| object.descriptor.digest)
            .collect();
        Ok((written_object_count, written_bytes))
    }
}
