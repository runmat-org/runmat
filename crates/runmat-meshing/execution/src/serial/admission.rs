//! Stage-scoped admission of imported wire buffers before provider allocation.
//! Typed decoding and kernel scratch are separate demands; this reader must not
//! label encoded bytes as the whole input heap or as measured process residency.
mod stage;
use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, ArtifactError, ArtifactResult};
use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryAllocationScope, GeometryEvaluationError,
};
use std::sync::Mutex;

pub(super) struct InputReader<'a, S> {
    source: &'a S,
    scope: &'a GeometryAllocationScope<'a>,
    failure: Mutex<Option<GeometryEvaluationError>>,
}
impl<'a, S> InputReader<'a, S> {
    pub fn new(source: &'a S, scope: &'a GeometryAllocationScope<'a>) -> Self {
        Self {
            source,
            scope,
            failure: Mutex::new(None),
        }
    }
    pub fn take_failure(&self) -> Option<GeometryEvaluationError> {
        self.failure.lock().expect("input control failure").take()
    }
    fn control_error(&self, error: GeometryEvaluationError) -> ArtifactError {
        let reason = error.to_string();
        let mut original = self.failure.lock().expect("input control failure");
        if original.is_none() {
            *original = Some(error);
        }
        ArtifactError::Limit(reason)
    }
}
impl<S: CacheImport> CacheImport for InputReader<'_, S> {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Err(ArtifactError::Limit(
            "stage inputs require a bounded provider read".into(),
        ))
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if maximum_bytes == 0 {
            return Err(ArtifactError::Limit(
                "input read bound must be positive".into(),
            ));
        }
        runmat_geometry_core::GeometryEvaluationControl::checkpoint(self.scope)
            .map_err(|error| self.control_error(error))?;
        let mut retained = None;
        let mut requested = 0;
        let value = self.source.read_verified_bounded_with_admission(
            digest,
            maximum_bytes,
            &mut |capacity| {
                if retained.is_some() || capacity > maximum_bytes {
                    return Err(ArtifactError::Limit(
                        "provider read admission is repeated or exceeds its bound".into(),
                    ));
                }
                requested = capacity;
                retained = Some(
                    GeometryAllocationReservation::new(self.scope, capacity)
                        .map_err(|error| self.control_error(error))?,
                );
                Ok(())
            },
        );
        runmat_geometry_core::GeometryEvaluationControl::checkpoint(self.scope)
            .map_err(|error| self.control_error(error))?;
        let value = value?;
        if let Some(bytes) = &value {
            let reservation = retained.as_mut().ok_or_else(|| {
                ArtifactError::Limit("input provider omitted buffer admission".into())
            })?;
            if bytes.capacity() as u64 > requested {
                return Err(ArtifactError::Limit(
                    "input provider exceeded its admitted buffer capacity".into(),
                ));
            }
            reservation.retain_bytes(bytes.capacity() as u64);
        }
        Ok(value)
    }
}

/// Import dispatch belongs beside the shared input admission, not in the stage orchestrator.
pub(super) fn import_inputs<S: CacheImport>(
    host: &crate::MeshingHostWorkload,
    roots: &[runmat_execution::value::ValueRef],
    input_reader: &InputReader<'_, S>,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> Result<Vec<super::PreparedMeshingInput>, super::MeshingSerialExecutionError> {
    use super::{failure, MeshingSerialExecutionError, PreparedMeshingInput};
    use crate::{
        evidence_objects::import_evidence_input_controlled,
        faceted_geometry_objects::import_faceted_geometry_input_controlled,
        geometry_objects::import_exact_geometry_input_controlled, MeshingExecutionError,
    };
    use runmat_geometry_core::GeometryEvaluationControl;
    use runmat_meshing_core::MeshingInputKind;
    let codec = crate::codec_control::CodecControl::new(input_reader.scope);
    let admit = |bytes| {
        codec
            .consume_allocation_bytes(bytes)
            .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))
    };
    let inputs = (|| -> Result<Vec<PreparedMeshingInput>, MeshingExecutionError> {
        admit(crate::object_support::import_slots::<PreparedMeshingInput>(
            host.workload.inputs.len(),
        )?)?;
        let mut inputs = Vec::with_capacity(host.workload.inputs.len());
        for (input, root) in host.workload.inputs.iter().zip(roots) {
            let inline = match input.kind {
                MeshingInputKind::ExactGeometry => {
                    std::mem::size_of::<crate::PreparedExactGeometryInput>()
                }
                MeshingInputKind::FacetedGeometry => {
                    std::mem::size_of::<crate::PreparedFacetedGeometryInput>()
                }
                MeshingInputKind::Evidence => std::mem::size_of::<crate::PreparedEvidenceInput>(),
                MeshingInputKind::StageArtifact => {
                    std::mem::size_of::<crate::PreparedMeshingResultPublication>()
                }
            };
            admit(inline as u64)?;
            if input.kind != MeshingInputKind::StageArtifact {
                admit(host.artifact_access.authorization_scope.len() as u64)?;
            }
            let prepared = match input.kind {
                MeshingInputKind::ExactGeometry => import_exact_geometry_input_controlled(
                    input_reader,
                    clone_document(host, &codec)?,
                    root,
                    host.artifact_access.clone(),
                    limits,
                    Some(&codec),
                )
                .map(Box::new)
                .map(PreparedMeshingInput::ExactGeometry),
                MeshingInputKind::FacetedGeometry => import_faceted_geometry_input_controlled(
                    input_reader,
                    clone_document(host, &codec)?,
                    root,
                    host.artifact_access.clone(),
                    limits,
                    Some(&codec),
                )
                .map(Box::new)
                .map(PreparedMeshingInput::FacetedGeometry),
                MeshingInputKind::Evidence => import_evidence_input_controlled(
                    input_reader,
                    root,
                    host.artifact_access.clone(),
                    limits,
                    Some(&codec),
                )
                .map(Box::new)
                .map(PreparedMeshingInput::Evidence),
                MeshingInputKind::StageArtifact => {
                    stage::import(input_reader, root, &host.artifact_access, limits)
                        .map(Box::new)
                        .map(PreparedMeshingInput::StageArtifact)
                }
            }?;
            inputs.push(prepared);
        }
        Ok(inputs)
    })()
    .map_err(|error| {
        match codec.take_failure().or_else(|| input_reader.take_failure()) {
            Some(original) => MeshingSerialExecutionError::Stage(failure(
                host.workload.stage,
                crate::geometry_failure::category(original.kind),
                &original.reason,
                None,
            )),
            None => MeshingSerialExecutionError::Bridge(error),
        }
    })?;
    Ok(inputs)
}

#[cfg(test)]
mod tests;

fn clone_document(
    host: &crate::MeshingHostWorkload,
    codec: &crate::codec_control::CodecControl<'_>,
) -> crate::MeshingExecutionResult<runmat_geometry_core::GeometryDocument> {
    use runmat_geometry_core::GeometryEvaluationControl;
    let document = host.geometry_document.as_ref().ok_or_else(|| {
        crate::MeshingExecutionError::Invalid("geometry input is missing its host document".into())
    })?;
    codec
        .consume_allocation_bytes(document.cloned_buffer_storage_bytes()?)
        .map_err(|error| crate::MeshingExecutionError::Invalid(error.to_string()))?;
    Ok(document.clone())
}
