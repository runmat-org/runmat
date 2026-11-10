//! Domain-derived typed retention and actual codec phases under one staging ceiling.
use super::*;
/// Inventory the borrowed original owner; resource rejection stays a resource
/// error instead of being mislabeled as a malformed physical result.
pub(in crate::analysis) fn retained_mesh_storage(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    mesh.retained_storage_bytes(maximum, &mut || {
        if crate::interrupt::is_cancelled() {
            Err(runmat_meshing_core::MeshingContractError::invalid(
                "publication storage inventory",
                "cancelled",
            ))
        } else {
            Ok(())
        }
    })
    .map_err(storage_error)
}
fn storage_error(error: runmat_meshing_core::MeshingContractError) -> AnalysisPublicationError {
    match error.field.as_str() {
        "mesh retained storage" | "mesh validation resources" => {
            AnalysisPublicationError::ResourceLimit {
                message: error.to_string(),
            }
        }
        _ => AnalysisPublicationError::from(error.to_string()),
    }
}
impl AnalysisPublicationBuilder {
    pub(in crate::analysis) fn admit_preparation_storage(
        &mut self,
        original_bytes: u64,
        copy_bytes: u64,
    ) -> Result<PreparationReservation, AnalysisPublicationError> {
        let required = original_bytes.checked_add(copy_bytes).ok_or_else(|| {
            AnalysisPublicationError::ResourceLimit {
                message: "typed preparation storage overflow".into(),
            }
        })?;
        self.admit(required)
            .map_err(|error| error.context("typed publication original and preparation copies"))?;
        self.preparation_resident_bytes = self
            .preparation_resident_bytes
            .checked_add(required)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "typed preparation accumulation overflow".into(),
            })?;
        Ok(PreparationReservation {
            scope: self.preparation_scope.clone(),
            resident_bytes: original_bytes,
        })
    }

    pub(in crate::analysis) fn admit_mesh_persistence(
        &self,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
        evidence: &runmat_meshing_core::MeshingEvidence,
    ) -> Result<(), AnalysisPublicationError> {
        let limits = CanonicalLimits::content_bounded(
            usize::try_from(self.maximum_bytes).map_err(|_| {
                AnalysisPublicationError::ResourceLimit {
                    message: "mesh codec budget exceeds address space".into(),
                }
            })?,
            65_536,
            64,
        );
        let mesh_validation_and_encoding = mesh
            .canonical_encoding_resident_bytes_estimate()
            .map_err(storage_error)?;
        let mesh = runmat_canonical_codec::profile_canonical_encoding(mesh, limits)?;
        let evidence = runmat_canonical_codec::profile_canonical_encoding(evidence, limits)?;
        // The first returned Vec is held while evidence is encoded. Reserve its
        // capacity growth, not only its logical wire length. Neither survives
        // persistence; whole staged output remains separately charged.
        let peak = mesh
            .encoded_bytes_upper_bound
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(evidence.resident_workspace_bytes_estimate))
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "mesh persistence workspace overflow".into(),
            })?
            .max(mesh_validation_and_encoding);
        // Captured providers may return an owned immutable object if a creator
        // wins the race after metadata inspection. Both encoded Vecs are still
        // retained then; admit the bounded comparison owner as a separate phase.
        let peak = if super::super::scoped_artifact_provider().is_some() {
            mesh.encoded_bytes_upper_bound
                .checked_add(evidence.encoded_bytes_upper_bound)
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| {
                    bytes.checked_add(
                        mesh.encoded_bytes_upper_bound
                            .max(evidence.encoded_bytes_upper_bound),
                    )
                })
                .and_then(|bytes| bytes.checked_add(8192))
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "mesh collision workspace overflow".into(),
                })?
                .max(peak)
        } else {
            peak
        };
        self.admit(peak).map_err(|error| {
            error.context(&format!(
                "original mesh encoding phase: mesh_wire_upper={}, mesh_validation_and_encoding_estimate={}, evidence_wire_upper={}, evidence_workspace_estimate={}",
                mesh.encoded_bytes_upper_bound, mesh_validation_and_encoding,
                evidence.encoded_bytes_upper_bound, evidence.resident_workspace_bytes_estimate,
            ))
        })
    }

    pub(in crate::analysis) fn admit_compiled_manifest_copy(
        &mut self,
        compiled: &runmat_analysis_fea::study::CompiledAnalysisStudy,
        metadata: &impl Serialize,
    ) -> Result<OwnedCopyReservation, AnalysisPublicationError> {
        let limits = CanonicalLimits::content_bounded(
            usize::try_from(self.maximum_bytes).map_err(|_| {
                AnalysisPublicationError::ResourceLimit {
                    message: "manifest clone budget exceeds address space".into(),
                }
            })?,
            65_536,
            64,
        );
        let compiled = compiled
            .shared_clone_storage_bytes_estimate(limits)
            .map_err(|message| AnalysisPublicationError::ResourceLimit { message })?;
        let copies = resident(metadata, self.maximum_bytes)?
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(compiled))
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "manifest copy storage overflow".into(),
            })?;
        let reservation = self.admit_owned_copies(copies)?;
        *self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")? = true;
        Ok(reservation)
    }

    /// Original typed inputs/copies were reserved before family completion. This
    /// phase constructs one JSON tree, with no CBOR decode/re-encode buffers.
    pub(in crate::analysis) fn serialize_retained_manifest(
        &self,
        value: &impl Serialize,
    ) -> Result<serde_json::Value, AnalysisPublicationError> {
        let mut guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        if self.preparation_resident_bytes == 0 {
            return Err("manifest lacks its retained preparation reservation".into());
        }
        let profile = profile_serialized(
            value,
            CanonicalLimits::content_bounded(
                usize::try_from(self.maximum_bytes).map_err(|_| {
                    AnalysisPublicationError::ResourceLimit {
                        message: "manifest budget exceeds address space".into(),
                    }
                })?,
                65_536,
                64,
            ),
        )?;
        self.admit(
            profile
                .json_projection_resident_bytes_estimate()
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "invalid JSON projection workspace".into(),
                })?,
        )
        .map_err(|error| error.context("retained typed manifest JSON projection"))?;
        *guard = true;
        serde_json::to_value(value).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests;
