//! Shared count-only ceiling used by task planning and direct numerical execution.
use super::AcousticHarmonicError;
use crate::solve::uniqueness::{
    SparseUniquenessEnvelope, SparseUniquenessError, SparseUniquenessPolicy,
};
use runmat_meshing_core::SolverMeshArtifact;

pub(super) fn certify(
    operator: &crate::algebra::RealSparseMatrix,
    policy: Option<SparseUniquenessPolicy>,
    admitted: Option<SparseUniquenessEnvelope>,
) -> Result<Option<crate::solve::uniqueness::SparseUniquenessRecord>, AcousticHarmonicError> {
    use crate::solve::uniqueness::{
        certify_sparse_uniqueness, SparseUniquenessRecord, SparseUniquenessVerdict,
    };
    let Some(policy) = policy else {
        if admitted.is_some() {
            return Err(AcousticHarmonicError::Uniqueness(
                SparseUniquenessError::Admission,
            ));
        }
        return Ok(None);
    };
    let admitted = admitted.ok_or(AcousticHarmonicError::Uniqueness(
        SparseUniquenessError::Admission,
    ))?;
    if crate::is_fea_cancelled() {
        return Err(AcousticHarmonicError::Cancelled);
    }
    if operator.rows() == 0 {
        let identity = operator
            .numerical_identity()
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        return SparseUniquenessRecord::no_free_unknowns(identity, policy)
            .map(Some)
            .map_err(AcousticHarmonicError::Uniqueness);
    }
    // The provider rederives work/storage from actual dimension, nnz, row width
    // and controls, then checks that envelope before factor allocation.
    let verdict = certify_sparse_uniqueness(operator, policy, admitted).map_err(|error| {
        if error == SparseUniquenessError::Cancelled {
            AcousticHarmonicError::Cancelled
        } else {
            AcousticHarmonicError::Uniqueness(error)
        }
    })?;
    match verdict {
        SparseUniquenessVerdict::Verified {
            certificate,
            iterations,
        } => SparseUniquenessRecord::from_certificate(policy, &certificate, iterations)
            .map(Some)
            .map_err(AcousticHarmonicError::Uniqueness),
        SparseUniquenessVerdict::Indeterminate {
            reason,
            completed_columns,
            iterations,
            defect_infinity_upper,
        } => {
            let operator_identity = operator
                .numerical_identity()
                .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
            Err(AcousticHarmonicError::UniquenessIndeterminate {
                operator_identity,
                reason,
                completed_columns,
                iterations,
                defect_infinity_upper,
            })
        }
    }
}

pub(crate) fn uniqueness_admission(
    mesh: &SolverMeshArtifact,
    policy: Option<SparseUniquenessPolicy>,
) -> Result<Option<SparseUniquenessEnvelope>, AcousticHarmonicError> {
    let Some(policy) = policy else {
        return Ok(None);
    };
    let checked = || -> Result<SparseUniquenessEnvelope, SparseUniquenessError> {
        let block = mesh
            .topology
            .nodes
            .len()
            .checked_mul(2)
            .ok_or(SparseUniquenessError::Overflow)?;
        let volume = mesh
            .topology
            .volume_elements
            .iter()
            .try_fold(0usize, |sum, element| {
                sum.checked_add(
                    element
                        .node_ids
                        .len()
                        .checked_mul(element.node_ids.len())
                        .ok_or(SparseUniquenessError::Overflow)?,
                )
                .ok_or(SparseUniquenessError::Overflow)
            })?;
        let boundary = mesh
            .topology
            .boundary_faces
            .iter()
            .try_fold(0usize, |sum, face| {
                sum.checked_add(
                    face.node_ids
                        .len()
                        .checked_mul(face.node_ids.len())
                        .ok_or(SparseUniquenessError::Overflow)?,
                )
                .ok_or(SparseUniquenessError::Overflow)
            })?;
        // K/M/B raw incidence; two real blocks; each recovery row has at most
        // two coefficients for unit-phase constraints. Cancellations reduce nnz.
        let entries = volume
            .checked_mul(2)
            .and_then(|z| z.checked_add(boundary))
            .and_then(|z| z.checked_mul(8))
            .ok_or(SparseUniquenessError::Overflow)?;
        let entries = entries.min(
            block
                .checked_mul(block)
                .ok_or(SparseUniquenessError::Overflow)?,
        );
        SparseUniquenessEnvelope::for_counts(block, entries, block, policy)
    };
    checked()
        .map(Some)
        .map_err(AcousticHarmonicError::Uniqueness)
}
