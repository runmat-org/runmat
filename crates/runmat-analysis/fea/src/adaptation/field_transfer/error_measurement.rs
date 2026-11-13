//! Numerical transfer-error measurement; portable digests bind original fields.
use super::{
    SolverFieldTransferError, SolverFieldTransferErrorEvidence, SolverFieldTransferResult,
};
use runmat_analysis_core::AnalysisField;
use runmat_canonical_codec::CanonicalLimits;
use runmat_meshing_core::StableDigest;

const FIELD_CODEC_PREFIX: &[u8] = b"runmat-analysis-fea-field-cbor/v1\0";
const FIELD_CODEC_DOMAIN: &str = "analysis.fea.field/v1";
const FIELD_CODEC_LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 4096, 16),
);

pub fn measure_solver_field_transfer_error(
    transfer: &SolverFieldTransferResult,
    reference: &AnalysisField,
) -> Result<SolverFieldTransferErrorEvidence, SolverFieldTransferError> {
    if transfer.field.field_id != reference.field_id || transfer.field.shape != reference.shape {
        return Err(SolverFieldTransferError::InvalidReferenceField);
    }
    let transferred = transfer
        .field
        .as_host_f64()
        .ok_or(SolverFieldTransferError::DeviceFieldRequiresHostTransfer)?;
    let reference_values = reference
        .as_host_f64()
        .ok_or(SolverFieldTransferError::DeviceFieldRequiresHostTransfer)?;
    if transferred.len() != reference_values.len()
        || transferred.iter().any(|value| !value.is_finite())
        || reference_values.iter().any(|value| !value.is_finite())
    {
        return Err(SolverFieldTransferError::InvalidReferenceField);
    }
    // hypot rescales each reduction: squaring finite values directly would
    // manufacture zero error at tiny scales or infinity at large scales.
    let absolute_l2_error = transferred
        .iter()
        .zip(reference_values)
        .fold(0.0_f64, |norm, (actual, expected)| {
            norm.hypot(actual - expected)
        });
    let reference_norm = reference_values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    if !absolute_l2_error.is_finite() || !reference_norm.is_finite() {
        return Err(SolverFieldTransferError::InvalidReferenceField);
    }
    let relative_l2_error = if reference_norm == 0.0 {
        if absolute_l2_error == 0.0 {
            Some(0.0)
        } else {
            None
        }
    } else {
        Some(absolute_l2_error / reference_norm)
    };
    if relative_l2_error.is_some_and(|error| !error.is_finite()) {
        return Err(SolverFieldTransferError::InvalidReferenceField);
    }
    Ok(SolverFieldTransferErrorEvidence {
        transfer: transfer.evidence.clone(),
        transferred_field_digest: field_digest(&transfer.field)?,
        reference_field_digest: field_digest(reference)?,
        absolute_l2_error,
        relative_l2_error,
    })
}

fn field_digest(field: &AnalysisField) -> Result<StableDigest, SolverFieldTransferError> {
    // Streamed: a mesh-sized field is hashed without materializing its encoding.
    let digest = crate::contracts::canonical_digest::canonical_contract_digest(
        FIELD_CODEC_PREFIX,
        FIELD_CODEC_DOMAIN,
        field,
        FIELD_CODEC_LIMITS,
    )
    .map_err(|_| SolverFieldTransferError::InvalidReferenceField)?;
    Ok(StableDigest::from_bytes(*digest.bytes()))
}
