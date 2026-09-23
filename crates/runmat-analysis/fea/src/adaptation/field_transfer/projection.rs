use std::collections::BTreeMap;

use runmat_meshing_core::{SolverEntityTransfer, SolverTransferMethod, StableDigest};

use super::{SolverFieldTransferError, SolverFieldTransferMethod};

pub(super) fn apply_projection(
    projection: &SolverEntityTransfer,
    source_values: &BTreeMap<StableDigest, &[f64]>,
    component_count: usize,
) -> Result<Vec<f64>, SolverFieldTransferError> {
    let mut result = vec![0.0; component_count];
    for source in &projection.sources {
        let values = source_values
            .get(&source.stable_identity)
            .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
        for (result, value) in result.iter_mut().zip(*values) {
            *result += source.weight * value;
        }
    }
    Ok(result)
}

pub(super) fn field_transfer_method(method: SolverTransferMethod) -> SolverFieldTransferMethod {
    match method {
        SolverTransferMethod::BarycentricInterpolation => {
            SolverFieldTransferMethod::BarycentricInterpolation
        }
        SolverTransferMethod::CentroidProjection => SolverFieldTransferMethod::CentroidProjection,
        SolverTransferMethod::ConservativeProjection => {
            SolverFieldTransferMethod::ConservativeProjection
        }
    }
}
