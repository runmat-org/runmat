//! Typed materialization admission precedes Serde allocation; expansion charges
//! its separate original dictionary inventory before cloning each collection.
use super::*;
use runmat_geometry_core::GeometryEvaluationControl;
mod typed;
pub(in crate::cdt::volume_mesh) fn indexed_materialization_bytes(
    value: &[u8],
    control: &dyn runmat_canonical_codec::CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    typed::requested_bytes(value, control)
}
#[cfg(test)]
mod tests;

pub(super) struct DecodePolicy<'a> {
    pub limits: CanonicalLimits,
    pub allocations: Option<&'a dyn GeometryEvaluationControl>,
}
impl From<CanonicalLimits> for DecodePolicy<'_> {
    fn from(limits: CanonicalLimits) -> Self {
        Self {
            limits,
            allocations: None,
        }
    }
}
impl DecodePolicy<'_> {
    pub fn admit_legacy(
        &self,
        bytes: &[u8],
        control: &control::Control<'_>,
    ) -> Result<(), CanonicalCodecError> {
        if self.allocations.is_none() {
            return Ok(());
        }
        let bytes = runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes], control)
            ?;
        self.consume(bytes)
    }
    pub fn admit_indexed(
        &self,
        value: &[u8],
        control: &control::Control<'_>,
    ) -> Result<(), CanonicalCodecError> {
        if self.allocations.is_none() {
            return Ok(());
        }
        self.consume(indexed_materialization_bytes(value, control)?)
    }
    fn consume(&self, bytes: u64) -> Result<(), CanonicalCodecError> {
        self.allocations
            .expect("admitted decode owns an allocation scope")
            .consume_allocation_bytes(bytes)
            .map_err(|failure| CanonicalCodecError {
                kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
                field: "volume typed decode admission",
                reason: failure.to_string(),
            })
    }
}

/// Charges are owned by the caller's scope, which must outlive the decoded mesh.
/// V4 counts its owned typed layout and one row's tagged-enum scratch; legacy
/// retains the conservative structural bound. Neither is measured RSS or whole
/// validation scratch. Existing unrestricted codec entry points retain
/// their original behavior for callers with their own admission authority.
pub(in crate::cdt) fn decode(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    allocations: Option<&dyn GeometryEvaluationControl>,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshCodecError> {
    decode_with_limits(
        bytes,
        topology,
        surface,
        metric_request,
        options,
        cancellation,
        DecodePolicy {
            limits: CONTRACT_LIMITS,
            allocations,
        },
    )
}
