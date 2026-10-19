//! Cross-worker sizing reuse. Input identity and typed metric admission belong to meshing.
use super::*;
use crate::codec_control::CodecControl;
use crate::{
    subtask_cache::{identity, record},
    MeshingSubtaskCacheError, MeshingSubtaskCacheStore,
};
use runmat_canonical_codec::{CanonicalCodecErrorKind, CanonicalLimits};

const PREFIX: &[u8] = b"runmat-prepared-geometry-metric/v1\0";
const DOMAIN: &str = "runmat-prepared-geometry-metric/v1";

fn invalid(reason: String) -> SharedCurveError {
    SharedCurveError {
        edge_id: None,
        kind: SharedCurveErrorKind::InvalidContract,
        field: "persistent geometry sizing".into(),
        reason,
    }
}
fn store_error(error: MeshingSubtaskCacheError) -> SharedCurveError {
    match error {
        MeshingSubtaskCacheError::Control(error) => control_error(error),
        MeshingSubtaskCacheError::Storage(reason) => invalid(reason),
    }
}
fn limits(store: &dyn MeshingSubtaskCacheStore) -> CanonicalLimits {
    CanonicalLimits::new(
        usize::try_from(store.maximum_entry_bytes())
            .unwrap_or(usize::MAX)
            .saturating_sub(record::HEADER_BYTES),
        1_000_000,
        1_048_576,
        64,
    )
}
fn admitted<'a>(
    store: Option<&'a Arc<dyn MeshingSubtaskCacheStore>>,
    control: &dyn GeometryEvaluationControl,
) -> Result<Option<&'a dyn MeshingSubtaskCacheStore>, SharedCurveError> {
    let Some(store) = store else {
        return Ok(None);
    };
    if store.implementation_identity() == runmat_meshing_core::StableDigest::ZERO {
        return Err(invalid("sizing cache lacks implementation identity".into()));
    }
    Ok(store
        .admit(control)
        .map_err(store_error)?
        .then_some(store.as_ref()))
}
pub(super) struct RestoredMetric<'a> {
    pub metric: Arc<MetricFieldRequest>,
    pub storage: runmat_geometry_core::GeometryAllocationReservation<'a>,
    pub reserved_bytes: u64,
}

pub(super) fn restore<'a>(
    store: Option<&Arc<dyn MeshingSubtaskCacheStore>>,
    geometry: &PreparedExactGeometryObjects,
    request: &MeshingRequest,
    control: &'a dyn GeometryEvaluationControl,
) -> Result<Option<RestoredMetric<'a>>, SharedCurveError> {
    let Some(store) = admitted(store, control)? else {
        return Ok(None);
    };
    let key = identity::metric(
        store.implementation_identity(),
        &geometry.root.digest,
        request,
        control,
    )
    .map_err(store_error)?;
    let Some(entry) = store.read(key, control).map_err(store_error)? else {
        return Ok(None);
    };
    if entry.bytes.len() as u64 > store.maximum_entry_bytes() {
        return Err(invalid("cached sizing exceeds byte limit".into()));
    }
    let bytes = record::body(&entry.bytes, key).map_err(store_error)?;
    let codec = CodecControl::new(control);
    let workspace =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            std::iter::once(bytes),
            &codec,
        )
        .map_err(|e| store_error(codec.error(e)))?;
    if workspace > entry.materialization_resident_bytes {
        return Err(invalid(
            "cached sizing exceeds admitted materialization inventory".into(),
        ));
    }
    let metric: MetricFieldRequest = runmat_canonical_codec::decode_contract_with_control(
        PREFIX,
        DOMAIN,
        bytes,
        limits(store),
        &codec,
    )
    .map_err(|e| store_error(codec.error(e)))?;
    runmat_meshing_curve::ResolvedCurveMetricField::new(&geometry.topology, &metric)?;
    let reserved_bytes = (entry.bytes.len() as u64)
        .checked_add(entry.materialization_resident_bytes)
        .ok_or_else(|| invalid("sizing reservation inventory overflow".into()))?;
    tracing::info!(
        event_code = "meshing.geometry_metric.persistent_hit",
        contribution_count = metric.contributions.len(),
        "reused original prepared sizing across workers"
    );
    Ok(Some(RestoredMetric {
        metric: Arc::new(metric),
        storage: entry.storage,
        reserved_bytes,
    }))
}
pub(super) fn retain(
    store: Option<&Arc<dyn MeshingSubtaskCacheStore>>,
    geometry: &PreparedExactGeometryObjects,
    request: &MeshingRequest,
    metric: &MetricFieldRequest,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), SharedCurveError> {
    let Some(store) = admitted(store, control)? else {
        return Ok(());
    };
    runmat_meshing_curve::ResolvedCurveMetricField::new(&geometry.topology, metric)?;
    control.checkpoint().map_err(control_error)?;
    let codec = CodecControl::new(control);
    let profile = match runmat_canonical_codec::profile_canonical_encoding_with_control(
        metric,
        limits(store),
        &codec,
    ) {
        Ok(value) => value,
        Err(error) if error.kind == CanonicalCodecErrorKind::ResourceLimit => return Ok(()),
        Err(error) => return Err(store_error(codec.error(error))),
    };
    control.checkpoint().map_err(control_error)?;
    let _encoding_storage = match runmat_geometry_core::GeometryAllocationReservation::new(
        control,
        profile.resident_workspace_bytes_estimate,
    ) {
        Ok(reservation) => reservation,
        Err(error)
            if error.kind
                == runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded =>
        {
            return Ok(())
        }
        Err(error) => return Err(control_error(error)),
    };
    let key = identity::metric(
        store.implementation_identity(),
        &geometry.root.digest,
        request,
        control,
    )
    .map_err(store_error)?;
    let codec = CodecControl::new(control);
    let mut bytes = runmat_canonical_codec::encode_contract_with_control(
        PREFIX,
        DOMAIN,
        metric,
        limits(store),
        &codec,
    )
    .map_err(|e| store_error(codec.error(e)))?;
    let workspace =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            std::iter::once(bytes.as_slice()),
            &codec,
        )
        .map_err(|e| store_error(codec.error(e)))?;
    record::frame(&mut bytes, key);
    if store
        .write(key, &bytes, workspace, control)
        .map_err(store_error)?
    {
        tracing::info!(
            event_code = "meshing.geometry_metric.persisted",
            encoded_bytes = bytes.len(),
            "retained prepared geometry sizing for current implementation and inputs"
        );
    }
    Ok(())
}
