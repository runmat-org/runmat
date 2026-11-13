//! Worker-scoped, single-flight preparation of the geometry sizing authority.
//!
//! A dispatcher retains at most one geometry/request pair. Failed or cancelled preparation is
//! never cached. Every consumer still admits the complete retained metric before borrowing it.
use crate::{
    ExactMeshingEvaluatorProvider, ExactMeshingGeometryEvaluation, PreparedExactGeometryObjects,
};
use runmat_execution::Digest;
use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::{
    CurveQualityTargets, MeshingRequest, MetricFieldRequest, SurfaceQualityTargets,
};
use runmat_meshing_curve::{SharedCurveError, SharedCurveErrorKind};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Duration;

#[derive(Clone, Debug)]
pub(crate) struct PreparedMetricProvider<P> {
    inner: P,
    cached: Arc<Mutex<Option<PreparedMetric>>>,
    persistent: Option<Arc<dyn crate::MeshingSubtaskCacheStore>>,
}
#[derive(Debug)]
struct PreparedMetric {
    geometry: Digest,
    requested: MetricFieldRequest,
    curve: CurveQualityTargets,
    surface: SurfaceQualityTargets,
    metric: Arc<MetricFieldRequest>,
    resident_bytes: u64,
}
impl<P> PreparedMetricProvider<P> {
    pub(crate) fn new(inner: P) -> Self {
        Self {
            inner,
            cached: Arc::new(Mutex::new(None)),
            persistent: None,
        }
    }
}
impl<P> PreparedMetricProvider<P> {
    pub(crate) fn with_subtask_cache(
        mut self,
        store: Arc<dyn crate::MeshingSubtaskCacheStore>,
    ) -> Self {
        self.persistent = Some(store);
        self
    }
}

mod persistent;

impl<P: ExactMeshingEvaluatorProvider> ExactMeshingEvaluatorProvider for PreparedMetricProvider<P> {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<
        Box<dyn ExactMeshingGeometryEvaluation + 'a>,
        runmat_geometry_core::GeometryContractError,
    > {
        self.inner.evaluator(geometry, control)
    }
    fn geometry_metric(
        &self,
        geometry: &PreparedExactGeometryObjects,
        evaluator: &dyn ExactMeshingGeometryEvaluation,
        control: &dyn GeometryEvaluationControl,
        request: &MeshingRequest,
    ) -> Result<Arc<MetricFieldRequest>, SharedCurveError> {
        let mut cached = loop {
            control.checkpoint().map_err(control_error)?;
            match self.cached.try_lock() {
                Ok(guard) => break guard,
                Err(TryLockError::WouldBlock) => {
                    std::thread::park_timeout(Duration::from_millis(1))
                }
                Err(TryLockError::Poisoned(_)) => {
                    return Err(SharedCurveError {
                        edge_id: None,
                        kind: SharedCurveErrorKind::InvalidContract,
                        field: "geometry metric preparation".into(),
                        reason: "metric preparation lock poisoned".into(),
                    })
                }
            }
        };
        if let Some(prepared) = cached.as_ref().filter(|prepared| {
            prepared.geometry == geometry.root.digest
                && prepared.requested == request.metric
                && prepared.curve == request.quality.curve
                && prepared.surface == request.quality.surface
        }) {
            control
                .consume_allocation_bytes(prepared.resident_bytes)
                .map_err(control_error)?;
            tracing::info!(
                event_code = "meshing.geometry_metric.reuse",
                contribution_count = prepared.metric.contributions.len(),
                resident_bytes = prepared.resident_bytes,
                "reused admitted geometry sizing"
            );
            return Ok(Arc::clone(&prepared.metric));
        }
        // Eviction precedes construction, so a new request does not accumulate old cached fields.
        *cached = None;
        let preparation = tracing::info_span!(
            "meshing.geometry_metric.prepare",
            edge_count = geometry.topology.edges.len(),
            face_count = geometry.topology.faces.len(),
        )
        .entered();
        let persistent = self.persistent.as_ref();
        let restored = persistent::restore(persistent, geometry, request, control)?;
        let freshly_prepared = restored.is_none();
        let (metric, restored_storage) = match restored {
            Some(restored) => (
                restored.metric,
                Some((restored.storage, restored.reserved_bytes)),
            ),
            None => (
                self.inner
                    .geometry_metric(geometry, evaluator, control, request)?,
                None,
            ),
        };
        drop(preparation);
        let resident_bytes = metric_storage(&metric, false)
            .saturating_add(metric_storage(&request.metric, true))
            .saturating_add(std::mem::size_of::<Mutex<Option<PreparedMetric>>>() as u64)
            .saturating_add(std::mem::size_of::<MetricFieldRequest>() as u64)
            .saturating_add((4 * std::mem::size_of::<usize>()) as u64);
        if let Some((mut storage, reserved)) = restored_storage {
            control
                .consume_allocation_bytes(resident_bytes.saturating_sub(reserved))
                .map_err(control_error)?;
            storage.retain_bytes(resident_bytes.min(reserved));
        } else {
            control
                .consume_allocation_bytes(resident_bytes)
                .map_err(control_error)?;
        }
        if freshly_prepared {
            persistent::retain(persistent, geometry, request, &metric, control)?;
        }
        *cached = Some(PreparedMetric {
            geometry: geometry.root.digest,
            requested: request.metric.clone(),
            curve: request.quality.curve,
            surface: request.quality.surface,
            metric: Arc::clone(&metric),
            resident_bytes,
        });
        Ok(metric)
    }
}
fn control_error(error: runmat_geometry_core::GeometryEvaluationError) -> SharedCurveError {
    SharedCurveError {
        edge_id: None,
        kind: SharedCurveErrorKind::GeometryEvaluation(error.kind),
        field: "geometry metric preparation".into(),
        reason: error.reason,
    }
}
fn metric_storage(metric: &MetricFieldRequest, cloned: bool) -> u64 {
    use runmat_meshing_core::{MetricContribution, MetricContributionScope};
    let count = if cloned {
        metric.contributions.len()
    } else {
        metric.contributions.capacity()
    };
    let mut bytes = (count as u64).saturating_mul(std::mem::size_of::<MetricContribution>() as u64);
    for contribution in &metric.contributions {
        let entity = match &contribution.scope {
            MetricContributionScope::Entity { entity_id } => entity_id,
            MetricContributionScope::Region { region_id } => region_id,
        };
        bytes = bytes
            .saturating_add(if cloned {
                entity.source_topology_id.len()
            } else {
                entity.source_topology_id.capacity()
            } as u64)
            .saturating_add(
                (if cloned {
                    entity.assembly_path.len()
                } else {
                    entity.assembly_path.capacity()
                } as u64)
                    .saturating_mul(std::mem::size_of::<String>() as u64),
            );
        for component in &entity.assembly_path {
            bytes = bytes.saturating_add(if cloned {
                component.len()
            } else {
                component.capacity()
            } as u64);
        }
    }
    bytes
}

#[cfg(test)]
mod tests;
