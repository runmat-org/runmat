use super::*;
use crate::codec_control::CodecControl;
use runmat_geometry_core::GeometryEvaluationErrorKind;
use runmat_meshing_curve::SharedCurveMesh;
use runmat_meshing_surface::decode_exact_face_reuse_record_with_cancellation;

pub(crate) struct SurfaceFaceReuse<'a> {
    store: Arc<dyn MeshingSubtaskCacheStore>,
    scope: StableDigest,
    topology: &'a runmat_geometry_core::ExactBRepTopology,
    curves: &'a SharedCurveMesh,
    control: &'a dyn GeometryEvaluationControl,
    hits: u64,
    retained: u64,
}

impl<'a> SurfaceFaceReuse<'a> {
    pub(crate) fn new(
        store: Arc<dyn MeshingSubtaskCacheStore>,
        geometry: &'a crate::PreparedExactGeometryObjects,
        request: &runmat_meshing_core::MeshingRequest,
        metric: &runmat_meshing_core::MetricFieldRequest,
        curves: &'a SharedCurveMesh,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<Self>, ExactFacePartitionError> {
        if !store.admit(control).map_err(store_error)? {
            return Ok(None);
        }
        if store.implementation_identity() == StableDigest::ZERO {
            return Err(invalid("face cache lacks an implementation identity"));
        }
        let scope = identity::scope(
            store.implementation_identity(),
            &geometry.root.digest,
            request,
            metric,
            control,
        )?;
        Ok(Some(Self {
            store,
            scope,
            topology: &geometry.topology,
            curves,
            control,
            hits: 0,
            retained: 0,
        }))
    }
    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }
    pub(crate) fn retained(&self) -> u64 {
        self.retained
    }

    fn materialization(&self, bytes: &[u8]) -> Result<u64, ExactFacePartitionError> {
        let control = CodecControl::new(self.control);
        runmat_meshing_surface::ExactFacePartitionEncoding::decoding_storage_bytes(bytes, &control)
            .map_err(|error| {
                control
                    .take_failure()
                    .map_or_else(|| invalid(error.to_string()), control_error)
            })
    }
}

impl ExactFaceMeshReuse for SurfaceFaceReuse<'_> {
    fn restore(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
    ) -> Result<Option<ExactFaceMesh>, ExactFacePartitionError> {
        let key = identity::face(self.scope, boundary, coordinates, self.control)?;
        let Some(mut entry) =
            super::face_records::read(self.store.as_ref(), key, self.control, |bytes| {
                let codec = CodecControl::new(self.control);
                runmat_meshing_surface::ExactFacePartitionEncoding::decoding_storage_bytes(
                    bytes, &codec,
                )
                .map_err(|error| codec.error(error))
            })
            .map_err(store_error)?
        else {
            return Ok(None);
        };
        let body = super::record::body(&entry.bytes, key).map_err(store_error)?;
        if self.materialization(body)? > entry.materialization_resident_bytes {
            return Err(invalid(
                "face cache record exceeds its admitted byte or materialization inventory",
            ));
        }
        let cancellation = CodecControl::new(self.control);
        let mesh = decode_exact_face_reuse_record_with_cancellation(
            body,
            self.topology,
            self.curves,
            boundary,
            &cancellation,
        )
        .map_err(|error| match cancellation.take_failure() {
            Some(error) => control_error(error),
            None => invalid(format!(
                "cached face failed current curve admission: {error}"
            )),
        })?;
        let retained = mesh
            .retained_storage_bytes(self.control)
            .map_err(control_error)?;
        if retained > entry.materialization_resident_bytes {
            return Err(invalid(
                "typed face exceeds admitted materialization inventory",
            ));
        }
        entry.storage.retain_bytes(retained);
        self.hits += 1;
        tracing::info!(event_code="meshing.face_cache.hit", face=?boundary.source_face_id, nodes=mesh.nodes.len(), triangles=mesh.triangles.len(), "original accepted face reused after current-boundary validation");
        Ok(Some(mesh))
    }

    fn retain(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
        accepted: AcceptedExactFace<'_>,
    ) -> Result<(), ExactFacePartitionError> {
        self.retain_borrowed(boundary, coordinates, &accepted)
    }
}

impl SurfaceFaceReuse<'_> {
    pub(super) fn retain_borrowed(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
        accepted: &AcceptedExactFace<'_>,
    ) -> Result<(), ExactFacePartitionError> {
        // The numerical codec bounds the complete record; the shared transport
        // enforces per-entry storage by chunking rather than dropping large faces.
        let maximum = 256 * 1024 * 1024 - super::record::HEADER_BYTES;
        self.control.checkpoint().map_err(control_error)?;
        let cancellation = CodecControl::new(self.control);
        let preparation_bytes = accepted
            .encoding_preparation_storage_bytes(&cancellation)
            .map_err(|error| cancellation.take_failure().map_or(error, control_error))?;
        let _preparation_storage = match runmat_geometry_core::GeometryAllocationReservation::new(
            self.control,
            preparation_bytes,
        ) {
            Ok(reservation) => reservation,
            Err(error) if error.kind == GeometryEvaluationErrorKind::AllocationBudgetExceeded => {
                return Ok(())
            }
            Err(error) => return Err(control_error(error)),
        };
        let prepared = accepted
            .prepare_encoding(&cancellation)
            .map_err(|error| cancellation.take_failure().map_or(error, control_error))?;
        let Some(profile) = prepared
            .bounded_encoding_profile(maximum, &cancellation)
            .map_err(|error| {
                cancellation
                    .take_failure()
                    .map_or_else(|| invalid(error.to_string()), control_error)
            })?
        else {
            return Ok(());
        };
        self.control.checkpoint().map_err(control_error)?;
        let _encoding_storage = match runmat_geometry_core::GeometryAllocationReservation::new(
            self.control,
            profile.resident_workspace_bytes_estimate,
        ) {
            Ok(reservation) => reservation,
            Err(error) if error.kind == GeometryEvaluationErrorKind::AllocationBudgetExceeded => {
                return Ok(())
            }
            Err(error) => return Err(control_error(error)),
        };
        let key = identity::face(self.scope, boundary, coordinates, self.control)?;
        let mut bytes = prepared
            .encode_profiled(&profile, &cancellation)
            .map_err(|error| {
                cancellation
                    .take_failure()
                    .map_or_else(|| invalid(error.to_string()), control_error)
            })?;
        let workspace = self.materialization(&bytes)?;
        super::record::frame(&mut bytes, key);
        if super::face_records::retain(self.store.as_ref(), key, &bytes, workspace, self.control)
            .map_err(store_error)?
        {
            self.retained += 1;
            tracing::info!(event_code="meshing.face_cache.retained", face=?boundary.source_face_id, encoded_bytes=bytes.len(), "original converged face retained independently of its enclosing batch");
        }
        Ok(())
    }
}
