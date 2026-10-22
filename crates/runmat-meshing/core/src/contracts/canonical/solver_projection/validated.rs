//! Process-local immutable authority; wire decoding never constructs these types.
use super::*;

/// What order elevation did beyond curving: Tet10 edges it kept straight because no valid curved
/// placement could be certified. Recorded in the stage evidence; not part of the mesh encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SolverElevationReport {
    pub straightened_elements: u64,
    pub straightened_edges: u64,
}

#[derive(Debug)]
pub struct ValidatedSolverMeshTopology {
    topology: SolverMeshTopology,
    request: MeshingRequest,
    elevation: SolverElevationReport,
}
impl ValidatedSolverMeshTopology {
    pub fn new(
        topology: SolverMeshTopology,
        request: MeshingRequest,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, MeshingContractError> {
        request.validate()?;
        topology.validate(&request, cancellation)?;
        Ok(Self {
            topology,
            request,
            elevation: SolverElevationReport::default(),
        })
    }
    pub fn with_elevation_report(mut self, elevation: SolverElevationReport) -> Self {
        self.elevation = elevation;
        self
    }
    pub fn elevation_report(&self) -> SolverElevationReport {
        self.elevation
    }
    pub fn into_topology(self) -> SolverMeshTopology {
        self.topology
    }
    pub fn into_projection(
        self,
        geometry: GeometryRevisionRef,
    ) -> Result<ValidatedSolverMeshProjection, MeshingContractError> {
        geometry.validate()?;
        Ok(ValidatedSolverMeshProjection(SolverMeshProjection {
            schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
            geometry,
            resolved_request: self.request,
            topology: self.topology,
        }))
    }
}

/// Owned validated topology bound to its original request, with immutable access
/// only. Fresh encoding reuses that check; ordinary/cold APIs validate independently.
#[derive(Debug)]
pub struct ValidatedSolverMeshProjection(SolverMeshProjection);
impl std::ops::Deref for ValidatedSolverMeshProjection {
    type Target = SolverMeshProjection;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ValidatedSolverMeshProjection {
    pub fn encoding_profile(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, MeshingContractError> {
        runmat_canonical_codec::profile_canonical_encoding_with_control(
            &self.0,
            runmat_canonical_codec::CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
            control,
        )
        .map_err(|error| {
            MeshingContractError::invalid("solver projection encoding resources", error.to_string())
        })
    }
    pub fn encode(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, MeshingContractError> {
        super::super::codec::encode_contract_with_cancellation(
            <SolverMeshProjection as CanonicalMeshingContract>::DOMAIN,
            &self.0,
            <SolverMeshProjection as CanonicalMeshingContract>::LIMITS,
            cancellation,
        )
    }
    pub fn into_projection(self) -> SolverMeshProjection {
        self.0
    }
}
