use super::DelaunayVolumeMeshErrorKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunaySolverTopologyErrorKind {
    InvalidOptions,
    InvalidGeometry,
    InvalidMesh,
    InvalidMaterials,
    UnsupportedOrder,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunaySolverTopologyError {
    pub kind: DelaunaySolverTopologyErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunaySolverTopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Delaunay solver topology {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunaySolverTopologyError {}

pub(super) fn failure(
    kind: DelaunaySolverTopologyErrorKind,
    reason: impl Into<String>,
) -> DelaunaySolverTopologyError {
    DelaunaySolverTopologyError {
        kind,
        reason: reason.into(),
    }
}

pub(super) fn volume(
    kind: DelaunayVolumeMeshErrorKind,
    reason: impl Into<String>,
) -> DelaunaySolverTopologyError {
    let mapped = match kind {
        DelaunayVolumeMeshErrorKind::InvalidOptions => {
            DelaunaySolverTopologyErrorKind::InvalidOptions
        }
        DelaunayVolumeMeshErrorKind::InvalidGeometry
        | DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint => {
            DelaunaySolverTopologyErrorKind::InvalidGeometry
        }
        DelaunayVolumeMeshErrorKind::ResourceLimit => {
            DelaunaySolverTopologyErrorKind::ResourceLimit
        }
        DelaunayVolumeMeshErrorKind::Cancelled => DelaunaySolverTopologyErrorKind::Cancelled,
        DelaunayVolumeMeshErrorKind::InvalidTopology
        | DelaunayVolumeMeshErrorKind::InvalidMetric
        | DelaunayVolumeMeshErrorKind::InvalidQuality => {
            DelaunaySolverTopologyErrorKind::InvalidMesh
        }
    };
    failure(mapped, reason)
}

pub(super) fn request(
    error: runmat_meshing_core::MeshingContractError,
) -> DelaunaySolverTopologyError {
    failure(
        DelaunaySolverTopologyErrorKind::InvalidOptions,
        error.to_string(),
    )
}

pub(super) fn solver(
    error: runmat_meshing_core::MeshingContractError,
) -> DelaunaySolverTopologyError {
    failure(
        DelaunaySolverTopologyErrorKind::InvalidMesh,
        error.to_string(),
    )
}
