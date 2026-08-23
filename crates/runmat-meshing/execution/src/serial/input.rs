use crate::{
    PreparedDomainModelInput, PreparedEvidenceInput, PreparedExactGeometryInput,
    PreparedFacetedGeometryInput, PreparedMeshingResultPublication,
};

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedMeshingInput {
    ExactGeometry(Box<PreparedExactGeometryInput>),
    FacetedGeometry(Box<PreparedFacetedGeometryInput>),
    DomainModel(Box<PreparedDomainModelInput>),
    Evidence(Box<PreparedEvidenceInput>),
    StageArtifact(Box<PreparedMeshingResultPublication>),
}

impl PreparedMeshingInput {
    pub const fn exact_geometry(&self) -> Option<&PreparedExactGeometryInput> {
        match self {
            Self::ExactGeometry(input) => Some(input),
            Self::FacetedGeometry(_)
            | Self::DomainModel(_)
            | Self::Evidence(_)
            | Self::StageArtifact(_) => None,
        }
    }

    pub const fn faceted_geometry(&self) -> Option<&PreparedFacetedGeometryInput> {
        match self {
            Self::FacetedGeometry(input) => Some(input),
            Self::ExactGeometry(_)
            | Self::DomainModel(_)
            | Self::Evidence(_)
            | Self::StageArtifact(_) => None,
        }
    }

    pub const fn domain_model(&self) -> Option<&PreparedDomainModelInput> {
        match self {
            Self::DomainModel(input) => Some(input),
            Self::ExactGeometry(_)
            | Self::FacetedGeometry(_)
            | Self::Evidence(_)
            | Self::StageArtifact(_) => None,
        }
    }

    pub const fn evidence(&self) -> Option<&PreparedEvidenceInput> {
        match self {
            Self::Evidence(input) => Some(input),
            Self::ExactGeometry(_)
            | Self::FacetedGeometry(_)
            | Self::DomainModel(_)
            | Self::StageArtifact(_) => None,
        }
    }

    pub const fn stage_artifact(&self) -> Option<&PreparedMeshingResultPublication> {
        match self {
            Self::ExactGeometry(_)
            | Self::FacetedGeometry(_)
            | Self::DomainModel(_)
            | Self::Evidence(_) => None,
            Self::StageArtifact(input) => Some(input),
        }
    }

    pub(super) fn objects(&self) -> &[runmat_execution_artifact::LogicalObject] {
        match self {
            Self::ExactGeometry(input) => &input.geometry_objects().objects,
            Self::FacetedGeometry(input) => &input.geometry_objects().objects,
            Self::DomainModel(input) => &input.domain_model_objects().objects,
            Self::Evidence(input) => &input.evidence_objects().objects,
            Self::StageArtifact(input) => &input.stage_objects().objects,
        }
    }
}
