//! Borrowed family dispatch. Solid-only constraints retain their original transfer provenance.
use runmat_analysis_core::{
    AnalysisModel, BoundaryTransferImplementation, StructuralConstraint, StructuralModel,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::{CompiledConstraint, ConstraintPlanError};
use crate::physics::structural::topology::{
    node_space::CanonicalStructuralNodeSpace, CanonicalStructuralTopologyIdentity,
};

pub(super) enum CompilationTopology<'a> {
    Solid(&'a SolverMeshArtifact),
    Explicit {
        structural: &'a StructuralModel,
        nodes: CanonicalStructuralNodeSpace,
    },
    Mixed {
        mesh: &'a SolverMeshArtifact,
        structural: &'a StructuralModel,
        nodes: CanonicalStructuralNodeSpace,
    },
}

impl<'a> CompilationTopology<'a> {
    pub(super) fn derive(
        model: &'a AnalysisModel,
        mesh: Option<&'a SolverMeshArtifact>,
    ) -> Result<Self, ConstraintPlanError> {
        let structural = model
            .structural
            .as_ref()
            .filter(|value| !value.elements.is_empty());
        match (mesh, structural) {
            (Some(mesh), Some(structural)) if !mesh.topology.volume_elements.is_empty() => {
                let nodes = CanonicalStructuralNodeSpace::derive(mesh, structural)
                    .map_err(node_space_error)?;
                Ok(Self::Mixed {
                    mesh,
                    structural,
                    nodes,
                })
            }
            (Some(mesh), _) => {
                mesh.validate().map_err(invalid)?;
                Ok(Self::Solid(mesh))
            }
            (None, Some(structural)) => {
                let nodes = CanonicalStructuralNodeSpace::derive_explicit(structural)
                    .map_err(node_space_error)?;
                Ok(Self::Explicit { structural, nodes })
            }
            (None, None) => Err(invalid(
                "structural constraint compilation requires a solid mesh or explicit connectivity",
            )),
        }
    }

    pub(super) fn identity(&self) -> CanonicalStructuralTopologyIdentity {
        match self {
            Self::Solid(mesh) => CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: mesh.canonical_digest,
            },
            Self::Explicit { nodes, .. } | Self::Mixed { nodes, .. } => nodes.identity,
        }
    }

    pub(super) fn lower(
        &self,
        model: &AnalysisModel,
        constraint: &StructuralConstraint,
    ) -> Result<(CompiledConstraint, BoundaryTransferImplementation), ConstraintPlanError> {
        use super::super::{compile, mixed};
        let (mesh, mixed_topology) = match self {
            Self::Solid(mesh) => (Some(*mesh), None),
            Self::Explicit { structural, nodes } => (None, Some((*structural, nodes))),
            Self::Mixed {
                mesh,
                structural,
                nodes,
            } => (Some(*mesh), Some((*structural, nodes))),
        };
        if matches!(
            constraint.kind,
            runmat_analysis_core::StructuralConstraintKind::Equation(_)
        ) {
            super::super::frame::require_global(constraint)?;
        }
        let mut prepared =
            super::super::prepared::PreparedTargets::derive(constraint, mesh, mixed_topology)?;
        let (compiled, implementation) = if !prepared.requires_mixed() {
            (
                compile::compile_constraint(
                    model,
                    mesh.expect("solid preparation requires a mesh"),
                    constraint,
                    &mut prepared,
                )?,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
        } else {
            let (structural, nodes) =
                mixed_topology.expect("mixed preparation requires node space");
            (
                mixed::compile_node_constraint(
                    model,
                    mesh,
                    structural,
                    nodes,
                    constraint,
                    &mut prepared,
                )?,
                BoundaryTransferImplementation::ExplicitRegionConnectivity,
            )
        };
        prepared.finish()?;
        Ok((compiled, implementation))
    }
}

fn invalid(error: impl std::fmt::Display) -> ConstraintPlanError {
    ConstraintPlanError::Invalid(error.to_string())
}

fn node_space_error(
    error: crate::physics::structural::topology::node_space::StructuralNodeSpaceError,
) -> ConstraintPlanError {
    match error {
        crate::physics::structural::topology::node_space::StructuralNodeSpaceError::Storage(
            error,
        ) => error.into(),
        other => invalid(other),
    }
}
