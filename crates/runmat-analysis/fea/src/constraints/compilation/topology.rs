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
        CompilationInputs::derive(model, mesh)?.construct()
    }

    pub(super) fn identity(&self) -> CanonicalStructuralTopologyIdentity {
        match self {
            Self::Solid(mesh) => CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: mesh.canonical_digest,
            },
            Self::Explicit { nodes, .. } | Self::Mixed { nodes, .. } => nodes.identity,
        }
    }

    #[cfg(test)]
    pub(super) fn lower(
        &self,
        model: &AnalysisModel,
        constraint: &StructuralConstraint,
    ) -> Result<(CompiledConstraint, BoundaryTransferImplementation), ConstraintPlanError> {
        self.lower_admitted(
            model,
            constraint,
            &mut super::CompilationAdmission::new(u64::MAX),
            0,
        )
    }

    pub(super) fn lower_admitted(
        &self,
        model: &AnalysisModel,
        constraint: &StructuralConstraint,
        admission: &mut super::CompilationAdmission,
        retained: u64,
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
        let selection = super::super::prepared::profile::derive(
            constraint,
            mesh,
            mixed_topology.map(|(structural, _)| structural),
        )?;
        admission.require(
            super::ConstraintCompilationPhase::Selection,
            retained,
            selection.total_requested_bytes,
        )?;
        let mut prepared =
            super::super::prepared::PreparedTargets::derive(constraint, mesh, mixed_topology)?;
        let lowering =
            super::lowering_profile::derive(constraint, mesh, mixed_topology, &prepared)?;
        admission.require(
            super::ConstraintCompilationPhase::Lowering,
            super::super::workspace::add(retained, selection.total_requested_bytes)?,
            lowering.total_requested_bytes()?,
        )?;
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

pub(super) enum CompilationInputs<'a> {
    Solid(&'a SolverMeshArtifact),
    Explicit(&'a StructuralModel),
    Mixed {
        mesh: &'a SolverMeshArtifact,
        structural: &'a StructuralModel,
    },
}
impl<'a> CompilationInputs<'a> {
    pub(super) fn derive(
        model: &'a AnalysisModel,
        mesh: Option<&'a SolverMeshArtifact>,
    ) -> Result<Self, ConstraintPlanError> {
        let structural = model
            .structural
            .as_ref()
            .filter(|value| !value.elements.is_empty());
        match (mesh, structural) {
            (Some(mesh), Some(structural))
                if !mesh.topology.volume_elements.is_empty()
                    || !mesh.resolved_request.dimension.is_solid() =>
            {
                Ok(Self::Mixed { mesh, structural })
            }
            (Some(mesh), _) => Ok(Self::Solid(mesh)),
            (None, Some(structural)) => Ok(Self::Explicit(structural)),
            (None, None) => Err(invalid(
                "structural constraint compilation requires a solid mesh or explicit connectivity",
            )),
        }
    }
    pub(super) fn mesh(&self) -> Option<&'a SolverMeshArtifact> {
        match self {
            Self::Solid(mesh) | Self::Mixed { mesh, .. } => Some(mesh),
            Self::Explicit(_) => None,
        }
    }
    pub(super) fn structural(&self) -> Option<&'a StructuralModel> {
        match self {
            Self::Explicit(structural) | Self::Mixed { structural, .. } => Some(structural),
            Self::Solid(_) => None,
        }
    }
    pub(super) fn workspace_bytes(&self) -> Result<u64, ConstraintPlanError> {
        match self {
            Self::Solid(mesh) => mesh.validation_resident_bytes_estimate().map_err(invalid),
            Self::Explicit(_) | Self::Mixed { .. } => {
                crate::physics::structural::topology::node_space::node_space_workspace_profile(
                    self.mesh(),
                    self.structural(),
                )
                .map_err(node_space_error)?
                .total_requested_bytes()
                .map_err(Into::into)
            }
        }
    }
    pub(super) fn construct(self) -> Result<CompilationTopology<'a>, ConstraintPlanError> {
        match self {
            Self::Solid(mesh) => {
                mesh.validate().map_err(invalid)?;
                Ok(CompilationTopology::Solid(mesh))
            }
            Self::Explicit(structural) => {
                let nodes = CanonicalStructuralNodeSpace::derive_explicit(structural)
                    .map_err(node_space_error)?;
                Ok(CompilationTopology::Explicit { structural, nodes })
            }
            Self::Mixed { mesh, structural } => {
                let nodes = CanonicalStructuralNodeSpace::derive(mesh, structural)
                    .map_err(node_space_error)?;
                Ok(CompilationTopology::Mixed {
                    mesh,
                    structural,
                    nodes,
                })
            }
        }
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
