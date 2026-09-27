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
                let nodes =
                    CanonicalStructuralNodeSpace::derive(mesh, structural).map_err(invalid)?;
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
                let nodes =
                    CanonicalStructuralNodeSpace::derive_explicit(structural).map_err(invalid)?;
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
        let solid = match self {
            Self::Solid(mesh) => Some(*mesh),
            Self::Mixed {
                mesh,
                structural,
                nodes,
            } if !mixed::requires_mixed_compilation(constraint, mesh, structural, nodes)? => {
                Some(*mesh)
            }
            _ => None,
        };
        if let Some(mesh) = solid {
            let (targets, rows, pairings, coupling) =
                compile::compile_constraint(model, mesh, constraint)?;
            return Ok((
                CompiledConstraint {
                    targets,
                    rows,
                    pairings,
                    coupling,
                },
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            ));
        }
        let compiled = match self {
            Self::Explicit { structural, nodes } => {
                mixed::compile_node_constraint(model, None, structural, nodes, constraint)?
            }
            Self::Mixed {
                mesh,
                structural,
                nodes,
            } => mixed::compile_node_constraint(model, Some(mesh), structural, nodes, constraint)?,
            Self::Solid(_) => unreachable!("solid topology was lowered above"),
        };
        Ok((
            compiled,
            BoundaryTransferImplementation::ExplicitRegionConnectivity,
        ))
    }
}

fn invalid(error: impl std::fmt::Display) -> ConstraintPlanError {
    ConstraintPlanError::Invalid(error.to_string())
}
