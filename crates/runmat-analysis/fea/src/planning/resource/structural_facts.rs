use runmat_analysis_core::AnalysisModel;
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::{dofs::StructuralDofKind, mixed_dof_layout, AssemblySummary},
    constraints::{
        CanonicalStructuralConstraintPlan, CanonicalStructuralDofOwner,
        CanonicalStructuralTopologyIdentity, ResolvedConstraintMatrix,
    },
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
    topology::AnalysisProjection,
};

use super::{
    ConstraintResourceFacts, ResourcePlanningError, SparseResourceFacts,
    StructuralSymbolicOperators, TopologyResourceFacts,
};

/// Assembly-free structural facts shared by every task-specific resource projection.
pub(super) struct StructuralResourceFacts {
    pub topology_digest: Digest,
    pub sparse: SparseResourceFacts,
    pub constraints: ConstraintResourceFacts,
    pub constraint_matrix: ResolvedConstraintMatrix,
    pub symbolic: StructuralSymbolicOperators,
    pub topology: TopologyResourceFacts,
    pub connector_assembly_buffer_bytes: u64,
    pub connector_recovery_retained_bytes: u64,
    pub connector_recovery_value_count: u64,
    pub recovery_node_count: usize,
    pub recovery_element_count: usize,
    pub recovery_solid_element_count: usize,
    pub recovery_beam_element_count: usize,
    pub recovery_shell_element_count: usize,
    pub recovery_has_rotational_dofs: bool,
}

/// Metered projection work for one structural resource-fact derivation.
/// Includes the connector realization used by symbolic planning. Static solve
/// candidates share these facts; selecting a fallback does not derive another
/// projection. This quote does not complete any work.
pub fn structural_resource_derivation_work_upper_bound(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<runmat_simulation_usage::OperationCount, ResourcePlanningError> {
    let projection = mesh
        .map(AnalysisProjection::derivation_work_upper_bound)
        .transpose()
        .map(|work| work.unwrap_or(runmat_simulation_usage::OperationCount::ZERO))
        .map_err(ResourcePlanningError::from_projection_error)?;
    let connectors =
        crate::assembly::explicit_connectors::connector_plan_projection_work_upper_bound(
            model, mesh,
        )
        .map_err(ResourcePlanningError::from_projection_error)?;
    projection
        .checked_add(connectors)
        .map_err(|cause| ResourcePlanningError::Accounting(cause.into()))
}

pub(super) fn derive_structural_resource_facts(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
) -> Result<StructuralResourceFacts, ResourcePlanningError> {
    let topology_digest = constraints.topology.discretization_digest();
    let projection = mesh
        .map(AnalysisProjection::derive)
        .transpose()
        .map_err(ResourcePlanningError::from_projection_error)?;
    let symbolic = match constraints.topology {
        CanonicalStructuralTopologyIdentity::Explicit { .. } => {
            StructuralSymbolicOperators::derive_explicit(constraints, model)?
        }
        CanonicalStructuralTopologyIdentity::Solid { .. }
        | CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => {
            let mesh = mesh.ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "solid structural resource planning requires a solver mesh".into(),
                )
            })?;
            let projection = projection.as_ref().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "solid structural resource planning requires an analysis projection".into(),
                )
            })?;
            StructuralSymbolicOperators::derive(projection, constraints, model, mesh)?
        }
    };
    let sparse = SparseResourceFacts::from_patterns(symbolic.patterns())?;
    let constraint_matrix = resolve_constraint_matrix(model, mesh, constraints)?;
    let mixed_topology = if matches!(
        constraints.topology,
        CanonicalStructuralTopologyIdentity::ConformalMixed { .. }
    ) {
        let structural = model.structural.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "mixed resource planning requires explicit structural topology".into(),
            )
        })?;
        Some((
            structural,
            CanonicalStructuralNodeSpace::derive(
                mesh.ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "mixed resource planning requires a solver mesh".into(),
                    )
                })?,
                structural,
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        ))
    } else {
        None
    };
    let recovery_node_count = mixed_topology
        .as_ref()
        .map(|(_, topology)| topology.nodes.len())
        .or_else(|| {
            model
                .structural
                .as_ref()
                .filter(|structural| !structural.elements.is_empty())
                .map(|structural| structural.nodes.len())
        })
        .or_else(|| projection.as_ref().map(|projection| projection.nodes.len()))
        .unwrap_or(0);
    let recovery_element_count = projection
        .as_ref()
        .map(|projection| projection.elements.len())
        .unwrap_or(0)
        .checked_add(
            model
                .structural
                .as_ref()
                .map(|structural| structural.elements.len())
                .unwrap_or(0),
        )
        .ok_or(ResourcePlanningError::Overflow(
            "structural recovery element count",
        ))?;
    let recovery_solid_element_count = projection
        .as_ref()
        .map(|projection| projection.elements.len())
        .unwrap_or(0);
    let (recovery_beam_element_count, recovery_shell_element_count) = model
        .structural
        .as_ref()
        .map(|structural| {
            structural
                .elements
                .iter()
                .fold((0_usize, 0_usize), |counts, element| match element.kind {
                    runmat_analysis_core::StructuralElementKind::Beam(_) => {
                        (counts.0 + 1, counts.1)
                    }
                    runmat_analysis_core::StructuralElementKind::Shell(_) => {
                        (counts.0, counts.1 + 1)
                    }
                })
        })
        .unwrap_or_default();
    let topology = if let Some((structural, topology)) = mixed_topology.as_ref() {
        TopologyResourceFacts::from_mixed_projection(
            projection.as_ref().ok_or_else(|| {
                ResourcePlanningError::Invalid("mixed resource planning lost its projection".into())
            })?,
            structural,
            topology,
        )?
    } else if matches!(
        constraints.topology,
        CanonicalStructuralTopologyIdentity::Explicit { .. }
    ) {
        let structural = model.structural.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "explicit resource planning requires structural topology".into(),
            )
        })?;
        let topology = CanonicalStructuralNodeSpace::derive_explicit(structural)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        TopologyResourceFacts::from_explicit(structural, &topology)?
    } else {
        TopologyResourceFacts::from_projection(projection.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid("solid resource planning lost its projection".into())
        })?)?
    };
    let connector_buffers = super::connector_buffers::estimate_connector_buffers(
        model,
        projection.as_ref(),
        topology.nodes.0,
    )?;
    Ok(StructuralResourceFacts {
        topology_digest,
        sparse,
        constraints: ConstraintResourceFacts::from_plan(constraints)?,
        constraint_matrix,
        symbolic,
        topology,
        connector_assembly_buffer_bytes: connector_buffers.assembly_peak_bytes,
        connector_recovery_retained_bytes: connector_buffers.recovery_retained_bytes,
        connector_recovery_value_count: connector_buffers.recovery_value_count,
        recovery_node_count,
        recovery_element_count,
        recovery_solid_element_count,
        recovery_beam_element_count,
        recovery_shell_element_count,
        recovery_has_rotational_dofs: recovery_beam_element_count > 0
            || recovery_shell_element_count > 0,
    })
}

/// The numerical matrices must fit the exact row/column graph promised by the preassembly
/// structural planner. Re-derive from authoritative inputs because assembly cannot assert it.
/// Returns the planned reduced pattern `T' S T` of the planned stiffness graph `S`: the exact
/// support an admitted affine solve factors, so its ordering and work repeat the plan's.
pub(crate) fn verify_assembled_structural_storage(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    summary: &AssemblySummary,
) -> Result<Option<crate::algebra::SparsePattern>, ResourcePlanningError> {
    let facts = derive_structural_resource_facts(model, mesh, constraints)?;
    facts
        .symbolic
        .verify_assembled(summary.operator.operators())?;
    // A constraint matrix without an affine reduction is never solved by affine elimination.
    let Ok(reduction) = facts.constraint_matrix.affine_reduction() else {
        return Ok(None);
    };
    reduction
        .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
        .map(Some)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

fn resolve_constraint_matrix(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
) -> Result<ResolvedConstraintMatrix, ResourcePlanningError> {
    if matches!(
        constraints.topology,
        CanonicalStructuralTopologyIdentity::Solid { .. }
    ) {
        let mesh = mesh.ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "solid constraint planning requires a solver mesh".into(),
            )
        })?;
        let node_identities = mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.stable_identity)
            .collect::<Vec<_>>();
        return ResolvedConstraintMatrix::from_solid_plan(constraints, &node_identities)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()));
    }
    let structural = model.structural.as_ref().ok_or_else(|| {
        ResourcePlanningError::Invalid(
            "mixed constraint planning requires explicit structural topology".into(),
        )
    })?;
    let topology = match constraints.topology {
        CanonicalStructuralTopologyIdentity::Explicit { .. } => {
            CanonicalStructuralNodeSpace::derive_explicit(structural)
        }
        CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => {
            CanonicalStructuralNodeSpace::derive(
                mesh.ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "mixed constraint planning requires a solver mesh".into(),
                    )
                })?,
                structural,
            )
        }
        CanonicalStructuralTopologyIdentity::Solid { .. } => unreachable!(),
    }
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let layout = mixed_dof_layout(&topology, constraints.reference_points.len());
    let nodes = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    ResolvedConstraintMatrix::from_plan(constraints, layout.total_dof_count(), |owner, dof| {
        match owner {
            CanonicalStructuralDofOwner::StructuralNode(identity) => nodes
                .get(&identity)
                .and_then(|node| layout.index(*node, StructuralDofKind::from_constraint(dof))),
            CanonicalStructuralDofOwner::ReferencePoint(identity) => constraints
                .reference_points
                .iter()
                .position(|point| point.identity == identity)
                .and_then(|index| {
                    layout.reference_point_index(index, StructuralDofKind::from_constraint(dof))
                }),
        }
    })
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}
