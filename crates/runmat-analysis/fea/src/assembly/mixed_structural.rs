use runmat_analysis_core::{
    AnalysisModel, BoundaryTransferImplementation, SourceImplementation,
    StructuralFormulationRealization, StructuralOperatorImplementation,
    StructuralRecoveryImplementation,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

use super::{
    constraint_projection::{
        apply_direct_constraints, resolve_constraint_matrix, rotational_row_count,
    },
    direct_constraints::DirectConstraintAssignments,
    dofs::{StructuralDofKind, StructuralDofLayout, StructuralNodeDofSet},
    explicit_connectors::{
        assemble_connectors_with_resolver, resolve_explicit_surface, resolve_projected_surface,
        ConnectorAssemblyContext, ConnectorTargetRole, ResolvedConnectorTarget,
    },
    explicit_elements::{assemble_explicit_elements, ExplicitElementAssembly},
    requested_moment_norm,
    solver_solid::assemble_solver_solid_operators,
    structural_loads::{
        apply as apply_structural_load_plan, compile_mixed as compile_mixed_load_plan,
    },
    structural_source_implementations, validate_structural_inputs, AssemblySummary,
    LinearAssemblyError, StructuralAssemblyError, StructuralProjectionBasis,
};
use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    constraints::CanonicalStructuralConstraintPlan,
    excitation::{RealizedExcitationKind, RealizedExcitations},
    physics::structural::topology::{
        node_space::CanonicalStructuralNodeSpace,
        target::{
            resolve_mixed_structural_target, MixedStructuralTargetError,
            MixedStructuralTargetOwnership,
        },
    },
};

pub(crate) fn mixed_dof_layout(
    topology: &CanonicalStructuralNodeSpace,
    reference_point_count: usize,
) -> StructuralDofLayout {
    StructuralDofLayout::from_node_sets(
        topology
            .nodes
            .iter()
            .map(|node| {
                if node.owns_explicit_rotation {
                    StructuralNodeDofSet::translational_rotational()
                } else {
                    StructuralNodeDofSet::translational()
                }
            })
            .collect(),
    )
    .with_reference_points(reference_point_count)
}

pub(super) fn assemble_mixed_structural_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    supplied_constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    let structural = model.structural.as_ref().ok_or({
        LinearAssemblyError::Structural(StructuralAssemblyError::MissingDiscretization)
    })?;
    let topology = CanonicalStructuralNodeSpace::derive(mesh, structural).map_err(|error| {
        LinearAssemblyError::Structural(StructuralAssemblyError::MixedTopology(error.to_string()))
    })?;
    let compiled_constraint_plan;
    let constraint_plan = if let Some(plan) = supplied_constraint_plan {
        plan.validate_against(model, mesh)
            .map_err(LinearAssemblyError::ConstraintPlan)?;
        plan
    } else {
        compiled_constraint_plan = CanonicalStructuralConstraintPlan::compile(model, mesh)
            .map_err(LinearAssemblyError::ConstraintPlan)?;
        &compiled_constraint_plan
    };
    let layout = mixed_dof_layout(&topology, constraint_plan.reference_points.len());
    let solid_properties =
        crate::properties::compile_solid_property_map(model, mesh).map_err(|error| {
            LinearAssemblyError::SolidOperator(
                super::solver_solid::SolverSolidAssemblyError::InvalidMaterialCoverage(
                    error.to_string(),
                ),
            )
        })?;
    let solid = assemble_solver_solid_operators(mesh, &solid_properties)
        .map_err(LinearAssemblyError::SolidOperator)?;
    let ExplicitElementAssembly {
        stiffness: explicit_stiffness,
        damping: explicit_damping,
        mass: explicit_mass,
        beam_recovery,
        shell_recovery,
        mut formulations,
        beam_count,
        shell_count,
    } = assemble_explicit_elements(
        structural,
        &solid_properties,
        &layout,
        &topology.explicit_node_indices,
    )
    .map_err(LinearAssemblyError::Structural)?;
    let solid_stiffness = embed_solid_matrix(&solid.stiffness, &layout, topology.solid_node_count)?;
    let solid_mass =
        embed_solid_matrix(&solid.consistent_mass, &layout, topology.solid_node_count)?;
    let solid_damping = embed_solid_matrix(&solid.damping, &layout, topology.solid_node_count)?;
    let element_stiffness = sum(
        &solid_stiffness,
        &explicit_stiffness,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )?;
    let mass = sum(
        &solid_mass,
        &explicit_mass,
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )?;
    let element_damping = sum(
        &solid_damping,
        &explicit_damping,
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )?;
    let mut connector_stiffness_assembler = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let mut connector_damping_assembler = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let connector_projection =
        crate::topology::AnalysisProjection::derive(mesh).map_err(|error| {
            LinearAssemblyError::Structural(StructuralAssemblyError::MixedTopology(
                error.to_string(),
            ))
        })?;
    let connector_coordinates_m = topology
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let connector_node_identities = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<Vec<_>>();
    let connector_summary = assemble_connectors_with_resolver(
        &model.connectors,
        ConnectorAssemblyContext {
            topology_identity: topology.identity,
            layout: &layout,
            coordinates_m: &connector_coordinates_m,
            node_identities: &connector_node_identities,
        },
        &mut connector_stiffness_assembler,
        &mut connector_damping_assembler,
        |connector, target, role| {
            let ownership =
                resolve_target(&connector.connector_id, target, mesh, structural, &topology)?;
            match role {
                ConnectorTargetRole::Point => Ok(ResolvedConnectorTarget {
                    node_indices: ownership.canonical_nodes().to_vec(),
                    surface_samples: Vec::new(),
                }),
                ConnectorTargetRole::Surface => match ownership {
                    MixedStructuralTargetOwnership::Solid { .. }
                    | MixedStructuralTargetOwnership::Shared { .. } => {
                        resolve_projected_surface(connector, target, &connector_projection)
                    }
                    MixedStructuralTargetOwnership::Explicit { .. } => {
                        resolve_explicit_surface(connector, target, structural, &topology)
                    }
                },
            }
        },
    )
    .map_err(LinearAssemblyError::Structural)?;
    let connector_stiffness = connector_stiffness_assembler
        .finish()
        .map_err(sparse_error)?;
    let connector_damping = connector_damping_assembler.finish().map_err(sparse_error)?;
    let stiffness = if connector_summary.stiffness_count == 0 {
        element_stiffness.clone()
    } else {
        sum(
            &element_stiffness,
            &connector_stiffness,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )?
    };
    let damping = if connector_summary.damping_count == 0 {
        element_damping
    } else {
        sum(
            &element_damping,
            &connector_damping,
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )?
    };
    let solid_formulation = match mesh.resolved_request.element_order {
        ElementOrder::Tet4 => StructuralFormulationRealization::new(
            StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet4ConstantStrain),
        ),
        ElementOrder::Tet10 => StructuralFormulationRealization::new(
            StructuralOperatorImplementation::Tet10LinearElasticity,
            Some(StructuralRecoveryImplementation::Tet10VolumeAverageStrain),
        ),
    };
    formulations.insert(0, solid_formulation);
    let dof_count = layout.total_dof_count();
    let operators = CanonicalRealOperators::try_new(stiffness, mass, damping)
        .map_err(|error| LinearAssemblyError::SparseOperator(error.to_string()))?;
    let mut rhs = vec![0.0; dof_count];
    let load_ownership = excitations
        .iter()
        .filter(|load| is_structural_load(&load.kind))
        .map(|load| {
            let ownership = match &load.target {
                crate::excitation::RealizedExcitationTarget::Region(target) => {
                    resolve_target(&load.excitation_id.0, target, mesh, structural, &topology)?
                }
                _ => MixedStructuralTargetOwnership::Solid {
                    canonical_nodes: Vec::new(),
                },
            };
            Ok((load.excitation_id.0.clone(), ownership))
        })
        .collect::<Result<Vec<_>, StructuralAssemblyError>>()
        .map_err(LinearAssemblyError::Structural)?;
    let explicit_load_ids = load_ownership
        .iter()
        .filter_map(|(load_id, ownership)| {
            matches!(
                ownership,
                MixedStructuralTargetOwnership::Explicit { .. }
                    | MixedStructuralTargetOwnership::Shared { .. }
            )
            .then_some(load_id.as_str())
        })
        .collect::<std::collections::BTreeSet<_>>();
    let explicit_excitations =
        excitations.filtered(|load| explicit_load_ids.contains(load.excitation_id.0.as_str()));
    let solid_excitations =
        excitations.filtered(|load| !explicit_load_ids.contains(load.excitation_id.0.as_str()));
    // The canonical constraint compiler owns target and relation validation. Projection-specific
    // input checks here are limited to sources so assembly does not maintain a second constraint
    // capability matrix.
    let mut source_validation_model = model.clone();
    source_validation_model
        .constraints
        .retain(|constraint| constraint.structural().is_none());
    validate_structural_inputs(
        &source_validation_model,
        &explicit_excitations,
        StructuralProjectionBasis::ExplicitConnectivity,
    )
    .map_err(LinearAssemblyError::Structural)?;
    validate_structural_inputs(
        &source_validation_model,
        &solid_excitations,
        StructuralProjectionBasis::CanonicalSolid,
    )
    .map_err(LinearAssemblyError::Structural)?;
    let mut direct_constraints = DirectConstraintAssignments::new(&layout);
    let source_projection = compile_mixed_load_plan(
        structural,
        &solid_properties,
        excitations,
        mesh,
        constraint_plan,
        &topology,
        &layout,
    )
    .map_err(|error| {
        LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(error.to_string()))
    })?;
    apply_structural_load_plan(
        &source_projection.plan,
        &topology,
        &constraint_plan.reference_points,
        &layout,
        &mut rhs,
    )
    .map_err(|error| {
        LinearAssemblyError::Structural(StructuralAssemblyError::InvalidLoadPlan(error.to_string()))
    })?;
    let spatial_projection_evidence = source_projection
        .plan
        .entries
        .iter()
        .flat_map(|entry| entry.evidence.iter().cloned())
        .collect();
    let structural_load_plan_identity = Some(source_projection.plan.plan_digest);
    let direct_rotational_moment_load_count = source_projection.direct_rotational_moment_count;
    let structural_moment_projection = source_projection.moment_projection;
    let physical_load = rhs.clone();
    let has_general_rows =
        constraint_plan.enforcement != crate::constraints::ConstraintEnforcement::DirectElimination;
    let resolved_constraint_matrix = resolve_constraint_matrix(constraint_plan, &topology, &layout)
        .map_err(LinearAssemblyError::ConstraintAlgebra)?;
    let (
        constrained,
        constrained_dof_count,
        rotational_constraint_count,
        constraint_matrix,
        affine_constraint_reduction,
    ) = if has_general_rows {
        let reduction = if constraint_plan.enforcement
            == crate::constraints::ConstraintEnforcement::LagrangeMultiplier
        {
            None
        } else {
            Some(
                resolved_constraint_matrix
                    .affine_reduction()
                    .map_err(LinearAssemblyError::ConstraintAlgebra)?,
            )
        };
        (
            vec![false; dof_count],
            resolved_constraint_matrix.row_count(),
            rotational_row_count(constraint_plan),
            Some(resolved_constraint_matrix.clone()),
            reduction,
        )
    } else {
        apply_direct_constraints(constraint_plan, &topology, &mut direct_constraints)
            .map_err(LinearAssemblyError::Structural)?;
        let constrained_dof_count = direct_constraints.constrained_dof_count();
        let rotational_constraint_count = direct_constraints.rotational_constraint_count();
        let constrained = direct_constraints
            .apply(&mut rhs)
            .map_err(LinearAssemblyError::Structural)?;
        (
            constrained,
            constrained_dof_count,
            rotational_constraint_count,
            None,
            None,
        )
    };
    let nullspace_evidence =
        match crate::algebra::NullspaceEliminationEvidence::prove_mixed_structural(
            crate::algebra::MixedStructuralNullspaceInput {
                mesh,
                structural,
                topology: &topology,
                layout: &layout,
                element_operator: &element_stiffness,
                connector_operator: (connector_summary.stiffness_count > 0)
                    .then_some(&connector_stiffness),
                combined_operator: operators.stiffness(),
                constraints: &resolved_constraint_matrix,
                constraint_plan,
                constrained_dofs: &constrained,
            },
        ) {
            Ok(evidence) => Some(evidence),
            Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
            Err(error) => {
                return Err(LinearAssemblyError::Structural(
                    StructuralAssemblyError::SparseOperator(error.to_string()),
                ))
            }
        };
    let operator = AssembledRealSystem::try_new_with_nullspace_evidence(
        operators,
        constrained,
        rhs,
        nullspace_evidence,
    )
    .map_err(|error| LinearAssemblyError::SparseOperator(error.to_string()))?;

    Ok(AssemblySummary {
        dof_count,
        structural_node_count: layout.node_count(),
        structural_translational_dof_count: layout.translational_dof_count(),
        structural_rotational_dof_count: layout.rotational_dof_count(),
        structural_rotation_node_count: layout.rotation_node_count(),
        structural_explicit_node_indices: topology.explicit_node_indices.clone(),
        structural_load_plan_identity,
        structural_moment_load_count: excitations
            .iter()
            .filter(|load| {
                matches!(
                    load.kind,
                    crate::excitation::RealizedExcitationKind::Moment(_)
                )
            })
            .count(),
        structural_direct_rotational_moment_load_count: direct_rotational_moment_load_count,
        structural_requested_moment_norm_n_m: requested_moment_norm(excitations),
        structural_moment_projection,
        spatial_projection_evidence,
        structural_rotational_constraint_count: rotational_constraint_count,
        structural_beam_element_count: beam_count,
        structural_shell_element_count: shell_count,
        structural_connector_stiffness_count: connector_summary.stiffness_count,
        structural_connector_damping_count: connector_summary.damping_count,
        structural_connector_plan: Some(connector_summary.plan),
        structural_solid_element_count: mesh.topology.volume_elements.len(),
        structural_solid_total_mass_kg: solid.total_mass_kg,
        structural_solid_recovery: super::solid_recovery_from_solver_mesh(mesh, &solid_properties)
            .map_err(LinearAssemblyError::SolidOperator)?,
        structural_dof_layout: layout,
        structural_beam_recovery: beam_recovery,
        structural_shell_recovery: shell_recovery,
        structural_formulations: formulations,
        structural_property_map_identity: Some(solid_properties.identity.clone()),
        structural_sources: merge_sources(
            structural_source_implementations(
                &solid_excitations,
                StructuralProjectionBasis::CanonicalSolid,
            ),
            structural_source_implementations(
                &explicit_excitations,
                StructuralProjectionBasis::ExplicitConnectivity,
            ),
        ),
        structural_boundary_transfers: constraint_boundary_implementations(constraint_plan),
        constrained_dof_count,
        physical_load,
        constraint_matrix,
        affine_constraint_reduction,
        load_count: excitations.len(),
        structural_material: super::structural_material_summary(model),
        thermo_mechanical: None,
        electro_thermal: None,
        operator,
    })
}

fn constraint_boundary_implementations(
    plan: &CanonicalStructuralConstraintPlan,
) -> Vec<BoundaryTransferImplementation> {
    let implementations = [
        BoundaryTransferImplementation::CanonicalBoundaryProjection,
        BoundaryTransferImplementation::ExplicitRegionConnectivity,
    ]
    .into_iter()
    .filter(|implementation| {
        plan.groups
            .iter()
            .any(|group| group.implementation == *implementation)
    })
    .collect::<Vec<_>>();
    if implementations.is_empty() {
        vec![BoundaryTransferImplementation::NotApplicable]
    } else {
        implementations
    }
}

fn resolve_target(
    target_id: &str,
    target: &runmat_analysis_core::RegionTarget,
    mesh: &SolverMeshArtifact,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
) -> Result<MixedStructuralTargetOwnership, StructuralAssemblyError> {
    resolve_mixed_structural_target(target, mesh, structural, topology).map_err(|error| match error
    {
        MixedStructuralTargetError::Unmapped { selector } => {
            StructuralAssemblyError::UnmappedMixedTarget {
                target_id: target_id.into(),
                selector,
            }
        }
        MixedStructuralTargetError::Ambiguous {
            selector,
            solid_node_count,
            explicit_node_count,
        } => StructuralAssemblyError::AmbiguousMixedTarget {
            target_id: target_id.into(),
            selector,
            solid_node_count,
            explicit_node_count,
        },
        MixedStructuralTargetError::InvalidTopology(message) => {
            StructuralAssemblyError::MixedTopology(message)
        }
    })
}

fn merge_sources(
    left: Vec<SourceImplementation>,
    right: Vec<SourceImplementation>,
) -> Vec<SourceImplementation> {
    merge_implementations(left, right, SourceImplementation::NotApplicable)
}

fn merge_implementations<T: Copy + PartialEq>(
    left: Vec<T>,
    right: Vec<T>,
    not_applicable: T,
) -> Vec<T> {
    let mut merged = Vec::new();
    for implementation in left.into_iter().chain(right) {
        if implementation != not_applicable && !merged.contains(&implementation) {
            merged.push(implementation);
        }
    }
    if merged.is_empty() {
        merged.push(not_applicable);
    }
    merged
}

fn is_structural_load(kind: &RealizedExcitationKind) -> bool {
    matches!(
        kind,
        RealizedExcitationKind::Force(_)
            | RealizedExcitationKind::Moment(_)
            | RealizedExcitationKind::Traction(_)
            | RealizedExcitationKind::Pressure(_)
            | RealizedExcitationKind::BodyForce(_)
    )
}

fn embed_solid_matrix(
    matrix: &RealSparseMatrix,
    layout: &StructuralDofLayout,
    solid_node_count: usize,
) -> Result<RealSparseMatrix, LinearAssemblyError> {
    let row_map = solid_row_map(layout, solid_node_count)?;
    if matrix.rows() != row_map.len() || matrix.columns() != row_map.len() {
        return Err(LinearAssemblyError::SparseOperator(
            "solid operator shape does not match canonical solid translations".into(),
        ));
    }
    let mut assembler =
        RealSparseAssembler::square(layout.total_dof_count(), matrix.role(), matrix.properties());
    for contribution in matrix.contributions() {
        assembler.add_contribution(contribution.clone());
    }
    for (source_row, target_row) in row_map.iter().copied().enumerate() {
        for entry in matrix.pattern().row_range(source_row) {
            assembler.add(
                target_row,
                row_map[matrix.pattern().column(entry)],
                matrix.values()[entry],
            );
        }
    }
    assembler.finish().map_err(sparse_error)
}

fn solid_row_map(
    layout: &StructuralDofLayout,
    solid_node_count: usize,
) -> Result<Vec<usize>, LinearAssemblyError> {
    let mut rows = Vec::with_capacity(solid_node_count.saturating_mul(3));
    for node in 0..solid_node_count {
        for kind in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ] {
            rows.push(layout.index(node, kind).ok_or_else(|| {
                LinearAssemblyError::SparseOperator(
                    "canonical solid node is missing a translation DOF".into(),
                )
            })?);
        }
    }
    Ok(rows)
}

fn sum(
    left: &RealSparseMatrix,
    right: &RealSparseMatrix,
    role: OperatorRole,
    properties: MatrixProperties,
) -> Result<RealSparseMatrix, LinearAssemblyError> {
    RealSparseMatrix::scaled_sum(&[(1.0, left), (1.0, right)], role, properties)
        .map_err(sparse_error)
}

fn sparse_error(error: impl std::fmt::Debug) -> LinearAssemblyError {
    LinearAssemblyError::SparseOperator(format!("{error:?}"))
}

#[cfg(test)]
mod tests {
    use runmat_execution::Digest;
    use runmat_meshing_core::StableDigest;

    use super::*;
    use crate::{
        assembly::dofs::StructuralDofKind,
        physics::structural::topology::{
            node_space::CanonicalStructuralNode, CanonicalStructuralTopologyIdentity,
        },
    };

    #[test]
    fn attached_and_unattached_explicit_nodes_share_only_authored_translations() {
        let topology = CanonicalStructuralNodeSpace {
            nodes: vec![
                node(true, true, 0),
                node(true, false, 1),
                node(true, false, 2),
                node(true, false, 3),
                node(false, true, 4),
            ],
            solid_node_count: 4,
            explicit_node_indices: vec![0, 4],
            identity: CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: Digest::sha256(b"mixed-topology"),
            },
        };

        let layout = mixed_dof_layout(&topology, 0);

        assert_eq!(layout.node_count(), 5);
        assert_eq!(layout.translational_dof_count(), 15);
        assert_eq!(layout.rotational_dof_count(), 6);
        assert_eq!(layout.total_dof_count(), 21);
        assert_eq!(layout.rotation_node_count(), 2);
        assert!(layout.index(0, StructuralDofKind::Rx).is_some());
        assert!(layout.index(1, StructuralDofKind::Rx).is_none());
        assert!(layout.index(4, StructuralDofKind::Rx).is_some());
        assert_ne!(
            layout.index(0, StructuralDofKind::Ux),
            layout.index(4, StructuralDofKind::Ux)
        );
    }

    fn node(
        owns_solid_translation: bool,
        owns_explicit_rotation: bool,
        identity_byte: u8,
    ) -> CanonicalStructuralNode {
        CanonicalStructuralNode {
            identity: StableDigest::from_bytes([identity_byte; 32]),
            coordinates_m: [f64::from(identity_byte), 0.0, 0.0],
            owns_solid_translation,
            owns_explicit_rotation,
        }
    }
}
