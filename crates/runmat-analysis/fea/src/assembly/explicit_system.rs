use super::{
    apply_direct_constraints, apply_structural_load_plan, assemble_explicit_connectors,
    assemble_explicit_elements_with_owner, compile_explicit_load_plan, requested_moment_norm,
    resolve_constraint_matrix, rotational_row_count, structural_material_summary,
    structural_source_implementations, validate_explicit_topology, validate_structural_inputs,
    AnalysisModel, AssembledRealSystem, AssemblySummary, BoundaryTransferImplementation,
    CanonicalRealOperators, CanonicalStructuralConstraintPlan, ConstraintEnforcement,
    DirectConstraintAssignments, ExplicitElementAssembly, MatrixProperties,
    NullspaceEliminationEvidence, OperatorRole, RealSparseAssembler, RealSparseMatrix,
    RealizedExcitationKind, RealizedExcitations, StructuralAssemblyError, StructuralDofLayout,
    StructuralNodeDofSet, StructuralProjectionBasis,
};

pub(super) async fn assemble_explicit_structural_system_with_owner<O, F, A>(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    supplied_constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    local_execution: crate::planning::resource::LocalExecutionPlan,
    before: O,
    awaited_elements: bool,
) -> Result<AssemblySummary, A>
where
    O: FnMut(
        super::explicit_elements::ExplicitCapacityPhase,
        runmat_simulation_usage::OperationCount,
    ) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<StructuralAssemblyError> + Send,
{
    let structural = model
        .structural
        .as_ref()
        .ok_or(StructuralAssemblyError::MissingDiscretization)?;
    validate_structural_inputs(
        model,
        excitations,
        StructuralProjectionBasis::ExplicitConnectivity,
    )?;
    let structural_material = structural_material_summary(model);
    let node_count = structural.nodes.len();
    let canonical_node_indices = (0..node_count).collect::<Vec<_>>();
    validate_explicit_topology(structural, &canonical_node_indices)?;
    let topology = crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace::derive_explicit(structural)
        .map_err(StructuralAssemblyError::from)?;
    let compiled_constraint_plan;
    let constraint_plan = if let Some(plan) = supplied_constraint_plan {
        plan.validate_against_explicit(model)
            .map_err(StructuralAssemblyError::from)?;
        plan
    } else {
        compiled_constraint_plan =
            CanonicalStructuralConstraintPlan::compile_for_model(model, None)
                .map_err(StructuralAssemblyError::from)?;
        &compiled_constraint_plan
    };
    let structural_dof_layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational();
        node_count
    ])
    .with_reference_points(constraint_plan.reference_points.len());
    let dof_count = structural_dof_layout.total_dof_count();
    let properties = crate::properties::compile_property_map(model, None)
        .map_err(map_explicit_property_error)?;
    let ExplicitElementAssembly {
        stiffness: element_stiffness,
        damping: element_damping,
        mass,
        beam_recovery: structural_beam_recovery,
        shell_recovery: structural_shell_recovery,
        formulations: structural_formulations,
        beam_count,
        shell_count,
    } = assemble_explicit_elements_with_owner(
        structural,
        &properties,
        &structural_dof_layout,
        &canonical_node_indices,
        local_execution,
        awaited_elements,
        before,
    )
    .await?;
    let mut rhs = vec![0.0_f64; dof_count];
    let mut connector_stiffness_assembler = RealSparseAssembler::square(
        dof_count,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let mut connector_damping_assembler = RealSparseAssembler::square(
        dof_count,
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let connector_summary = assemble_explicit_connectors(
        &model.connectors,
        structural,
        &topology,
        &structural_dof_layout,
        local_execution,
        &mut connector_stiffness_assembler,
        &mut connector_damping_assembler,
    )?;
    let connector_stiffness = connector_stiffness_assembler
        .finish()
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))?;
    let connector_damping = connector_damping_assembler
        .finish()
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))?;
    let source_projection = compile_explicit_load_plan(
        structural,
        &properties,
        excitations,
        constraint_plan,
        &topology,
        &structural_dof_layout,
    )
    .map_err(|error| StructuralAssemblyError::InvalidLoadPlan(error.to_string()))?;
    apply_structural_load_plan(
        &source_projection.plan,
        &topology,
        &constraint_plan.reference_points,
        &structural_dof_layout,
        &mut rhs,
    )
    .map_err(|error| StructuralAssemblyError::InvalidLoadPlan(error.to_string()))?;
    let physical_load = rhs.clone();
    let direct_rotational_moment_load_count = source_projection.direct_rotational_moment_count;
    let structural_moment_projection = source_projection.moment_projection;
    let spatial_projection_evidence = source_projection
        .plan
        .entries
        .iter()
        .flat_map(|entry| entry.evidence.iter().cloned())
        .collect();
    let structural_load_plan_identity = Some(source_projection.plan.plan_digest);

    let mut direct_constraints = DirectConstraintAssignments::new(&structural_dof_layout);
    let resolved_constraint_matrix =
        resolve_constraint_matrix(constraint_plan, &topology, &structural_dof_layout)
            .map_err(|error| StructuralAssemblyError::InvalidConstraintPlan(error.to_string()))?;
    let has_general_rows = constraint_plan.enforcement != ConstraintEnforcement::DirectElimination;
    let (
        constrained,
        constrained_dof_count,
        rotational_constraint_count,
        constraint_matrix,
        affine_constraint_reduction,
    ) = if has_general_rows {
        let reduction = if constraint_plan.enforcement == ConstraintEnforcement::LagrangeMultiplier
        {
            None
        } else {
            Some(
                resolved_constraint_matrix
                    .affine_reduction()
                    .map_err(|error| {
                        StructuralAssemblyError::InvalidConstraintPlan(error.to_string())
                    })?,
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
        apply_direct_constraints(constraint_plan, &topology, &mut direct_constraints)?;
        let constrained_dof_count = direct_constraints.constrained_dof_count();
        let rotational_constraint_count = direct_constraints.rotational_constraint_count();
        let constrained = direct_constraints.apply(&mut rhs)?;
        (
            constrained,
            constrained_dof_count,
            rotational_constraint_count,
            None,
            None,
        )
    };
    drop(direct_constraints);

    let stiffness = if connector_summary.stiffness_count == 0 {
        element_stiffness.clone()
    } else {
        RealSparseMatrix::scaled_sum(
            &[(1.0, &element_stiffness), (1.0, &connector_stiffness)],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))?
    };
    let damping = if connector_summary.damping_count == 0 {
        element_damping
    } else {
        RealSparseMatrix::scaled_sum(
            &[(1.0, &element_damping), (1.0, &connector_damping)],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| StructuralAssemblyError::SparseOperator(format!("{error:?}")))?
    };
    let operators = CanonicalRealOperators::try_new(stiffness, mass, damping)
        .expect("assembled explicit operator roles and dimensions are valid");
    let nullspace_evidence = match if connector_summary.stiffness_count == 0 {
        NullspaceEliminationEvidence::prove_structural_frame(
            structural,
            &structural_dof_layout,
            operators.stiffness(),
            constraint_plan,
            &resolved_constraint_matrix,
            &constrained,
        )
    } else {
        NullspaceEliminationEvidence::prove_connector_anchored_structural_frame(
            crate::algebra::ConnectorAnchoredStructuralFrameInput {
                structural,
                layout: &structural_dof_layout,
                element_operator: &element_stiffness,
                connector_operator: &connector_stiffness,
                combined_operator: operators.stiffness(),
                plan: constraint_plan,
                constraints: &resolved_constraint_matrix,
                constrained_dofs: &constrained,
            },
        )
    } {
        Ok(evidence) => Some(evidence),
        Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
        Err(error) => return Err(StructuralAssemblyError::SparseOperator(error.to_string()).into()),
    };
    let operator = AssembledRealSystem::try_new_with_nullspace_evidence(
        operators,
        constrained,
        rhs,
        nullspace_evidence,
    )
    .expect("assembled explicit structural system is consistent");

    Ok(AssemblySummary {
        dof_count,
        structural_node_count: structural_dof_layout.node_count(),
        structural_translational_dof_count: structural_dof_layout.translational_dof_count(),
        structural_rotational_dof_count: structural_dof_layout.rotational_dof_count(),
        structural_rotation_node_count: structural_dof_layout.rotation_node_count(),
        structural_explicit_node_indices: (0..structural.nodes.len()).collect(),
        structural_load_plan_identity,
        structural_moment_load_count: excitations
            .iter()
            .filter(|load| matches!(load.kind, RealizedExcitationKind::Moment(_)))
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
        structural_solid_element_count: 0,
        structural_solid_total_mass_kg: 0.0,
        structural_solid_recovery: Vec::new(),
        structural_dof_layout,
        structural_reduced_basis: None,
        structural_beam_recovery,
        structural_shell_recovery,
        structural_formulations,
        structural_property_map_identity: Some(properties.identity.clone()),
        structural_sources: structural_source_implementations(
            excitations,
            StructuralProjectionBasis::ExplicitConnectivity,
        ),
        structural_boundary_transfers: if constraint_plan.groups.is_empty() {
            vec![BoundaryTransferImplementation::NotApplicable]
        } else {
            vec![BoundaryTransferImplementation::ExplicitRegionConnectivity]
        },
        constrained_dof_count,
        physical_load,
        constraint_matrix,
        affine_constraint_reduction,
        load_count: excitations.len(),
        structural_material,
        thermo_mechanical: None,
        electro_thermal: None,
        operator,
    })
}

fn map_explicit_property_error(
    error: crate::properties::PropertyCompilationError,
) -> StructuralAssemblyError {
    match error {
        crate::properties::PropertyCompilationError::InvalidSection {
            role: "beam",
            element_id,
            section_id,
        } => StructuralAssemblyError::UnknownBeamSection {
            element_id,
            section_id,
        },
        crate::properties::PropertyCompilationError::InvalidSection {
            role: "shell",
            element_id,
            section_id,
        } => StructuralAssemblyError::UnknownShellSection {
            element_id,
            section_id,
        },
        error => StructuralAssemblyError::InvalidMaterialCoverage(error.to_string()),
    }
}
