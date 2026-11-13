//! Runtime admission for a completed structural linear solve.
//!
//! Solver execution may be direct or portable. This module is the single point that turns the
//! host-backed numerical result into a public, persisted-runtime result shape. It owns no solver
//! or artifact-store I/O.

use runmat_analysis_core::{AnalysisField, AnalysisModel, AnalysisProcedure, SolutionSpace};
use runmat_analysis_fea::properties::compile_structural_property_map;
use runmat_analysis_fea::{
    ComputeBackend, ExecutionBackend, FeaRunResult, SolverPreconditioner,
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    FEA_FIELD_STRUCTURAL_ROTATION, FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_VON_MISES,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::contracts::{
    AnalysisFieldDescriptor, AnalysisFieldLocation, AnalysisRunOptions, AnalysisRunResult,
    FallbackEvent, PreconditionerMode, QualityGate, QualityPolicy, QualityReason,
    QualityReasonCode, RunProvenance,
};
use super::result_policy::classify_run_status;
use super::{render_topology_from_solver_mesh, storage};

pub(super) struct LinearStaticResultInput<'a> {
    pub model: &'a AnalysisModel,
    pub solver_mesh: Option<&'a SolverMeshArtifact>,
    pub requested_backend: ComputeBackend,
    pub options: &'a AnalysisRunOptions,
    pub assigned_run_id: Option<&'a str>,
    /// The effective memory ceiling and solve path the placement admitted (provenance only).
    pub placement: super::contracts::StaticSolvePlacement,
}

pub(super) fn admit_linear_static_result(
    input: LinearStaticResultInput<'_>,
    run: FeaRunResult,
) -> Result<AnalysisRunResult, String> {
    validate_result(&input, &run)?;

    let mut fallback_events = Vec::new();
    match input.options.preconditioner_mode {
        PreconditionerMode::Auto | PreconditionerMode::Jacobi => {}
        PreconditionerMode::Ilu => {
            if run.preconditioner != SolverPreconditioner::Ilu0 {
                fallback_events.push(FallbackEvent::preconditioner(run.preconditioner));
            }
        }
        PreconditionerMode::Amg => {
            if run.preconditioner != SolverPreconditioner::Amg {
                fallback_events.push(FallbackEvent::preconditioner(run.preconditioner));
            }
        }
    }
    if input.requested_backend == ComputeBackend::Gpu
        && run.execution_backend != ExecutionBackend::Accelerator
    {
        fallback_events.push(FallbackEvent::solver_backend());
    }

    let solver_convergence = if run
        .structural_static_convergence
        .as_ref()
        .is_some_and(|e| e.converged)
    {
        QualityGate::Pass
    } else {
        QualityGate::Warn
    };
    let has_material_assignment_conflict = run.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .code
            .starts_with("ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_")
    });
    let field_topology_reasons = field_topology_quality_reasons(&run.fields, input.solver_mesh);
    let solid_mesh_reasons = solid_mesh_quality_reasons(input.model, input.solver_mesh);
    let mesh_validation_reasons = Vec::new();
    let missing_solid_mesh_reasons =
        missing_solid_solver_mesh_reasons(input.model, input.solver_mesh);
    let solid_mesh_has_failure = solid_mesh_reasons.iter().any(|reason| {
        matches!(
            reason.code,
            QualityReasonCode::SolidMeshNoVolumeElements
                | QualityReasonCode::SolidMeshUnsupportedElementKind
                | QualityReasonCode::SolidMeshUnmappedLoadRegion
                | QualityReasonCode::SolidMeshUnmappedConstraintRegion
                | QualityReasonCode::SolidMeshUnmappedBoundaryRegion
                | QualityReasonCode::StructuralPropertyCompilationFailed
                | QualityReasonCode::SolidMeshRenderTopologyIncomplete
                | QualityReasonCode::SolidMeshQualityMinJacobianFailed
        )
    }) || !mesh_validation_reasons.is_empty();
    let result_quality =
        if run.fields_are_empty() || solid_mesh_has_failure || !field_topology_reasons.is_empty() {
            QualityGate::Fail
        } else if !missing_solid_mesh_reasons.is_empty()
            || !solid_mesh_reasons.is_empty()
            || has_material_assignment_conflict
        {
            QualityGate::Warn
        } else {
            QualityGate::Pass
        };

    let mut quality_reasons = Vec::new();
    if has_material_assignment_conflict {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::MaterialAssignmentConflict,
            detail: "material assignment confidence conflict detected".to_string(),
        });
    }
    quality_reasons.extend(solid_mesh_reasons);
    quality_reasons.extend(mesh_validation_reasons);
    quality_reasons.extend(field_topology_reasons);
    quality_reasons.extend(missing_solid_mesh_reasons);
    if solver_convergence == QualityGate::Warn {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverNotConverged,
            detail: "solver convergence gate is warning".to_string(),
        });
    }
    if fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback)
    {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::SolverBackendFallback,
            detail: "solver backend fell back from accelerator to cpu_reference".to_string(),
        });
    }
    let numerically_acceptable = match input.options.quality_policy {
        QualityPolicy::Strict => {
            solver_convergence == QualityGate::Pass
                && result_quality == QualityGate::Pass
                && quality_reasons.is_empty()
        }
        QualityPolicy::Balanced => {
            solver_convergence == QualityGate::Pass && result_quality == QualityGate::Pass
        }
        QualityPolicy::Exploratory => {
            solver_convergence == QualityGate::Pass && result_quality != QualityGate::Fail
        }
    };
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    Ok(AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: input
            .assigned_run_id
            .map(str::to_owned)
            .unwrap_or_else(storage::next_run_id),
        render_topology: render_topology_from_solver_mesh(input.solver_mesh),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence,
        result_quality,
        run_status,
        numerically_acceptable,
        quality_reasons,
        provenance: RunProvenance {
            requested_backend: input.requested_backend,
            execution_backend: run.execution_backend,
            solver_backend: run.solver_backend,
            solver_device_apply_k_ratio: run.solver_device_apply_k_ratio,
            solver_host_sync_count: run.solver_host_sync_count,
            precision_mode: super::contracts::format_precision_mode(input.options.precision_mode),
            deterministic_mode: input.options.deterministic_mode,
            solver_method: run.solver_method,
            preconditioner: run.preconditioner,
            quality_policy: super::contracts::format_quality_policy(input.options.quality_policy),
            fallback_events,
            effective_memory_ceiling_bytes: input.placement.effective_memory_ceiling_bytes,
            static_solve_path: input.placement.static_solve_path,
        },
        run,
    })
}

fn validate_result(input: &LinearStaticResultInput<'_>, run: &FeaRunResult) -> Result<(), String> {
    run.validate_solver_formulation()
        .map_err(|e| e.to_string())?;
    if run.requested_backend != input.requested_backend {
        return Err(
            "structural result requested backend differs from its execution request".into(),
        );
    }
    if run.identity.procedure() != AnalysisProcedure::Static
        || run.identity.solution_space() != SolutionSpace::Direct
        || !input.model.sequence.steps.iter().any(|step| {
            step.family == run.identity.family()
                && step.procedure == run.identity.procedure()
                && step.solution_space == run.identity.solution_space()
        })
    {
        return Err("structural result identity differs from the requested model procedure".into());
    }
    if let Some(mesh) = input.solver_mesh {
        mesh.validate().map_err(|error| error.to_string())?;
        if mesh.geometry.geometry_revision != u64::from(input.model.geometry_revision) {
            return Err(
                "structural result mesh revision differs from the analysis model revision".into(),
            );
        }
    }
    Ok(())
}

fn field_topology_quality_reasons(
    fields: &[AnalysisField],
    solver_mesh: Option<&SolverMeshArtifact>,
) -> Vec<QualityReason> {
    let Some(mesh) = solver_mesh else {
        return Vec::new();
    };
    fields
        .iter()
        .filter_map(|field| {
            primary_solver_mesh_field_expected_count(field, mesh).map(|expected| (field, expected))
        })
        .filter_map(|(field, expected)| {
            let descriptor = AnalysisFieldDescriptor::from_field(field);
            let actual = descriptor_entity_count(field, &descriptor);
            (actual != expected).then(|| QualityReason {
                code: QualityReasonCode::FieldTopologyMismatch,
                detail: format!(
                    "field {} topology mismatch: topology_id={} location={:?} element_kind={} expected_entity_count={} actual_entity_count={}",
                    field.field_id,
                    descriptor.topology_id.as_deref().unwrap_or("none"),
                    descriptor.location,
                    descriptor.element_kind.as_deref().unwrap_or("none"),
                    expected,
                    actual
                ),
            })
        })
        .collect()
}

fn solid_mesh_quality_reasons(
    model: &AnalysisModel,
    solver_mesh: Option<&SolverMeshArtifact>,
) -> Vec<QualityReason> {
    let Some(mesh) = solver_mesh else {
        return Vec::new();
    };
    let mut reasons = Vec::new();
    // Sheet and wire meshes carry shells and beams realized on faces and edges, not volumes.
    if mesh.topology.volume_elements.is_empty() && mesh.resolved_request.dimension.is_solid() {
        reasons.push(QualityReason {
            code: QualityReasonCode::SolidMeshNoVolumeElements,
            detail: "solver mesh has no volume elements".to_string(),
        });
    }
    reasons.extend(property_compilation_quality_reason(model, mesh));
    if !mesh.topology.volume_elements.is_empty() && mesh.topology.boundary_faces.is_empty() {
        reasons.push(QualityReason {
            code: QualityReasonCode::SolidMeshRenderTopologyIncomplete,
            detail: format!(
                "solver mesh has {} volume elements but no boundary faces for solver field visualization",
                mesh.topology.volume_elements.len()
            ),
        });
    }
    reasons.extend(boundary_region_mapping_reasons(model, mesh));
    reasons
}

pub(super) fn property_compilation_quality_reason(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Option<QualityReason> {
    compile_structural_property_map(model, Some(mesh))
        .err()
        .map(|error| QualityReason {
            code: QualityReasonCode::StructuralPropertyCompilationFailed,
            detail: error.to_string(),
        })
}

fn missing_solid_solver_mesh_reasons(
    model: &AnalysisModel,
    solver_mesh: Option<&SolverMeshArtifact>,
) -> Vec<QualityReason> {
    if solver_mesh.is_some()
        || model
            .structural
            .as_ref()
            .is_some_and(|structural| !structural.elements.is_empty())
    {
        return Vec::new();
    }
    vec![QualityReason {
        code: QualityReasonCode::MissingSolidAnalysisMesh,
        detail: "linear static structural result requires a canonical solver mesh".to_string(),
    }]
}

fn boundary_region_mapping_reasons(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Vec<QualityReason> {
    let mut reasons = Vec::new();
    for action in &model.excitations {
        let runmat_analysis_core::ExcitationTarget::Region(target) = &action.excitation().target
        else {
            continue;
        };
        if target.expected_topology == runmat_analysis_core::ExpectedTopology::BoundaryFace
            && !mesh_has_boundary_target(mesh, target)
        {
            reasons.push(QualityReason {
                code: QualityReasonCode::SolidMeshUnmappedLoadRegion,
                detail: format!(
                    "excitation `{}` references region `{}` but the solver mesh has no matching boundary provenance",
                    action.excitation().excitation_id.0,
                    target.region_id
                ),
            });
        }
    }
    for constraint in &model.constraints {
        if let Some(target) = constraint.target() {
            if !mesh_has_boundary_target(mesh, target) {
                reasons.push(QualityReason {
                    code: QualityReasonCode::SolidMeshUnmappedConstraintRegion,
                    detail: format!(
                        "constraint `{}` references region `{}` but the solver mesh has no matching boundary provenance",
                        constraint.constraint_id(), target.region_id
                    ),
                });
            }
        }
    }
    for boundary in &model.boundaries {
        if !mesh_has_boundary_target(mesh, &boundary.target) {
            reasons.push(QualityReason {
                code: QualityReasonCode::SolidMeshUnmappedBoundaryRegion,
                detail: format!(
                    "boundary `{}` references region `{}` but the solver mesh has no matching boundary provenance",
                    boundary.boundary_id, boundary.target.region_id
                ),
            });
        }
    }
    reasons
}

fn mesh_has_boundary_target(
    mesh: &SolverMeshArtifact,
    target: &runmat_analysis_core::RegionTarget,
) -> bool {
    // Face, edge and vertex targets resolve through the provenance of boundary faces, boundary
    // edges and nodes respectively.
    let matches = |provenance: &[runmat_geometry_core::PersistentEntityId]| {
        provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
    };
    mesh.topology
        .boundary_faces
        .iter()
        .any(|face| matches(&face.provenance))
        || mesh
            .topology
            .boundary_edges
            .iter()
            .any(|edge| matches(&edge.provenance))
        || mesh
            .topology
            .nodes
            .iter()
            .any(|node| matches(&node.provenance))
}

fn primary_solver_mesh_field_expected_count(
    field: &AnalysisField,
    mesh: &SolverMeshArtifact,
) -> Option<usize> {
    if matches!(
        field.field_id.as_str(),
        FEA_FIELD_STRUCTURAL_ROTATION
            | FEA_FIELD_STRUCTURAL_REACTION_FORCE
            | FEA_FIELD_STRUCTURAL_REACTION_MOMENT
    ) {
        return None;
    }
    let descriptor = AnalysisFieldDescriptor::from_field(field);
    if let (Some(topology_id), Some(location)) = (
        descriptor.topology_id.as_deref(),
        solver_mesh_field_topology_location(descriptor.location),
    ) {
        if let Some(topology) =
            mesh.topology.field_topologies.iter().find(|topology| {
                topology.topology_id == topology_id && topology.location == location
            })
        {
            return Some(topology.ordered_entity_ids.len());
        }
    }
    match field.field_id.as_str() {
        FEA_FIELD_STRUCTURAL_DISPLACEMENT | FEA_FIELD_STRUCTURAL_NODAL_VON_MISES => {
            Some(mesh.topology.nodes.len())
        }
        FEA_FIELD_STRUCTURAL_STRAIN
        | FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY
        | FEA_FIELD_STRUCTURAL_STRESS
        | FEA_FIELD_STRUCTURAL_VON_MISES => Some(mesh.topology.volume_elements.len()),
        _ => None,
    }
}

fn solver_mesh_field_topology_location(
    location: AnalysisFieldLocation,
) -> Option<runmat_meshing_core::FieldTopologyLocation> {
    match location {
        AnalysisFieldLocation::Node => Some(runmat_meshing_core::FieldTopologyLocation::Node),
        AnalysisFieldLocation::Element => {
            Some(runmat_meshing_core::FieldTopologyLocation::VolumeElement)
        }
        AnalysisFieldLocation::BoundaryFace => {
            Some(runmat_meshing_core::FieldTopologyLocation::BoundaryFace)
        }
        _ => None,
    }
}

fn descriptor_entity_count(field: &AnalysisField, descriptor: &AnalysisFieldDescriptor) -> usize {
    if descriptor.entity_count > 0 {
        return descriptor.entity_count;
    }
    if matches!(
        descriptor.location,
        AnalysisFieldLocation::Global | AnalysisFieldLocation::Mode
    ) {
        return field.element_count();
    }
    if let Some(first_dim) = field.shape.first().copied() {
        if field.shape.len() > 1 || descriptor.component_count.is_some() {
            return first_dim;
        }
    }
    field.element_count()
}

#[cfg(test)]
mod lineage_tests {
    use super::*;
    use runmat_analysis_core::RegionTarget;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    #[test]
    fn boundary_quality_uses_resolved_occurrence_provenance() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let source_face = mesh.topology.boundary_faces[0].provenance[0].clone();
        let mut target = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
            "authored_support",
        ));
        target
            .bind_resolved_entities(vec![source_face.clone()])
            .expect("resolve authored face");
        assert!(mesh_has_boundary_target(&mesh, &target));

        let mut other_occurrence = source_face;
        other_occurrence
            .assembly_path
            .push("other-instance".to_string());
        target
            .bind_resolved_entities(vec![other_occurrence])
            .expect("resolve another occurrence");
        assert!(!mesh_has_boundary_target(&mesh, &target));
    }
}

#[cfg(test)]
mod convergence_tests {
    use super::*;
    #[test]
    fn static_admission_uses_original_typed_convergence_without_diagnostic_text() {
        let mut model = runmat_analysis_fea::fixtures::fixture_model(
            runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
        );
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
            runmat_meshing_core::ElementOrder::Tet4,
        );
        model.signals.clear();
        model.excitations.clear();
        model.excitation_cases.clear();
        model.constraints = mesh
            .topology
            .nodes
            .iter()
            .map(|node| {
                runmat_analysis_core::AnalysisConstraint::Structural(
                    runmat_analysis_core::StructuralConstraint {
                        constraint_id: format!("fixed_{}", node.node_id),
                        frame: runmat_analysis_core::CoordinateFrame::Global,
                        kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                            target: runmat_analysis_core::RegionTarget::point(format!(
                                "node:{}",
                                node.node_id
                            )),
                        },
                    },
                )
            })
            .collect();
        let mut run = runmat_analysis_fea::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            runmat_analysis_fea::LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..Default::default()
            },
        )
        .unwrap();
        let options = AnalysisRunOptions {
            quality_policy: QualityPolicy::Exploratory,
            ..Default::default()
        };
        let input = || LinearStaticResultInput {
            model: &model,
            solver_mesh: Some(&mesh),
            requested_backend: ComputeBackend::Cpu,
            options: &options,
            assigned_run_id: Some("typed-convergence"),
            placement: Default::default(),
        };
        run.diagnostics.clear();
        assert_eq!(
            admit_linear_static_result(input(), run.clone())
                .unwrap()
                .solver_convergence,
            QualityGate::Pass
        );
        run.structural_static_convergence
            .as_mut()
            .unwrap()
            .converged = false;
        run.diagnostics
            .push(runmat_analysis_fea::diagnostics::FeaDiagnostic {
                code: "FEA_CONVERGENCE".into(),
                severity: runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Info,
                message: "converged=true".into(),
            });
        assert_eq!(
            admit_linear_static_result(input(), run.clone())
                .unwrap()
                .solver_convergence,
            QualityGate::Warn
        );
        assert!(
            !admit_linear_static_result(input(), run.clone())
                .unwrap()
                .numerically_acceptable
        );
        run.structural_static_convergence = None;
        assert!(admit_linear_static_result(input(), run).is_err());
    }
}
