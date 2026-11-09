use super::*;
use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
};
use runmat_analysis_core::{AnalysisSequence, Temperature, ThermalDomain};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    // The authored ceiling is part of the encoded model, so resident requirements are
    // measured under a non-binding authored ceiling (present, like the ceilings tests author).
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(runmat_types::wire_integer::MAX_WIRE_INTEGER);
    ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::ComputeBackend::Cpu,
        procedure: crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

/// The reservation law: resident components, the compiled properties, and the larger of the
/// numerical phase and the publication phase (histories, field construction, manifest encoding).
fn composed_reservation(facts: &ThermalPreassemblyResourceFacts) -> u64 {
    let numerical = facts
        .operators
        .numerical_reservation_bytes(&facts.history)
        .unwrap()
        .0;
    let post_solve = facts.history.reservation_bytes().unwrap().0
        + facts.field_construction.reservation_bytes().0
        + facts.manifest_metadata.reservation_bytes().0;
    facts.model_materialization_bytes.0
        + numerical.max(post_solve)
        + facts.sources.reservation_bytes().unwrap().0
        + facts.observations.reservation_bytes().unwrap().0
        + facts.realization.reservation_bytes().unwrap().0
        + facts.metadata.reservation_bytes().unwrap().0
        + facts.property_compilation_bytes.0
        + facts.manifest_catalog.reservation_bytes().unwrap().0
        + facts.manifest_retained.reservation_bytes().unwrap().0
}

// witness: FEA-THERMAL-EXECUTION-001/admission @ thermal_transient/tet4|tet10/public_study;
//   boundary: Both-order field construction and manifest scratch are charged inside the
//   numerical maximum of the preassembly ceiling; no measured RSS or whole native pipeline
//   evidence.
#[test]
fn post_solve_scratch_is_charged_inside_the_numerical_maximum() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        let step = &workload.model.sequence.steps[0];
        let facts = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            step,
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh),
        )
        .unwrap();
        let scratch = facts.field_construction.reservation_bytes().0;
        assert!(scratch > 0);
        assert_eq!(
            scratch,
            facts
                .field_construction
                .largest_identity_materialization_bytes
                .0
                .max(
                    facts
                        .field_construction
                        .largest_partition_construction_bytes
                        .0
                )
        );
        let required = facts.reservation_bytes().unwrap().0;
        assert_eq!(required, composed_reservation(&facts));
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required - 1);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { required: actual, .. }
        )) if actual == required)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required);
        let result = workload.execute(&mesh).unwrap();
        assert_eq!(result.run.identity, facts.admitted_identity);
        assert_eq!(result.run.identity, admitted_identity(&workload, &mesh));
        let mut overflow = facts;
        overflow
            .field_construction
            .largest_identity_materialization_bytes = ByteCount(u64::MAX);
        assert!(overflow.reservation_bytes().is_err());
    }
}

#[test]
fn manifest_catalog_is_admitted_with_checked_retained_and_validation_costs() {
    let mut workload = workload();
    // An authored ceiling with the same canonical integer width as the values set below, so the
    // workload encoding (and the requirement derived from it) does not change with the value.
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(u64::from(u32::MAX));
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let schedule = workload.solve_schedule().unwrap();
    let facts = ThermalPreassemblyResourceFacts::derive(
        &[],
        &workload.model,
        &workload.model.sequence.steps[0],
        &mesh,
        &schedule,
        &admitted_identity(&workload, &mesh),
    )
    .unwrap();
    let catalog = facts.manifest_catalog.reservation_bytes().unwrap().0;
    let required = facts.reservation_bytes().unwrap().0;
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(required - 1);
    // The catalog is part of that complete requirement.
    assert!(catalog > 0 && catalog < required);
    let outcome = workload.execute(&mesh);
    assert!(
        matches!(&outcome, Err(crate::FeaRunError::Resource(
        ResourcePlanningError::AccuracyCeiling { required: actual, .. }
    )) if *actual == required),
        "{:?} versus {required}",
        outcome.as_ref().err()
    );
    let mut overflow = facts.manifest_catalog;
    overflow.validation_scratch_bytes = ByteCount(u64::MAX);
    assert!(overflow.reservation_bytes().is_err());
}

#[test]
fn backend_evidence_bound_tracks_solves_and_rejects_overflow() {
    let mut workload = workload();
    for (procedure, expected) in [
        (crate::ThermalProcedureControl::Steady, 1),
        (
            crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            4,
        ),
        (
            crate::ThermalProcedureControl::BackwardEuler {
                snapshot_count: 257,
            },
            512,
        ),
    ] {
        workload.procedure = procedure;
        workload.model.sequence.steps[0] = match procedure {
            crate::ThermalProcedureControl::Steady => ProcedureStep::thermal_steady("heat"),
            crate::ThermalProcedureControl::BackwardEuler { .. } => {
                ProcedureStep::thermal_transient("heat")
            }
        };
        let schedule = workload.solve_schedule().unwrap();
        let facts = super::super::ThermalBackendEvidenceResourceFacts::derive(&schedule).unwrap();
        assert_eq!(facts.maximum_records, expected);
        let record =
            crate::solve::accelerator::AcceleratorSolveEvidence::completed_evidence_fixture();
        facts.verify(&record).unwrap();
        assert!(
            facts.reservation_bytes().unwrap().0
                > expected as u64 * facts.record.materialization_resident_bytes_estimate
        );
        let mut overflow = facts;
        overflow.record.materialization_resident_bytes_estimate = u64::MAX;
        assert!(overflow.reservation_bytes().is_err());
    }
}

#[test]
fn assignment_diagnostics_are_admitted_and_checked_before_retention() {
    use runmat_analysis_core::EvidenceConfidence;
    let mut workload = workload();
    let step = workload.model.sequence.steps[0].clone();
    let schedule = workload.solve_schedule().unwrap();
    let base = ThermalMetadataResourceFacts::derive(&workload.model, &step, &schedule).unwrap();
    let mut alternate = workload.model.materials[0].clone();
    alternate.material_id = "expected".repeat(512);
    workload.model.material_assignments[0].expected_material_id = alternate.material_id.clone();
    workload.model.material_assignments[0].confidence = EvidenceConfidence::Probable;
    workload.model.materials.push(alternate);
    let facts = ThermalMetadataResourceFacts::derive(&workload.model, &step, &schedule).unwrap();
    assert!(facts.reservation_bytes().unwrap().0 > base.reservation_bytes().unwrap().0);
    let diagnostics = crate::diagnostics::builders::material_assignment_diagnostics(
        &workload.model.material_assignments,
    );
    facts.verify_assignment_diagnostics(&diagnostics).unwrap();
    assert!(base.verify_assignment_diagnostics(&diagnostics).is_err());
    let oversized_solver_diagnostic = crate::diagnostics::FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: crate::diagnostics::FeaDiagnosticSeverity::Info,
        message: "x".repeat(facts.solver_diagnostics.encoded_bytes_upper_bound as usize + 1),
    };
    assert!(facts
        .verify_solver_diagnostics(&[oversized_solver_diagnostic])
        .is_err());
    let mut overflowing = facts;
    overflowing
        .solver_diagnostics
        .materialization_resident_bytes_estimate = u64::MAX;
    assert!(overflowing.reservation_bytes().is_err());
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        let preassembly = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &step,
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh),
        )
        .unwrap();
        assert_eq!(
            preassembly.metadata.assignment_diagnostics,
            facts.assignment_diagnostics
        );
        let mut bounded = step.clone();
        bounded.accuracy.resources.maximum_resident_bytes =
            Some(preassembly.reservation_bytes().unwrap().0 - 1);
        assert!(matches!(
            ThermalPreassemblyResourceFacts::derive(
                &[],
                &workload.model,
                &bounded,
                &mesh,
                &schedule,
                &admitted_identity(&workload, &mesh)
            ),
            Err(ResourcePlanningError::AccuracyCeiling { .. })
        ));
    }
}

#[test]
fn mesh_validation_is_admitted_before_projection_in_derivation_and_execution() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut workload = workload();
    // Raw field maps must be counted even when invalid. A stale seal makes validation fail
    // if either execution or projection reaches it before the resource gate.
    let valid = mesh.validation_resident_bytes_estimate().unwrap();
    mesh.topology.field_topologies[0]
        .ordered_entity_ids
        .extend(10_000..110_000);
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    let counted = mesh.validation_resident_bytes_estimate().unwrap();
    // The validation estimate counts the raw field map, including the invalid entries (at
    // least one encoded byte each).
    assert!(counted >= valid + 100_000, "{counted} vs {valid}");
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(counted);
    let schedule = workload.solve_schedule().unwrap();
    assert!(matches!(
        ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh)
        ),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { .. }
        ))
    ));
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = None;
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            ResourcePlanningError::Invalid(_)
        ))
    ));
}

#[test]
fn producer_metadata_is_bounded_and_verified_against_actual_initial_state() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        let base = super::super::ThermalMetadataResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &workload.solve_schedule().unwrap(),
        )
        .unwrap();
        let long_id = "thermal_long_step".repeat(256);
        workload.step_id = runmat_analysis_core::ProcedureStepId(long_id.clone());
        workload.model.sequence.steps[0].step_id = workload.step_id.clone();
        let grown = super::super::ThermalMetadataResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0],
            &workload.solve_schedule().unwrap(),
        )
        .unwrap();
        assert!(
            grown.initial_state.encoded_bytes_upper_bound
                > base.initial_state.encoded_bytes_upper_bound
        );
        assert!(
            grown.physical_diagnostics.encoded_bytes_upper_bound
                > base.physical_diagnostics.encoded_bytes_upper_bound
        );
        let result = workload.execute(&mesh).unwrap();
        grown.verify_initial_state(&result.initial_state).unwrap();
        let mut substituted = result.initial_state.clone();
        substituted.procedure_step_id.push_str(&long_id);
        assert!(grown.verify_initial_state(&substituted).is_err());
        let mut too_many = result.initial_state.clone();
        too_many
            .applied_initial_condition_ids
            .push("undeclared".into());
        assert!(grown.verify_initial_state(&too_many).is_err());
    }
}

#[test]
fn model_materialization_is_composed_and_enforced_before_assembly() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        let schedule = workload.solve_schedule().unwrap();
        let base = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh),
        )
        .unwrap();
        workload.model.model_id.0 = "large_model_identifier".repeat(512);
        let grown = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh),
        )
        .unwrap();
        assert!(grown.model_materialization_bytes.0 > base.model_materialization_bytes.0);
        let required = grown.reservation_bytes().unwrap().0;
        assert_eq!(required, composed_reservation(&grown));
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required - 1);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { required: actual, maximum, .. }
        )) if actual == required && maximum == required - 1)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required);
        let result = workload.execute(&mesh).unwrap();
        assert!(result
            .temperature_snapshots
            .iter()
            .flat_map(|field| field.as_host_f64().unwrap())
            .all(|value| (value - 293.15).abs() < 1e-9));
        let mut overflowing = grown;
        overflowing.model_materialization_bytes = ByteCount(u64::MAX);
        assert!(overflowing.reservation_bytes().is_err());
    }
}

#[test]
fn planning_ceiling_precedes_geometric_projection_at_exact_boundary() {
    use runmat_meshing_core::CanonicalMeshingContract;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        // Canonical connectivity remains valid, but physical projection must reject zero volume.
        for node in &mut mesh.topology.nodes {
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([0.0; 3]);
        }
        mesh.seal_canonical_digest().unwrap();
        mesh.validate_canonical().unwrap();
        let mut workload = workload();
        let schedule = workload.solve_schedule().unwrap();
        let history = ThermalHistoryResourceFacts::derive(&mesh, &schedule, &[]).unwrap();
        let model_bytes = runmat_canonical_codec::profile_serialized_shape(
            &workload.model,
            crate::execution::THERMAL_WORKLOAD_LIMITS,
        )
        .unwrap()
        .materialization_resident_bytes_estimate
            * 2;
        let planning = ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(&mesh)
            .unwrap()
            .0;
        let required = history.reservation_bytes().unwrap().0 + model_bytes + planning;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required - 1);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(
            ResourcePlanningError::AccuracyCeiling { required: actual, maximum, .. }
        )) if actual == required && maximum == required - 1)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required);
        let expected = crate::topology::AnalysisProjectionError::InvalidElementGeometry {
            element_id: mesh.topology.volume_elements[0].element_id,
        }
        .to_string();
        assert!(matches!(
            workload.execute(&mesh),
            Err(crate::FeaRunError::Resource(
                ResourcePlanningError::Invalid(message)
            )) if message == expected
        ));
    }
}

#[test]
fn oversized_input_strings_fail_the_shared_codec_limit() {
    let mut workload = workload();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let schedule = workload.solve_schedule().unwrap();
    workload.model.model_id.0 =
        "x".repeat(crate::execution::THERMAL_WORKLOAD_LIMITS.maximum_string_bytes + 1);
    assert!(ThermalPreassemblyResourceFacts::derive(
        &[],
        &workload.model,
        &workload.model.sequence.steps[0],
        &mesh,
        &schedule,
        &admitted_identity(&workload, &mesh)
    )
    .is_err());
}

#[test]
fn retained_property_metadata_is_reserved_before_assembly() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        let base = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh),
        )
        .unwrap();
        workload.model.materials[0].name = "long_material_description".repeat(512);
        let facts = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
            &admitted_identity(&workload, &mesh),
        )
        .unwrap();
        assert!(facts.property_compilation_bytes.0 > base.property_compilation_bytes.0);
        assert_eq!(
            facts.properties.solid_materials().len(),
            mesh.topology.volume_elements.len()
        );
        let required = facts.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required - 1);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: actual, .. })) if actual == required)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required);
        workload.execute(&mesh).unwrap();
    }
}

fn admitted_identity(
    workload: &ThermalWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> crate::admission::AnalysisRunIdentity {
    crate::admission::admit_nonstructural(
        runmat_analysis_core::PhysicsFamily::Thermal,
        workload.procedure.procedure(),
        Some(mesh),
        runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
        runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
    )
    .unwrap()
}

/// Calibration probe (ignored): prints every thermal reservation component for a linear steady
/// slab of `RM_DIAG_N`³ hexahedra (Tet10), then optionally solves it and prints the process
/// peak RSS. Run alone (`--exact --ignored`) so the RSS belongs to this solve.
#[test]
#[ignore]
fn diagnostic_thermal_components_at_scale() {
    use runmat_analysis_core::{
        AnalysisBoundary, BoundaryOperator, HeatTransferCoefficient, PrescribedValue, RegionTarget,
        Temperature,
    };
    let n = std::env::var("RM_DIAG_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mesh = runmat_meshing_core::fixtures::quadratic_slab_mesh(n);
    let mut workload = crate::physics::thermal::qualification::workload();
    workload
        .model
        .boundaries
        .retain(|b| b.boundary_id == "insulated-sides");
    workload.model.boundaries.push(AnalysisBoundary {
        boundary_id: "right-film".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(300.0)),
            coefficient: HeatTransferCoefficient::new(25.0),
        },
    });
    crate::physics::thermal::qualification::left_temperature(
        &mut workload,
        PrescribedValue::Constant(Temperature::new(350.0)),
    );
    let schedule = workload.solve_schedule().unwrap();
    let facts = ThermalPreassemblyResourceFacts::derive(
        &[],
        &workload.model,
        &workload.model.sequence.steps[0],
        &mesh,
        &schedule,
        &admitted_identity(&workload, &mesh),
    )
    .unwrap();
    let ops = &facts.operators;
    let admission = ops.solver_admission();
    println!(
        "nodes {} elements {} support_nnz {}",
        mesh.topology.nodes.len(),
        mesh.topology.volume_elements.len(),
        ops.support().column_indices().len()
    );
    println!(
        "operators total {} csr {} factor_admission {} workspace {} history {}",
        ops.numerical_reservation_bytes(&facts.history).unwrap().0,
        ops.csr_payload_bytes_upper_bound().0,
        admission.resident_bytes_upper_bound.0,
        ops.workspace().reservation_bytes().unwrap().0,
        facts.history.reservation_bytes().unwrap().0
    );
    println!(
        "sources {} observations {} realization {} metadata {} model {} properties {} \
         field_construction {} catalog {} manifest_metadata {} retained {} total {}",
        facts.sources.reservation_bytes().unwrap().0,
        facts.observations.reservation_bytes().unwrap().0,
        facts.realization.reservation_bytes().unwrap().0,
        facts.metadata.reservation_bytes().unwrap().0,
        facts.model_materialization_bytes.0,
        facts.property_compilation_bytes.0,
        facts.field_construction.reservation_bytes().0,
        facts.manifest_catalog.reservation_bytes().unwrap().0,
        facts.manifest_metadata.reservation_bytes().0,
        facts.manifest_retained.reservation_bytes().unwrap().0,
        facts.reservation_bytes().unwrap().0
    );
    if std::env::var("RM_DIAG_SOLVE").is_ok() {
        let peak = || {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
            unsafe { usage.assume_init() }.ru_maxrss
        };
        let before = peak();
        let result = workload.execute(&mesh).unwrap();
        println!(
            "rss_peak_before_solve {before} rss_peak_after_solve {} temperatures {}",
            peak(),
            result.temperature_snapshots.len()
        );
    }
}
