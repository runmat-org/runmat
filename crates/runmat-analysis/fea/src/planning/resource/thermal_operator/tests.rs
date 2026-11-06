use super::*;
use crate::planning::resource::Count;
use crate::{
    algebra::{MatrixProperties, OperatorRole, RealSparseAssembler},
    execution::ThermalWorkload,
};
use runmat_analysis_core::{
    AnalysisBoundary, AnalysisSequence, BoundaryOperator, HeatTransferCoefficient, PrescribedValue,
    ProcedureStep, RegionTarget, Temperature, ThermalDomain,
};
use runmat_meshing_core::{
    fixtures::{canonical_tetrahedron_solver_mesh, three_region_tetrahedron_solver_mesh},
    ElementOrder,
};

fn workload() -> ThermalWorkload {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_transient("heat")]);
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

#[test]
fn thermal_support_covers_actual_operators_and_exact_proof_work() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let facts = ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let mut workload = workload();
        workload.model.boundaries.push(AnalysisBoundary {
            boundary_id: "robin".into(),
            target: RegionTarget::boundary(
                projection.boundary_faces[0].provenance[0]
                    .analysis_selection_id()
                    .unwrap(),
            ),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(293.15)),
                coefficient: HeatTransferCoefficient::new(10.0),
            },
        });
        let sample = crate::excitation::ExcitationSamplePoint::time(0.0);
        let excitations = crate::excitation::realize_step_excitations(
            &workload.model,
            &workload.model.sequence.steps[0],
            sample,
        )
        .unwrap();
        let assembly = crate::physics::thermal::assembly::assemble(
            &workload.model,
            &mesh,
            &crate::properties::compile_thermal_property_map(&workload.model, Some(&mesh)).unwrap(),
            facts.support(),
            &excitations,
            sample,
        )
        .unwrap();
        facts.verify_matrix(&assembly.conductivity).unwrap();
        facts.verify_matrix(&assembly.capacity).unwrap();
        let nodes = projection.nodes.len() as u64;
        let insufficient = crate::algebra::SparsePattern::try_new(
            nodes,
            nodes,
            (0..=nodes).collect(),
            (0..nodes).collect(),
        )
        .unwrap();
        let rejected = crate::physics::thermal::assembly::assemble(
            &workload.model,
            &mesh,
            &crate::properties::compile_thermal_property_map(&workload.model, Some(&mesh)).unwrap(),
            &insufficient,
            &excitations,
            sample,
        )
        .unwrap_err();
        assert!(
            rejected
                .to_string()
                .contains("outside the symbolic pattern"),
            "{rejected}"
        );
        let live_assembly_arrays =
            std::mem::size_of_val(assembly.applied_source.canonical_values().as_slice())
                + std::mem::size_of_val(
                    assembly.positive_robin_weight.canonical_values().as_slice(),
                )
                + std::mem::size_of_val(
                    assembly
                        .prescribed_temperature_k
                        .canonical_values()
                        .as_slice(),
                )
                + std::mem::size_of_val(
                    assembly
                        .element_conductivity_w_per_mk
                        .canonical_values()
                        .as_slice(),
                )
                + std::mem::size_of_val(
                    assembly
                        .element_volumetric_capacity_j_per_m3k
                        .canonical_values()
                        .as_slice(),
                );
        let construction_property_arrays = 2 * projection.elements.len() * 8;
        assert_eq!(
            facts.workspace().assembly_vector_bytes.0,
            (2 * live_assembly_arrays + construction_property_arrays) as u64
        );
        assert_eq!(
            facts.workspace().projection_estimated_bytes.0,
            2 * crate::planning::resource::TopologyResourceFacts::from_projection(&projection)
                .unwrap()
                .estimated_resident_bytes
                .0
        );
        for operator in [
            assembly
                .steady_increment_operator(
                    &workload.model,
                    sample,
                    &vec![293.15; projection.nodes.len()],
                )
                .unwrap(),
            assembly
                .transient_operator(
                    &workload.model,
                    sample,
                    0.25,
                    &vec![293.15; projection.nodes.len()],
                    &vec![0.0; projection.nodes.len()],
                )
                .unwrap(),
        ] {
            let prepared = operator.prepare_linear().unwrap();
            facts.verify_matrix(prepared.matrix()).unwrap();
            let certificate = prepared.admissibility().unwrap().coercivity();
            assert!(facts.solver_admission().admits_certificate(certificate));
            let mut insufficient = facts.solver_admission().clone();
            insufficient.factor_nonzero_upper_bound = Count(0);
            assert!(!insufficient.admits_certificate(certificate));
        }
        let result = workload.execute(&mesh).unwrap();
        assert!(result.temperature_snapshots.iter().all(|field| field
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|value| (*value - 293.15).abs() < 1e-8)));
    }
}

#[test]
fn disconnected_thermal_support_rejects_unplanned_cross_body_coupling() {
    let mesh = three_region_tetrahedron_solver_mesh();
    let facts = ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
    let dimension = facts.support().rows();
    let unconnected = (1..dimension)
        .find(|column| {
            !facts.support().column_indices()[facts.support().row_range(0)]
                .contains(&(*column as u64))
        })
        .unwrap();
    let mut extra = RealSparseAssembler::square(
        dimension,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    );
    extra.add(0, unconnected, 1.0);
    extra.add(unconnected, 0, 1.0);
    assert!(facts.verify_matrix(&extra.finish().unwrap()).is_err());
    let wrong_shape = crate::algebra::RealSparseMatrix::diagonal(
        vec![1.0; dimension + 1],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    assert!(facts.verify_matrix(&wrong_shape).is_err());
}

#[test]
fn thermal_numerical_reservation_is_enforced_before_solve() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut workload = workload();
    let facts = ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
    let history =
        ThermalHistoryResourceFacts::derive(&mesh, &workload.solve_schedule().unwrap(), &[])
            .unwrap();
    let required = facts.numerical_reservation_bytes(&history).unwrap().0;
    let rows = facts.support().rows() as u64;
    let entries = facts.support().column_indices().len() as u64;
    assert_eq!(
        required,
        (facts.csr_payload_bytes_upper_bound().0
            + facts.solver_admission().resident_bytes_upper_bound.0
            + facts.workspace().reservation_bytes().unwrap().0)
            .max(facts.planning_resident_bytes.0)
            + history.retained_field_bytes.0
            + history.retained_metadata_bytes.0
            + history.validation_scratch_bytes.0
            + 8 * (rows + 1 + entries)
    );
    assert_eq!(
        facts.planning_resident_bytes,
        ThermalOperatorResourceFacts::derivation_resident_bytes_estimate(&mesh).unwrap()
    );
    assert!(required > history.retained_field_bytes.0);
    let required = crate::planning::resource::ThermalPreassemblyResourceFacts::derive(
        &[],
        &workload.model,
        &workload.model.sequence.steps[0],
        &mesh,
        &workload.solve_schedule().unwrap(),
        &crate::admission::admit_nonstructural(
            runmat_analysis_core::PhysicsFamily::Thermal,
            workload.procedure.procedure(),
            Some(&mesh),
            runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
            runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
        )
        .unwrap(),
    )
    .unwrap()
    .reservation_bytes()
    .unwrap()
    .0;
    // The old payload-only reservation must no longer admit a solve with missing metadata.
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(required - history.retained_metadata_bytes.0);
    assert!(matches!(workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling {
            required: actual, ..
        })) if actual == required));
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(required - 1);
    let error = workload.execute(&mesh).unwrap_err();
    assert!(
        matches!(error, crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: actual, maximum, .. }) if actual == required && maximum == required - 1),
        "{error}"
    );
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(required);
    workload.execute(&mesh).unwrap();
    assert!(csr_bytes(u64::MAX, 1).is_err());
    assert!(csr_bytes(1, u64::MAX).is_err());
}

#[test]
fn thermal_admission_reserves_the_proving_factor_and_no_energy_budget() {
    // Thermal assembly captures no local-energy source: the Cholesky proof always runs, and its
    // factor serves the solve. The admission is the structural factor law; the graph-count
    // local-energy storage (192 B per stored entry) is never reserved.
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let facts = ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
        let envelope = crate::algebra::CoercivityWorkEnvelope::derive(facts.support()).unwrap();
        let admission = facts.solver_admission();
        assert_eq!(
            *admission,
            crate::planning::resource::preflight::factor_admission(facts.support()).unwrap()
        );
        assert_eq!(
            admission.update_count_upper_bound.0,
            envelope.factor_flop_limit()
        );
        assert!(admission.resident_bytes_upper_bound.0 >= envelope.factor_resident_limit());
    }
}

// witness: FEA-THERMAL-SOLVER-001/conduction_multigrid_plan @ thermal_steady/tet4/resource_plan;
//   coverage: partial;
//   boundary: The conduction multigrid solver term (AMG, PCG vectors, proof inventory) is below
//   the proving factor's on box meshes and the gap grows with the mesh; on these sizes the
//   planning-derivation estimate dominates both totals, so the plan keeps the factor.
#[test]
fn conduction_multigrid_solver_storage_grows_slower_than_the_factor() {
    let mut model = workload().model;
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_steady("heat")]);
    let step = model.sequence.steps[0].clone();
    let schedule = crate::execution::ThermalSolveSchedule::derive(
        &model,
        &step,
        crate::ThermalProcedureControl::Steady,
    )
    .unwrap();
    let mut ratios = Vec::new();
    for cells in [12, 24] {
        let mesh = runmat_meshing_core::fixtures::layered_box_mesh(
            [cells, cells, cells],
            [1.0, 1.0, 1.0],
            &[(2.0, "body")],
        );
        let history = ThermalHistoryResourceFacts::derive(&mesh, &schedule, &[]).unwrap();
        let facts = ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
        let factor = facts.solver_resident_bytes().unwrap().0;
        let multigrid = facts.clone().on_path(ThermalSolvePath::ConductionMultigrid);
        assert_eq!(multigrid.solver_admission().factor_nonzero_upper_bound, Count(0));
        let multigrid_bytes = multigrid.solver_resident_bytes().unwrap().0;
        assert!(multigrid_bytes < factor, "{multigrid_bytes} vs {factor}");
        // Within the budget the factor needs, the plan keeps the factor.
        let mut fitting = step.clone();
        fitting
            .accuracy
            .resources
            .set_resident_byte_budget(facts.numerical_reservation_bytes(&history).unwrap().0);
        assert_eq!(
            super::super::thermal_preassembly::select_thermal_solve_path(
                facts.clone(),
                &history,
                &model,
                &fitting,
            )
            .unwrap()
            .path(),
            ThermalSolvePath::ProvingFactorization
        );
        // Below both totals the refusal names both requirements and the budget.
        let total = multigrid.numerical_reservation_bytes(&history).unwrap().0;
        let mut small = step.clone();
        small.accuracy.resources.set_resident_byte_budget(total - 1);
        assert!(matches!(
            super::super::thermal_preassembly::select_thermal_solve_path(
                facts.clone(), &history, &model, &small,
            ),
            Err(ResourcePlanningError::ThermalSolveCeiling { multigrid_resident_bytes: Some(_), .. })
        ));
        ratios.push(factor as f64 / multigrid_bytes as f64);
    }
    assert!(ratios[1] > ratios[0], "{ratios:?}");
}
