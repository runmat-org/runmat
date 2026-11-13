use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::{MagnetostaticSolverResult, SemanticStatePublicationSource},
    fixtures::{fixture_model, FixtureId},
};
use runmat_execution::ProgramEnvironment;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
use std::collections::BTreeSet;

fn workload() -> MagnetostaticWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.initial_conditions.clear();
    model.sequence = AnalysisSequence::from_steps(
        "magnetic-bridge",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.sequence.steps[0].state_outputs = vec![StateOutput {
        state_id: StateId("retained".into()),
        kind: StateKind::ElectromagneticStatic,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: Vec::new(),
    });
    for face in 0..4 {
        model
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: format!("ground:{face}"),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(format!("face:{face}")),
                        ),
                    ),
                },
            }));
    }
    model
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "gauge".into(),
            kind: MagneticConstraintKind::CoulombGauge {
                target: RegionTarget::volume(runmat_meshing_core::fixtures::fixture_solid("solid")),
            },
        }));
    MagnetostaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    }
}

fn revision() -> runmat_execution::ProgramRevision {
    runmat_execution::ProgramRevision::new(
        Digest::sha256("graph"),
        Digest::sha256("source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256("runtime"),
            Digest::sha256("catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn magnetic_plan_cold_transport_and_exact_preflight_bind_both_orders_and_chunks() {
    let source = workload();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let reference = source.execute(&mesh).unwrap();
        for chunk in [1024, 8192] {
            let policy =
                AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap();
            let mut prepared = prepare_magnetostatic_inputs(
                &source,
                &mesh,
                policy,
                runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            )
            .unwrap();
            prepared.inputs.push(
                prepare_analysis_objects(
                    AnalysisNumericalObjectKind::ResourcePlan,
                    prepared.resource_plan.identity.digest(),
                    &prepared.resource_plan.canonical_bytes().unwrap(),
                    chunk,
                    ObjectNamespace::InputValue,
                    policy.inventory_limits(),
                )
                .unwrap(),
            );
            let plan =
                rederive_magnetostatic_resource_plan(&source, &mesh, &prepared.inputs, policy)
                    .unwrap();
            assert_eq!(plan, prepared.resource_plan);
            let result = source.execute_with_resource_plan(&mesh, &plan).unwrap();
            assert_eq!(result, reference);
            let bytes = result.canonical_bytes().unwrap();
            assert!(
                bytes.len() as u64
                    <= magnetostatic_result_payload_bytes_upper_bound(&source, &mesh).unwrap()
            );
            let cold = MagnetostaticSolverResult::from_canonical_bytes_for(
                &bytes,
                Digest::sha256(&bytes),
                &source,
                &mesh,
            )
            .unwrap();
            assert_eq!(cold, reference);
            let mut substituted = prepared.inputs.clone();
            substituted[0] = prepare_analysis_objects(
                AnalysisNumericalObjectKind::MagnetostaticWorkload,
                source.identity().unwrap().digest(),
                b"foreign canonical source",
                chunk,
                ObjectNamespace::InputValue,
                policy.inventory_limits(),
            )
            .unwrap();
            assert!(
                rederive_magnetostatic_resource_plan(&source, &mesh, &substituted, policy).is_err()
            );
            let other = AnalysisOutputPolicy::new(
                if chunk == 1024 { 8192 } else { 1024 },
                policy.inventory_limits(),
            )
            .unwrap();
            assert!(
                rederive_magnetostatic_resource_plan(&source, &mesh, &prepared.inputs, other)
                    .is_err()
            );
            let mut changed = source.clone();
            changed.relative_residual_target *= 2.0;
            assert!(changed.execute_with_resource_plan(&mesh, &plan).is_err());
        }
    }
}

#[test]
fn magnetic_publication_driver_requires_distinct_work_kind_mesh_and_sampled_layout() {
    let source = workload();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    let task = crate::prepare_magnetostatic_task(crate::MagnetostaticTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: Digest::sha256("authored"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "magnetic-bridge-test".into(),
            encryption_context: Digest::sha256("scope"),
        },
        workload: source.clone(),
        mesh: mesh.clone(),
        output_policy: policy,
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    let result = source
        .execute_with_resource_plan(&mesh, &task.resource_plan)
        .unwrap();
    let bytes = result.canonical_bytes().unwrap();
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.source_result_identity().unwrap().digest(),
        &bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )
    .unwrap();
    let profiles = crate::magnetic_state_publication_profiles(&source, &objects).unwrap();
    let profile_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles.identity(),
        &profiles.canonical_bytes().unwrap(),
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )
    .unwrap();
    let input = || crate::SemanticStatePublicationPreparation {
        source_host: task.host.clone(),
        source_result: objects.clone(),
        source_profiles: profile_objects.clone(),
        source: SemanticStatePublicationSource::MagnetostaticNativeFields,
        declaration: source.model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: result
            .sampling_topology()
            .identity()
            .unwrap()
            .discretization_digest(),
        program_revision: revision(),
        chunk_bytes: policy.chunk_bytes(),
        inventory_limits: policy.inventory_limits(),
    };
    let prepared = crate::prepare_semantic_state_publication(input()).unwrap();
    assert_eq!(
        prepared.request,
        result.state_publication_requests().unwrap()[0]
    );
    let mut raw_mesh = input();
    raw_mesh.topology_digest = result.mesh_digest();
    assert!(crate::prepare_semantic_state_publication(raw_mesh).is_err());
    let mut foreign_mesh = input();
    foreign_mesh
        .source_host
        .workload
        .inputs
        .iter_mut()
        .find(|i| i.kind == AnalysisNumericalObjectKind::Mesh)
        .unwrap()
        .logical_identity = Digest::sha256("foreign mesh");
    assert!(crate::prepare_semantic_state_publication(foreign_mesh).is_err());
    let mut foreign_workload = input();
    foreign_workload
        .source_host
        .workload
        .inputs
        .iter_mut()
        .find(|i| i.kind == AnalysisNumericalObjectKind::MagnetostaticWorkload)
        .unwrap()
        .logical_identity = Digest::sha256("foreign workload");
    assert!(crate::prepare_semantic_state_publication(foreign_workload).is_err());
}

#[test]
fn magnetic_plan_carries_admitted_threads_and_worker_rederivation_converges() {
    let source = workload();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    let threads =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(3_000)
            .unwrap();
    let mut prepared = prepare_magnetostatic_inputs(&source, &mesh, policy, threads).unwrap();
    assert_eq!(prepared.resource_plan.request.cpu_millicores, 3_000);
    prepared.inputs.push(
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            prepared.resource_plan.identity.digest(),
            &prepared.resource_plan.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )
        .unwrap(),
    );
    let plan =
        rederive_magnetostatic_resource_plan(&source, &mesh, &prepared.inputs, policy).unwrap();
    assert_eq!(plan, prepared.resource_plan);
    assert_eq!(
        source.execute_with_resource_plan(&mesh, &plan).unwrap(),
        source.execute(&mesh).unwrap()
    );
}
