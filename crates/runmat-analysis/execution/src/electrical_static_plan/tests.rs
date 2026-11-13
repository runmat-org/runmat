use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::ElectricalStaticSolverResult,
    fixtures::{fixture_model, FixtureId},
};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload(
    mesh: &SolverMeshArtifact,
    equation: StaticElectricalEquation,
) -> ElectricalStaticWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.electromagnetic = Some(ElectromagneticDomain::Static { equation });
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.initial_conditions.clear();
    model.sequence = AnalysisSequence::from_steps(
        "electrical",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(4.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: Vec::new(),
    });
    for node in &mesh.topology.nodes {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: format!("node-{}", node.node_id),
                kind: ElectricalConstraintKind::PrescribedPotential {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    value: PrescribedValue::Constant(ElectricPotential::new(
                        2.0 * node.coordinates_m.canonical_value()[0],
                    )),
                },
            }));
    }
    ElectricalStaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-11,
        element_temperature_k: None,
        operating_time_s: None,
    }
}

fn resource_input(plan: &FeaResourcePlan, policy: AnalysisOutputPolicy) -> PreparedAnalysisObjects {
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes().unwrap(),
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )
    .unwrap()
}

#[test]
fn electrical_planning_and_admitted_solve_preserve_physics_and_cold_transport() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for equation in [
            StaticElectricalEquation::Electrostatics,
            StaticElectricalEquation::SteadyConduction,
        ] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let source = workload(&mesh, equation);
            let reference = source.execute(&mesh).unwrap();
            for chunk in [1024, 8192] {
                let policy =
                    AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap();
                let mut prepared = prepare_electrical_static_inputs(
                    &source,
                    &mesh,
                    policy,
                    LocalExecutionPlan::serial(),
                    16 << 30,
                )
                .unwrap();
                prepared
                    .inputs
                    .push(resource_input(&prepared.resource_plan, policy));
                let plan = rederive_electrical_static_resource_plan(
                    &source,
                    &mesh,
                    &prepared.inputs,
                    policy,
                )
                .unwrap();
                assert_eq!(plan, prepared.resource_plan);
                let result = source.execute_with_resource_plan(&mesh, &plan).unwrap();
                assert_eq!(result, reference);
                let bytes = result.canonical_bytes().unwrap();
                assert!(
                    bytes.len() as u64
                        <= electrical_static_result_payload_bytes_upper_bound(&source, &mesh)
                            .unwrap()
                );
                let output = prepare_analysis_objects(
                    AnalysisNumericalObjectKind::SolverResult,
                    result.source_result_identity().unwrap().digest(),
                    &bytes,
                    chunk,
                    ObjectNamespace::ResultValue,
                    policy.inventory_limits(),
                )
                .unwrap();
                let runmat_analysis_fea::planning::resource::FeaResourceDemand::ElectricalStatic(
                    demand,
                ) = &plan.input.demand
                else {
                    panic!("wrong demand")
                };
                crate::object::validate_prepared_output_resource_facts(
                    &output,
                    &demand.result_output,
                )
                .unwrap();
                let cold = ElectricalStaticSolverResult::from_canonical_bytes_for(
                    &output.payload(),
                    &source,
                    &mesh,
                )
                .unwrap();
                assert_eq!(cold, reference);
                let mut changed = prepared.inputs.clone();
                changed[0].manifest.logical_identity = Digest::sha256("wrong-source");
                assert!(
                    rederive_electrical_static_resource_plan(&source, &mesh, &changed, policy)
                        .is_err()
                );
                // A self-consistent transport closure with the correct claimed identity
                // still cannot substitute different canonical source bytes.
                let mut substituted = prepared.inputs.clone();
                substituted[0] = prepare_analysis_objects(
                    AnalysisNumericalObjectKind::ElectricalStaticWorkload,
                    source.identity().unwrap().digest(),
                    b"substituted workload",
                    chunk,
                    ObjectNamespace::InputValue,
                    policy.inventory_limits(),
                )
                .unwrap();
                assert!(rederive_electrical_static_resource_plan(
                    &source,
                    &mesh,
                    &substituted,
                    policy
                )
                .is_err());
                let other = AnalysisOutputPolicy::new(
                    if chunk == 1024 { 8192 } else { 1024 },
                    policy.inventory_limits(),
                )
                .unwrap();
                assert!(rederive_electrical_static_resource_plan(
                    &source,
                    &mesh,
                    &prepared.inputs,
                    other
                )
                .is_err());
                let mut wrong = source.clone();
                wrong.relative_residual_target *= 2.0;
                assert!(wrong.execute_with_resource_plan(&mesh, &plan).is_err());
            }
        }
    }
}

#[test]
fn electrical_ceiling_rejection_precedes_numerical_matrix() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut source = workload(&mesh, StaticElectricalEquation::Electrostatics);
    source.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    assert!(prepare_electrical_static_inputs(
        &source,
        &mesh,
        policy,
        LocalExecutionPlan::serial(),
        16 << 30
    )
    .is_err());
}

#[test]
fn electrical_plan_carries_admitted_threads_and_worker_rederivation_converges() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let source = workload(&mesh, StaticElectricalEquation::Electrostatics);
    let policy = AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap();
    let threads = LocalExecutionPlan::from_cpu_millicores(3_000).unwrap();
    let mut prepared =
        prepare_electrical_static_inputs(&source, &mesh, policy, threads, 16 << 30).unwrap();
    assert_eq!(prepared.resource_plan.request.cpu_millicores, 3_000);
    assert_eq!(prepared.resource_plan.input.worker.cpu_millicores, 3_000);
    prepared
        .inputs
        .push(resource_input(&prepared.resource_plan, policy));
    let plan =
        rederive_electrical_static_resource_plan(&source, &mesh, &prepared.inputs, policy).unwrap();
    assert_eq!(plan, prepared.resource_plan);
    assert_eq!(
        source.execute_with_resource_plan(&mesh, &plan).unwrap(),
        source.execute(&mesh).unwrap()
    );
}
