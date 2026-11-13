use super::*;
use crate::prepare_analysis_objects;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, AccuracySourceReceipt},
    fixtures::{fixture_model, FixtureId},
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};
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

fn objects(kind: Kind, identity: Digest, bytes: Vec<u8>) -> PreparedAnalysisObjects {
    prepare_analysis_objects(
        kind,
        identity,
        &bytes,
        1024,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

#[test]
fn electrical_context_cold_authenticates_original_three_inputs_and_authored_requirements() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let workload = workload(&mesh, equation);
            let result = workload.execute(&mesh).unwrap();
            let receipt =
                ElectricalNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
            let request = AccuracyEvaluationRequest::new(
                receipt.receipt().producer,
                AccuracySourceKind::ElectricalStatic,
                receipt.source_result(),
                None,
                workload.model.sequence.steps[0].accuracy.clone(),
            )
            .unwrap();
            let inputs = [
                objects(
                    Kind::ElectricalStaticWorkload,
                    workload.identity().unwrap().digest(),
                    workload.canonical_bytes().unwrap(),
                ),
                objects(
                    Kind::Mesh,
                    Digest::from_bytes(*mesh.canonical_digest.bytes()),
                    mesh.canonical_encode().unwrap(),
                ),
                objects(
                    Kind::SolverResult,
                    result.source_result_identity().unwrap().digest(),
                    result.canonical_bytes().unwrap(),
                ),
            ];
            let refs = inputs.iter().collect::<Vec<_>>();
            let source = AccuracySourceReceipt::ElectricalStatic(receipt.clone());
            crate::validate_accuracy_source_context(&request, &source, None, &refs).unwrap();
            for missing in 0..3 {
                let partial = refs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, input)| (i != missing).then_some(*input))
                    .collect::<Vec<_>>();
                assert!(validate(&request, &receipt, None, &partial).is_err());
                let mut duplicated = refs.clone();
                duplicated[missing] = refs[(missing + 1) % 3];
                assert!(validate(&request, &receipt, None, &duplicated).is_err());
            }
            let mut changed_plan = request.plan.clone();
            // The plan identifier participates alongside all unchanged authored criteria.
            changed_plan.accuracy_plan_id = AccuracyPlanId("foreign-accuracy".into());
            let changed = AccuracyEvaluationRequest::new(
                request.producer,
                request.source_kind,
                request.source_result,
                None,
                changed_plan,
            )
            .unwrap();
            assert!(validate(&changed, &receipt, None, &refs).is_err());
            let mut foreign = workload.clone();
            foreign.model.sequence.sequence_id = "foreign".into();
            let changed = AccuracyEvaluationRequest::new(
                foreign.producer_identity().unwrap(),
                request.source_kind,
                request.source_result,
                None,
                request.plan.clone(),
            )
            .unwrap();
            assert!(validate(&changed, &receipt, None, &refs).is_err());
            let foreign_objects = objects(
                Kind::ElectricalStaticWorkload,
                foreign.identity().unwrap().digest(),
                foreign.canonical_bytes().unwrap(),
            );
            assert!(validate(
                &request,
                &receipt,
                None,
                &[&foreign_objects, refs[1], refs[2]]
            )
            .is_err());
            let mut false_identity = inputs[2].clone();
            false_identity.manifest.logical_identity = workload.identity().unwrap().digest();
            assert!(validate(
                &request,
                &receipt,
                None,
                &[refs[0], refs[1], &false_identity]
            )
            .is_err());
        }
    }
}
