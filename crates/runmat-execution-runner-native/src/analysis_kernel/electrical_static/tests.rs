//! Native admitted scalar execution and cold original fields with analytic references.
use super::fixture;
use super::*;
use crate::NativeAnalysisKernel;
use runmat_analysis_core::StaticElectricalEquation;
use runmat_analysis_execution::{
    prepare_electrical_static_task, AnalysisKernel, AnalysisKernelInvocation, AnalysisOutputPolicy,
    ElectricalStaticTaskPreparation, PreparedElectricalStaticTask,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisWorkKind, ElectricalNumericalReceiptArtifact, ElectricalStaticSolverResult,
    },
    physics::electrical::{ElectricalConstitutiveResponse, VACUUM_PERMITTIVITY_F_PER_M},
    planning::resource::FeaResourceDemand,
};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::ElementOrder;

fn task(
    order: ElementOrder,
    equation: StaticElectricalEquation,
    chunk: usize,
) -> PreparedElectricalStaticTask {
    let (workload, mesh) = fixture::electrical_fixture(order, equation);
    let digest = Digest::sha256("native-anchored-electrical");
    prepare_electrical_static_task(ElectricalStaticTaskPreparation {
        machine_memory_bytes: 16 << 30,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "native-anchored-electrical".into(),
            encryption_context: digest,
        },
        workload,
        mesh,
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap()
}

#[test]
fn native_original_electrical_fields_receipts_and_cold_lineage_match_analytic_slabs() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for equation in [
            StaticElectricalEquation::Electrostatics,
            StaticElectricalEquation::SteadyConduction,
        ] {
            let (workload, mesh) = fixture::electrical_fixture(order, equation);
            let mut identity = None;
            for chunk in [1024, 8192] {
                let task = task(order, equation, chunk);
                let outputs = NativeAnalysisKernel
                    .execute(AnalysisKernelInvocation {
                        kind: AnalysisWorkKind::SolveElectricalStatic,
                        host: &task.host,
                        inputs: &task.inputs,
                        output_policy: task.output_policy,
                    })
                    .unwrap();
                assert_eq!(outputs.len(), 3);
                let mut outputs = outputs.into_iter();
                let encoded = outputs
                    .next()
                    .unwrap()
                    .into_prepared(chunk, task.output_policy.inventory_limits())
                    .unwrap()
                    .payload();
                let result = ElectricalStaticSolverResult::from_canonical_bytes_for(
                    &encoded, &workload, &mesh,
                )
                .unwrap();
                let receipt = outputs.next().unwrap();
                let receipt = ElectricalNumericalReceiptArtifact::from_canonical_bytes(
                    receipt.whole_payload().unwrap(),
                )
                .unwrap();
                receipt.validate_against(&workload, &result, &mesh).unwrap();
                for field in result
                    .response()
                    .element_electric_field_v_per_m
                    .canonical_values()
                {
                    assert!(
                        (field[0] + 2.).abs() < 1e-8
                            && field[1].abs() < 1e-8
                            && field[2].abs() < 1e-8
                    );
                }
                let (total, exact) = match &result.response().constitutive {
                    ElectricalConstitutiveResponse::Electrostatics { total_energy_j, .. } => (
                        total_energy_j.canonical_value(),
                        2. * VACUUM_PERMITTIVITY_F_PER_M.canonical_value(),
                    ),
                    ElectricalConstitutiveResponse::SteadyConduction {
                        total_joule_power_w,
                        ..
                    } => (total_joule_power_w.canonical_value(), 12.),
                };
                assert!((total - exact).abs() <= 1e-8 * exact);
                let current = result.source_result_identity().unwrap();
                if let Some(previous) = identity {
                    assert_eq!(previous, current);
                }
                identity = Some(current);
            }
        }
    }
}

#[test]
fn native_electrical_rejects_valid_hash_understated_original_proof_budget() {
    let task = task(
        ElementOrder::Tet10,
        StaticElectricalEquation::SteadyConduction,
        1024,
    );
    let mut input = task.resource_plan.input.clone();
    let FeaResourceDemand::ElectricalStatic(demand) = &mut input.demand else {
        panic!("scalar plan")
    };
    demand.solver_admission.resident_bytes_upper_bound.0 -= 1;
    let plan = runmat_analysis_fea::planning::resource::build_resource_plan(input).unwrap();
    let mut inputs = task.inputs.clone();
    let source = inputs
        .iter_mut()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    *source = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::InputValue,
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveElectricalStatic,
            host: &task.host,
            inputs: &inputs,
            output_policy: task.output_policy,
        })
        .is_err());
}
