//! Closed physical walls and a distinct gauge through actual native admission.
use super::*;
use runmat_meshing_core::fixtures;

pub(super) fn closed_task() -> (
    FluidSteadyWorkload,
    SolverMeshArtifact,
    PreparedFluidSteadyTask,
) {
    let (mut workload, _, prior) = task(false, 1024);
    let mesh = fixtures::slab_mesh(2);
    let outlet = workload.model.boundaries.pop().unwrap();
    workload
        .model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "closed-wall".into(),
            kind: FluidConstraintKind::NoSlipWall {
                target: outlet.target.clone(),
            },
        }));
    workload
        .model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "closed-pressure-gauge".into(),
            kind: FluidConstraintKind::PressureReference {
                target: outlet.target,
                value: PrescribedValue::Constant(Pressure::new(101325.0)),
            },
        }));
    let task = prepare_fluid_steady_task(FluidSteadyTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: Digest::sha256(b"closed original fluid"),
        frozen_project_digest: None,
        program_revision: prior.program.request.recipe.program_revision.clone(),
        artifact_access: prior.host.artifact_access.clone(),
        workload: workload.clone(),
        mesh: mesh.clone(),
        output_policy: prior.output_policy,
        required_capabilities: Default::default(),
    })
    .unwrap();
    (workload, mesh, task)
}

#[test]
fn native_closed_pressure_gauge_retains_original_wall_laws_and_cold_fields() {
    let (w, m, task) = closed_task();
    let mut outputs = invoke(&task).unwrap().into_iter();
    let original = outputs
        .next()
        .unwrap()
        .into_prepared(1024, task.output_policy.inventory_limits())
        .unwrap();
    let bytes = original.payload();
    let r =
        FluidSteadySolverResult::from_canonical_bytes_for(&bytes, Digest::sha256(&bytes), &w, &m)
            .unwrap();
    assert!(r
        .original()
        .pressure_pa
        .canonical_values()
        .iter()
        .all(|p| *p == 101325.0));
    assert!(r
        .original()
        .pressure_fluctuation_pa
        .canonical_values()
        .iter()
        .all(|p| *p == 0.0));
    assert!(r
        .original()
        .face_volume_flux_m3_per_s
        .canonical_values()
        .iter()
        .all(|q| *q == 0.0));
    let receipt = outputs
        .next()
        .unwrap()
        .into_prepared(1024, task.output_policy.inventory_limits())
        .unwrap();
    let receipt = FluidNumericalReceiptArtifact::from_canonical_bytes(&receipt.payload()).unwrap();
    receipt.validate_against(&w, &r, &m).unwrap();
    assert!(outputs.next().is_none());
}
