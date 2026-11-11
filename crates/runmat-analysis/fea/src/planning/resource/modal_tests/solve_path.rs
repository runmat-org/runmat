//! G29: the plan chooses the modal solve path from the admitted budget and records it.
use super::*;
use crate::planning::resource::{
    build_structural_modal_resource_plan_selecting, recorded_modal_solve_path, FeaResourcePlan,
    ModalSolveSelection, ResourcePlanningError, SubspaceSolvePath,
};

#[test]
fn modal_plan_selects_the_factorization_when_it_fits_lobpcg_otherwise_and_refuses_below_both() {
    let (workload, mesh, constraints, observations) = modal_case(1, ComputeBackend::Cpu);
    let compiled = super::super::constraint_compilation::compile_original_inventory(
        workload.model(),
        Some(&mesh),
        &constraints,
        workload
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .unwrap();
    let plan = |selection| {
        build_structural_modal_resource_plan_selecting(
            crate::planning::resource::StructuralModalResourcePlanningInput {
                mesh: Some(&mesh),
                compilation: &compiled,
                observations: &observations,
                workload: &workload,
                prestress_state: None,
                accelerator_requests: &[],
                prepared: prepared(&mesh, &constraints, &workload),
            },
            selection,
        )
    };
    let direct = plan(ModalSolveSelection::Recorded(
        SubspaceSolvePath::ShiftInvertFactorization,
    ))
    .unwrap();
    let lobpcg = plan(ModalSolveSelection::Recorded(
        SubspaceSolvePath::PreconditionedLobpcg,
    ))
    .unwrap();
    assert_eq!(
        recorded_modal_solve_path(&direct),
        Some(SubspaceSolvePath::ShiftInvertFactorization)
    );
    assert_eq!(
        recorded_modal_solve_path(&lobpcg),
        Some(SubspaceSolvePath::PreconditionedLobpcg)
    );
    let need = |plan: &FeaResourcePlan| plan.request.memory_bytes.max(plan.request.scratch_bytes);
    let within = |machine| {
        plan(ModalSolveSelection::WithinCeiling {
            machine_memory_bytes: machine,
        })
    };
    // Without a machine figure or authored limit the factorization always fits.
    assert_eq!(within(None).unwrap(), direct);
    assert_eq!(within(Some(need(&direct))).unwrap(), direct);
    // Below the factorization's need, LOBPCG is chosen when its plan fits.
    if need(&lobpcg) < need(&direct) {
        assert_eq!(within(Some(need(&direct) - 1)).unwrap(), lobpcg);
        assert_eq!(within(Some(need(&lobpcg))).unwrap(), lobpcg);
    }
    // Below both, a typed refusal names both requirements and the ceiling.
    let below = need(&direct).min(need(&lobpcg)) - 1;
    assert_eq!(
        within(Some(below)).unwrap_err(),
        ResourcePlanningError::ModalSolveCeiling {
            factorization_resident_bytes: need(&direct),
            lobpcg_resident_bytes: need(&lobpcg),
            maximum: below,
        }
    );
}
