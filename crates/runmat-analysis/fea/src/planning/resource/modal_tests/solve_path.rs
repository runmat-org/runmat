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
            Some(&mesh),
            &compiled,
            &observations,
            &workload,
            None,
            &[],
            prepared(&mesh, &constraints, &workload),
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

/// An authored `solver = factorization | iterative` plans that path alone, and below its need
/// the study is refused with the typed memory budget, never moved to the other path.
#[test]
fn authored_modal_solver_method_is_planned_alone_and_refused_with_the_memory_budget() {
    use runmat_analysis_core::ModalSolverMethod;
    for (method, path) in [
        (
            ModalSolverMethod::Factorization,
            SubspaceSolvePath::ShiftInvertFactorization,
        ),
        (
            ModalSolverMethod::Iterative,
            SubspaceSolvePath::PreconditionedLobpcg,
        ),
    ] {
        let (workload, mesh, constraints, observations) =
            modal_case_with(1, ComputeBackend::Cpu, method);
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
                Some(&mesh),
                &compiled,
                &observations,
                &workload,
                None,
                &[],
                prepared(&mesh, &constraints, &workload),
                selection,
            )
        };
        let recorded = plan(ModalSolveSelection::Recorded(path)).unwrap();
        let forced = plan(ModalSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        })
        .unwrap();
        assert_eq!(forced, recorded, "{method:?}");
        assert_eq!(recorded_modal_solve_path(&forced), Some(path));
        let need = forced
            .request
            .memory_bytes
            .max(forced.request.scratch_bytes);
        assert_eq!(
            plan(ModalSolveSelection::WithinCeiling {
                machine_memory_bytes: Some(need),
            })
            .unwrap(),
            forced
        );
        assert_eq!(
            plan(ModalSolveSelection::WithinCeiling {
                machine_memory_bytes: Some(need - 1),
            })
            .unwrap_err(),
            ResourcePlanningError::MemoryBudget {
                required_memory_bytes: forced.request.memory_bytes,
                available_memory_bytes: need - 1,
                required_scratch_bytes: forced.request.scratch_bytes,
                available_scratch_bytes: need - 1,
            },
            "{method:?}"
        );
    }
}
