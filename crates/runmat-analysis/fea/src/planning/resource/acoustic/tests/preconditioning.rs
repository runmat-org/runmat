use super::*;
use crate::contracts::SolverPreconditioner;

#[test]
fn real_block_phase_fill_and_lu_are_additive_checked_and_identity_bound() {
    let support = 16;
    let block = 8;
    for row_support in [1, 2] {
        let base = sparse_workspace_bound(support, block, row_support, SolverPreconditioner::None)
            .unwrap();
        let ilu = sparse_workspace_bound(support, block, row_support, SolverPreconditioner::Ilu0)
            .unwrap();
        let reduced = 2 * support * row_support * row_support;
        assert_eq!(
            base,
            (2 * support + reduced) * 160 + (block + 1) * 256 + 4096
        );
        assert_eq!(ilu - base, reduced * 32 + (block + 1) * 32);
    }
    assert!(sparse_workspace_bound(u64::MAX, block, 2, SolverPreconditioner::Ilu0).is_err());
    assert!(sparse_workspace_bound(support, u64::MAX, 2, SolverPreconditioner::Ilu0).is_err());
    assert!(sparse_workspace_bound(support, block, 2, SolverPreconditioner::Jacobi).is_err());
    for invalid in [0, 3, u64::MAX] {
        assert!(
            sparse_workspace_bound(support, block, invalid, SolverPreconditioner::Ilu0).is_err()
        );
    }
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let workload = workload();
        let plan = build_acoustic_resource_plan(&workload, &mesh, facts(&workload, &mesh)).unwrap();
        let FeaResourceDemand::AcousticHarmonic(demand) = &plan.input.demand else {
            panic!("acoustic demand")
        };
        assert_eq!(demand.preconditioner, SolverPreconditioner::Ilu0);
        assert_eq!(demand.recovery_row_support_upper_bound.0, 1);
        assert_eq!(
            demand.sparse_solve_reservation_bytes.0,
            sparse_workspace_bound(
                demand.operator_nonzeros_upper_bound.0,
                demand.block_dimension_upper_bound.0,
                demand.recovery_row_support_upper_bound.0,
                demand.preconditioner
            )
            .unwrap()
        );
        for alteration in 0..5 {
            let mut input = plan.input.clone();
            let FeaResourceDemand::AcousticHarmonic(demand) = &mut input.demand else {
                unreachable!()
            };
            match alteration {
                0 => demand.sparse_solve_reservation_bytes.0 -= 1,
                1 => {
                    demand.numerical_reservation_bytes.0 =
                        demand.sparse_solve_reservation_bytes.0 - 1
                }
                2 => demand.preconditioner = SolverPreconditioner::None,
                3 => demand.operator_nonzeros_upper_bound.0 -= 1,
                _ => demand.recovery_row_support_upper_bound.0 = 2,
            }
            assert!(build_resource_plan(input).is_err());
        }
        let mut none = workload.clone();
        none.solver.preconditioner = SolverPreconditioner::None;
        let none_plan = build_acoustic_resource_plan(&none, &mesh, facts(&none, &mesh)).unwrap();
        assert_ne!(plan.identity, none_plan.identity);
        assert!(plan.breakdown.workspace_bytes > none_plan.breakdown.workspace_bytes);
        let mut limited = plan.input.clone();
        limited.accuracy_ceiling.maximum_resident_bytes = plan.request.memory_bytes - 1;
        assert!(matches!(
            build_resource_plan(limited),
            Err(ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                ..
            })
        ));
    }
}
