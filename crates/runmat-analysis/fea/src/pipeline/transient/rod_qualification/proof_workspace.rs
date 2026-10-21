//! Compare preassembly workspace admission with proofs of actual physical operators.
use super::*;
use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    },
    assembly::{assemble_structural_system_borrowed, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{
        StructuralTransientBlockRequest, StructuralTransientSolverPlan, StructuralTransientWorkload,
    },
    planning::resource::{
        build_structural_transient_block_resource_plan, FeaResourceDemand, SolveResourcePolicy,
    },
};

#[test]
fn direct_block_workspace_covers_actual_mass_and_newmark_proofs_for_both_orders() {
    for quadratic in [false, true] {
        for cells in [1, 2, 4] {
            let mesh = mesh(quadratic, cells);
            let model = model(&mesh, true);
            let workload = StructuralTransientWorkload {
                schema_version: 2,
                initial_fields: Vec::new(),
                step_id: model.sequence.steps[0].step_id.clone(),
                model,
                backend: ComputeBackend::Cpu,
                solver: StructuralTransientSolverPlan::Direct,
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
                residual_target: 1.0e-10,
            };
            let scope =
                super::super::prepare_transient_checkpoint_scope(&workload, &mesh, None).unwrap();
            let request = StructuralTransientBlockRequest::new(scope, None, 2).unwrap();
            let plan = build_structural_transient_block_resource_plan(
                &workload,
                &mesh,
                None,
                &request,
                None,
                None,
                None,
                crate::planning::resource::LocalExecutionPlan::serial(),
            )
            .unwrap();
            let FeaResourceDemand::StructuralTransientBlock(demand) = &plan.input.demand else {
                panic!("expected transient block");
            };
            let SolveResourcePolicy::TransientBlock {
                workspace_bytes_upper_bound,
                ..
            } = demand.operator.solve
            else {
                panic!("expected transient workspace");
            };
            let loads = realize_step_excitations(
                &workload.model,
                &workload.model.sequence.steps[0],
                ExcitationSamplePoint::time(0.0),
            )
            .unwrap();
            let summary = assemble_structural_system_borrowed(
                &workload.model,
                &loads,
                Some(&mesh),
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .unwrap();
            if cells == 4 {
                // The former direct plan reserved 64 physical-dimension dense modal
                // matrices. Actual direct sparse execution has no such allocation.
                let unused_dense_bytes = 64 * 8 * (summary.dof_count as u64).pow(2);
                assert!(workspace_bytes_upper_bound.0 < unused_dense_bytes,
                    "direct sparse proof workspace must not retain the former dense modal reservation");
            }
            let source = summary.operator.operators();
            let dt = END_TIME / 2.0;
            let effective = RealSparseMatrix::scaled_sum(
                &[
                    (1.0, source.stiffness()),
                    (4.0 / (dt * dt), source.mass()),
                    (2.0 / dt, source.damping()),
                ],
                OperatorRole::EffectiveTangent,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap();
            let initial_mass = RealSparseMatrix::scaled_sum(
                &[(1.0, source.mass())],
                OperatorRole::EffectiveTangent,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap();
            for operator in [initial_mass, effective] {
                let operators = CanonicalRealOperators::try_new(
                    operator,
                    source.mass().clone(),
                    source.damping().clone(),
                )
                .unwrap();
                let system = AssembledRealSystem::try_new(
                    operators,
                    summary.operator.constrained_dofs().to_vec(),
                    vec![0.0; summary.dof_count],
                )
                .unwrap();
                let prepared = system.prepare_linear().unwrap();
                let certificate = prepared.admissibility().unwrap().coercivity();
                let (_, _, _, actual_proof_bytes) = certificate.work_counts();
                assert!(workspace_bytes_upper_bound.0 >= actual_proof_bytes,
                    "order={quadratic} cells={cells} reservation={} actual_proof={actual_proof_bytes}", workspace_bytes_upper_bound.0);
                eprintln!("direct proof order={quadratic} cells={cells} workspace={} actual_proof={actual_proof_bytes}", workspace_bytes_upper_bound.0);
            }
        }
    }
}
