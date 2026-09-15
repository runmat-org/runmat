use runmat_analysis_core::SignalSampleApproach;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    assembly::{assemble_structural_system, AssemblySummary, StructuralFormulationRequirement},
    contracts::{ComputeBackend, ExecutionBackend, FeaRunError, FeaTransientRunResult},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{
        StructuralTransientCheckpointScope, StructuralTransientCheckpointScopeInput,
        StructuralTransientNumericalReceipt, StructuralTransientSolverResult,
        StructuralTransientWorkload,
    },
    solve::{
        backend::kind::LinearAlgebraBackendKind,
        transient::{
            reconstruct_completed_transient_solve, CompletedTransientSolveInput,
            TransientSolveError,
        },
    },
};

pub struct CompletedStructuralTransientRunInput<'a> {
    pub workload: &'a StructuralTransientWorkload,
    pub solver_mesh: &'a SolverMeshArtifact,
    pub modal_state: Option<&'a crate::execution::CanonicalSemanticState>,
    pub result: &'a StructuralTransientSolverResult,
    pub numerical_receipt: &'a StructuralTransientNumericalReceipt,
}

/// Recovers the public physical result of a completed temporal block sequence.
///
/// This path deterministically reassembles reporting snapshots from the exact authored model and
/// mesh. It never advances Newmark state, changes the accepted schedule, or recomputes solver-owned
/// reactions, work, balance, counters, or accelerator evidence.
pub fn recover_completed_transient_run(
    input: CompletedStructuralTransientRunInput<'_>,
) -> Result<FeaTransientRunResult, FeaRunError> {
    input
        .workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    input
        .solver_mesh
        .validate_canonical()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    input
        .numerical_receipt
        .validate_for_result(input.result)
        .map_err(|error| FeaRunError::Solve(error.to_string()))?;

    let options = input.workload.solve_options(input.solver_mesh.clone());
    let basis = super::prepare_transient_block_basis(&input.workload.model, &options)?;
    let expected_scope = StructuralTransientCheckpointScope::from_prepared_workload(
        StructuralTransientCheckpointScopeInput {
            workload: input.workload,
            solver_mesh: input.solver_mesh,
            initial_assembly: &basis.initial_assembly,
            modal_state: input.modal_state,
            schedule: basis.schedule.clone(),
        },
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    if input.result.manifest().scope != expected_scope {
        return Err(FeaRunError::Solve(
            "completed transient result differs from its exact workload, mesh, or schedule".into(),
        ));
    }
    let expected_backend = match input.workload.backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let execution_backend = input
        .numerical_receipt
        .blocks
        .first()
        .map(|block| block.solver_backend)
        .ok_or_else(|| FeaRunError::Solve("transient receipt contains no solver blocks".into()))?;
    if execution_backend != expected_backend {
        return Err(FeaRunError::Solve(
            "completed transient solver backend differs from its requested workload backend".into(),
        ));
    }

    let initial_state = super::realize_transient_initial_state(
        &basis.active_model,
        &basis.execution_step,
        &basis.initial_assembly,
        Some(input.solver_mesh),
        &basis.motions,
        basis.schedule.points[0].time_s,
    )?;
    let initial_sample =
        input.result.samples().first().ok_or_else(|| {
            FeaRunError::Solve("transient result contains no initial sample".into())
        })?;
    if initial_state.displacement != initial_sample.state.displacement
        || initial_state.velocity != initial_sample.state.velocity
    {
        return Err(FeaRunError::Solve(
            "completed transient initial state differs from the authored realization".into(),
        ));
    }

    let mut assembly_snapshots = Vec::with_capacity(input.result.samples().len());
    for sample in input.result.samples() {
        let excitations = realize_step_excitations(
            &basis.active_model,
            &basis.execution_step,
            ExcitationSamplePoint::TimeSeconds {
                value: sample.time_s,
                approach: SignalSampleApproach::At,
            },
        )
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        let summary = assemble_structural_system(
            &basis.active_model,
            &excitations,
            Some(input.solver_mesh.clone()),
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        let summary = if basis.motions.is_empty() {
            summary
        } else {
            super::motion::physical_summary_for_recovery(
                summary,
                &basis.motions,
                sample.time_s,
                SignalSampleApproach::At,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?
        };
        validate_reporting_summary(&basis.initial_assembly, &summary)?;
        assembly_snapshots.push(summary);
    }
    let transient = reconstruct_completed_transient_solve(CompletedTransientSolveInput {
        options: &options,
        result: input.result,
        receipt: input.numerical_receipt,
        assembly_snapshots,
    })
    .map_err(|error| match error {
        TransientSolveError::Assembly(never) => match never {},
        other => FeaRunError::Solve(other.to_string()),
    })?;
    let identity = super::super::admission::admit_structural(
        &basis.initial_assembly,
        &basis.execution_step,
        input.workload.formulation_requirement,
    )?;
    let recovered = super::finish_transient_run(
        super::TransientRunFinalization {
            model: &input.workload.model,
            execution_step: &basis.execution_step,
            active_model: &basis.active_model,
            backend: input.workload.backend,
            options: &options,
            identity,
            initial_state: initial_state.evidence,
        },
        transient,
    )?;
    if recovered.run.execution_backend != ExecutionBackend::from_solver_backend(expected_backend) {
        return Err(FeaRunError::Solve(
            "recovered transient execution backend differs from its canonical receipt".into(),
        ));
    }
    Ok(recovered)
}

fn validate_reporting_summary(
    initial: &AssemblySummary,
    candidate: &AssemblySummary,
) -> Result<(), FeaRunError> {
    if candidate.dof_count != initial.dof_count
        || candidate.structural_dof_layout != initial.structural_dof_layout
        || candidate.structural_formulations != initial.structural_formulations
        || candidate.operator.operators().stiffness() != initial.operator.operators().stiffness()
        || candidate.operator.operators().mass() != initial.operator.operators().mass()
        || candidate.operator.operators().damping() != initial.operator.operators().damping()
    {
        return Err(FeaRunError::Assembly(
            "transient reporting reassembly changed the immutable structural system".into(),
        ));
    }
    Ok(())
}
