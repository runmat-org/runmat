use super::*;
use crate::scalar_program::ScalarLiteral;
use crate::stochastic_evolution::{
    ProviderStochasticEvolutionPhysicalProposal, ProviderStochasticEvolutionPlanRequest,
    ProviderStochasticEvolutionSelection, StochasticEvolutionArithmetic,
    PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};
use crate::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use futures::FutureExt;
use runmat_types::NumericClass;

fn selection() -> ProviderStochasticEvolutionSelection {
    let representation = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![65, 1],
        residency: ProviderResidency::Device,
    };
    let arithmetic = StochasticEvolutionArithmetic::new(
        NumericClass::Single,
        NumericClass::Single,
        ScalarLiteral::F32Bits(0),
        ScalarLiteral::F32Bits(0),
    )
    .unwrap();
    let plan = ProviderStochasticEvolutionPlanRequest {
        state: representation.clone(),
        output: representation,
        first_arithmetic: arithmetic,
        continuing_arithmetic: arithmetic,
        steps: 2,
        noise_tile_elements: 64,
    };
    let proposal = ProviderStochasticEvolutionPhysicalProposal {
        plan_identity: plan.identity().unwrap(),
        first_output_buffer_bytes: 65 * 4,
        continuing_output_buffer_bytes: 65 * 4,
        final_output_buffer_bytes: 65 * 4,
        max_noise_tile_buffer_bytes: 64 * 4,
        uniform_buffer_bytes_per_tile: 32,
        completion_fence_bytes_per_tile: 8,
        observation_fence_bytes_per_tile: 8,
        completion_control_host_bytes: 1,
        authentication_host_bytes_per_tile: 8 * std::mem::size_of::<usize>() as u64,
        submission_retention_host_bytes_per_tile: 1,
        source_transfer_retirement_host_bytes: 1,
        observation_host_control_bytes_per_tile: 1,
        output_view_host_bytes: crate::GpuTensorHandle::view_host_upper_bytes(2).unwrap(),
        tiles_per_step: 2,
        dispatches_per_step: 2,
        queue_submissions_per_step: 2,
        full_call_dispatches: 4,
        full_call_queue_submissions: 4,
        peak_provisional_output_buffers: 1,
        frozen_workgroup_size: 64,
        frozen_tile_capacity: 64,
    };
    ProviderStochasticEvolutionSelection { plan, proposal }
}

#[test]
fn stochastic_source_transfers_share_completion_without_proving_arithmetic() {
    let selected = selection();
    let mut upload_only = ProviderOperationReceipt::new();
    upload_only.require_stochastic_evolution_source_transfers(7, &selected);
    let recorder = upload_only.recorder();
    assert!(recorder.expects_stochastic_evolution_selection(&selected));
    assert!(recorder.observes_stochastic_evolution_source_transfer(7));
    assert!(!recorder.observes_stochastic_evolution_source_transfer(8));
    recorder.submission().unwrap().complete();
    assert_eq!(
        upload_only.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );

    let mut wrong_operation = ProviderOperationReceipt::new();
    wrong_operation.require_stochastic_evolution_source_transfers(7, &selected);
    let recorder = wrong_operation.recorder();
    let before_compute = recorder.submission_checkpoint();
    recorder.submission().unwrap().complete();
    recorder.confirm_submitted_provider_execution_after(
        7,
        &crate::ProviderOperationIdentity::new("fixture.not-stochastic-evolution"),
        before_compute,
    );
    assert_eq!(
        wrong_operation.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );

    let mut whole_call = ProviderOperationReceipt::new();
    whole_call.require_stochastic_evolution_source_transfers(7, &selected);
    let recorder = whole_call.recorder();
    let mut forged = selection();
    forged.proposal.max_noise_tile_buffer_bytes -= 1;
    assert!(!recorder.expects_stochastic_evolution_selection(&forged));
    let initial_upload = recorder.submission().unwrap();
    let after_upload = recorder.submission_checkpoint();
    let compute = recorder.submission().unwrap();
    recorder.confirm_submitted_provider_execution_after(
        7,
        &crate::ProviderOperationIdentity::new(PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION),
        after_upload,
    );
    assert!(recorder.observes_stochastic_evolution_source_transfer(7));
    let later_noise_upload = recorder.submission().unwrap();
    whole_call.publish().unwrap();
    assert_eq!(whole_call.status(), ProviderObservationStatus::Pending);
    initial_upload.complete();
    compute.complete();
    assert_eq!(whole_call.status(), ProviderObservationStatus::Pending);
    later_noise_upload.complete();
    assert!(matches!(
        whole_call.status(),
        ProviderObservationStatus::Completed { submissions: 3, .. }
    ));

    let mut rejected = ProviderOperationReceipt::new();
    rejected.require_stochastic_evolution_source_transfers(7, &forged);
    assert!(!rejected
        .recorder()
        .observes_stochastic_evolution_source_transfer(7));
    assert_eq!(
        rejected.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    );
}

#[test]
fn stochastic_source_transfer_retirement_is_exact_and_one_shot() {
    let selected = selection();
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_stochastic_evolution_source_transfers(7, &selected);
    let recorder = receipt.recorder();
    let mut retirement = recorder
        .arm_stochastic_evolution_source_transfer_retirement(7)
        .expect("exact selected transfer can arm");
    assert!((&mut retirement).now_or_never().is_none());
    let notifier = recorder
        .take_stochastic_evolution_source_transfer_retirement(7)
        .expect("provider takes exact armed transfer");
    assert!(recorder
        .take_stochastic_evolution_source_transfer_retirement(7)
        .is_none());
    notifier.finish(true);
    assert!(futures::executor::block_on(retirement).is_ok());

    // Fencing a transfer is not arithmetic execution evidence.
    recorder.submission().unwrap().complete();
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::ExecutionUnproven)
    );
}
