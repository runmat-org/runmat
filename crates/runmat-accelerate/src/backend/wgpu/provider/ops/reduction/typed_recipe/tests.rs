//! Actual-device publication ownership witness.
use super::*;
use futures::executor::block_on;

#[test]
#[ignore = "requires actual WGPU adapter; unavailable is failure"]
fn published_output_retains_exact_submission_and_retires() {
    let provider = block_on(WgpuProvider::new_async(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ))
    .expect("actual WGPU adapter required; unavailable is not a pass");
    let before = provider.allocations.record_count().unwrap();
    let (buffer, first_id, exact_id) = provider
        .with_emission(|emission, pending| -> Result<_> {
            let buffer = provider.create_storage_buffer_checked_with_element_size(
                emission,
                1,
                "typed-reduction-tracking-test",
                BufferUsageClass::FusionOut,
                4,
            )?;
            let first_encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("typed-reduction-tracking-prior"),
                },
            );
            let first_id = provider.submit(emission, pending, first_encoder)?;
            let encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("typed-reduction-tracking-test"),
                },
            );
            let id = provider.submit(emission, pending, encoder)?;
            Ok((buffer, first_id, id))
        })
        .unwrap()
        .unwrap();
    assert_ne!(first_id, exact_id);
    let output = provider
        .register_reduction_output(buffer, vec![1, 1], 1, NumericElementType::F32, None)
        .expect("register output once");
    let output = provider
        .publish_reduction_output(output, exact_id)
        .expect("publish tracked output");
    assert_eq!(
        provider.get_entry_raw(&output).unwrap().last_submission_id,
        Some(exact_id),
    );
    assert_eq!(provider.allocations.record_count().unwrap(), before + 1);
    let id = output.buffer_id();
    assert_eq!(
        output.release().expect("retire tracked output"),
        runmat_accelerate_api::allocation::AllocationReleaseOutcome::Retired,
    );
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    assert!(provider
        .allocations
        .record_submission(id, exact_id)
        .is_err());
}
