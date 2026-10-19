use super::*;

#[test]
fn query_cost_covers_direct_and_host_submission_encoders() {
    // Direct filtering has compute encoders and a separate kernel readback;
    // host mediation has no compute encoder but still submits GPU transfers.
    let direct = crate::backend::wgpu::dispatch::observation::prospective_encoder_query(2, 3)
        .expect("direct physical geometry");
    assert_eq!(direct.max_compute_encoders, 2);
    assert_eq!(direct.max_total_encoders, 3);
    assert_eq!(direct.max_query_sets, 3);
    let host = crate::backend::wgpu::dispatch::observation::prospective_encoder_query(0, 3)
        .expect("host-mediated physical geometry");
    assert_eq!(host.max_compute_encoders, 0);
    assert_eq!(host.max_total_encoders, 3);
    assert_eq!(host.max_query_sets, 3);
    assert!(crate::backend::wgpu::dispatch::observation::prospective_encoder_query(0, 0).is_none());
    assert!(crate::backend::wgpu::dispatch::observation::prospective_encoder_query(4, 3).is_none());
}

#[test]
fn direct_route_geometry_separates_readback_flush_compute_and_empty() {
    use crate::backend::wgpu_geometry::permutation_submission_geometry;
    let capacity = crate::backend::wgpu_geometry::elementwise_chunk_capacity(512).unwrap();
    let strided = permutation_submission_geometry(capacity + 1, 512).unwrap();
    for (filter, readback, permutation, expected) in [
        (2, 1, None, (2, 3, 2, 2)),
        (2, 1, Some(strided), (6, 7, 5, 4)),
        (0, 0, permutation_submission_geometry(0, 512), (0, 0, 0, 0)),
    ] {
        let geometry = direct_submission_geometry(filter, readback, permutation).unwrap();
        assert_eq!(
            (
                geometry.output_queue_slots,
                geometry.submissions,
                geometry.compute_encoders,
                geometry.dispatches,
            ),
            expected
        );
    }
}

#[test]
fn zero_length_readback_stage_has_no_staging_or_queue_demand() {
    // This is the physical stage projection only; build_imfilter_plan rejects
    // an empty kernel before the public provider route is admitted.
    let empty = kernel_readback_stage(7, 0, 8, 8).unwrap();
    assert_eq!(
        empty.iter().collect::<Vec<_>>(),
        vec![ProviderInputStageDemand {
            device_id: 7,
            incremental_peak_bytes: ResourceByteBound::Known(0),
            queue_slots: 0,
        }]
    );
    let nonempty = kernel_readback_stage(7, 1, 8, 32).unwrap();
    assert_eq!(
        nonempty.iter().collect::<Vec<_>>(),
        vec![ProviderInputStageDemand {
            device_id: 7,
            incremental_peak_bytes: ResourceByteBound::Known(40),
            queue_slots: 1,
        }]
    );
}
