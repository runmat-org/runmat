use super::*;
use std::error::Error as _;

#[test]
#[ignore = "requires an actual SHADER_F64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_allocation_and_submission_faults_keep_source_and_original_cause() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F64);
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0; 8]),
            shape: &[2, 2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let plan = provider
        .ordered_mean_plan(
            &input,
            &[1, 2],
            ProviderPrecision::F32,
            ProviderRealMeanNaNMode::Include,
        )
        .unwrap()
        .unwrap();
    let before_download = provider.telemetry_snapshot().download_bytes;
    for (fault, occurrence) in [
        (OrderedMeanCheckpoint::OutputAllocated, 1),
        (OrderedMeanCheckpoint::StageSubmitted, 1),
        (OrderedMeanCheckpoint::OutputAllocated, 2),
    ] {
        let mut seen = 0;
        let failure = provider
            .execute_ordered_mean_plan_with(
                &input,
                &plan,
                ProviderRealMeanNaNMode::Include,
                |checkpoint| {
                    if checkpoint == fault {
                        seen += 1;
                        if seen == occurrence {
                            return Err(anyhow::Error::new(std::io::Error::new(
                                std::io::ErrorKind::BrokenPipe,
                                "ordered mean injected fault",
                            )));
                        }
                    }
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(!failure.is_unsupported());
        assert_eq!(
            failure
                .source()
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::BrokenPipe)
        );
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            before_download
        );
    }
}
