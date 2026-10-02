use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{NumericDType, Tensor};

#[test]
fn finite_host_scratch_tracks_real_buffers_and_one_output_location() {
    test_support::with_test_provider(|provider| {
        let source = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_with_dtype(vec![0.0; 4], vec![2, 2], NumericDType::F32).unwrap(),
        )
        .unwrap();
        // Only imhist-owned f64 binning (32) and unreturned locations (24)
        // belong here. Shared admission adds the provider's native readback.
        assert_eq!(host_scratch_bound(&source, 3, 1), Some(56));
        assert_eq!(host_scratch_bound(&source, 3, 2), Some(32));
        assert_eq!(
            provider.download_numeric_host_workspace_bound(&source),
            ResourceByteBound::Known(16)
        );
        let transposed = source
            .with_view(
                runmat_accelerate_api::GpuTensorView::contiguous(vec![2, 2]).with_layout(
                    runmat_accelerate_api::GpuTensorLayout::Transpose(
                        runmat_accelerate_api::TransposeInfo {
                            base_rows: 2,
                            base_cols: 2,
                        },
                    ),
                ),
            )
            .unwrap();
        assert_eq!(host_scratch_bound(&transposed, 3, 1), Some(56));
        assert_eq!(host_scratch_bound(&transposed, 3, 2), Some(32));
    });
}
