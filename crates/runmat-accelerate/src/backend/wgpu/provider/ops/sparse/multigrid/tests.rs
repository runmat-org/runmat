use super::*;
use crate::backend::wgpu::provider::{WgpuPrecisionPolicy, WgpuProviderOptions};
use runmat_accelerate_api::MultigridLevel;

#[test]
#[ignore = "requires an actual WGPU device; no unavailable-device pass"]
fn uploaded_hierarchy_executes_and_reuses_the_original_device_operands() {
    let provider = WgpuProvider::new(WgpuProviderOptions {
        precision: WgpuPrecisionPolicy::RequireF32,
        ..Default::default()
    })
    .expect("actual single-precision WGPU device required");
    let request = SparseMultigridRequest {
        levels: vec![MultigridLevel {
            matrix: MultigridCsr {
                rows: 2,
                columns: 2,
                row_offsets: &[0, 2, 4],
                column_indices: &[0, 1, 0, 1],
                values: &[4.0, 1.0, 1.0, 3.0],
            },
            block: 1,
            inverse_blocks: &[0.25, 1.0 / 3.0],
            lambda_max: 2.0,
            prolongation: None,
            restriction: None,
        }],
        coarse_inverse: &[3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0],
        smoother_degree: 2,
        chebyshev_ratio: 10.0,
    };
    let handle = prepare(&provider, &request).expect("upload hierarchy through emission");
    for (rhs, apply, expected) in [
        (
            [1.0, 2.0],
            SparseMultigridApply::VCycle,
            [1.0 / 11.0, 7.0 / 11.0],
        ),
        (
            [2.0, 1.0],
            SparseMultigridApply::VCycle,
            [5.0 / 11.0, 2.0 / 11.0],
        ),
        (
            [1.0, 2.0],
            SparseMultigridApply::Operator { repetitions: 2 },
            [6.0, 7.0],
        ),
    ] {
        let result = futures::executor::block_on(solve(
            &provider,
            &SparseMultigridSolveRequest {
                hierarchy: &handle,
                rhs: &rhs,
                apply,
            },
        ))
        .expect("execute original uploaded operands");
        assert!(result.dispatches > 0);
        assert_eq!(result.solution.len(), expected.len());
        for (actual, expected) in result.solution.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
    free(&provider, &handle).unwrap();
    assert!(hierarchy(&provider, &handle).is_err());
}
