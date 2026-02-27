#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider_impl::WgpuProviderOptions;
use runmat_accelerate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::fusion_exec::{execute_reduction, FusionExecutionRequest};
use runmat_accelerate::graph::{InstrSpan, ShapeInfo, ValueId};
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView, ReductionFlavor};
use runmat_builtins::Value;
use std::collections::HashMap;

static TEST_MUTEX: once_cell::sync::Lazy<tokio::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| tokio::sync::Mutex::new(()));

fn upload(
    provider: &runmat_accelerate::backend::wgpu::provider_impl::WgpuProvider,
    rows: usize,
    cols: usize,
    data: &[f64],
) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data,
            shape: &[rows, cols],
        })
        .expect("upload")
}

#[tokio::test]
async fn fused_sum_square_dim0_matches_manual() {
    let _guard = TEST_MUTEX.lock().await;
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");

    let rows = 5usize;
    let cols = 4usize;
    let mut xh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + (c as f64) + 1.0;
        }
    }
    let x = upload(provider, rows, cols, &xh);

    let vid_x: ValueId = 0;
    let vid_mul: ValueId = 1;

    // Build reduction plan for sum(x .* x) reducing rows (MATLAB dim=1)
    let group = FusionGroup {
        id: 0,
        kind: FusionKind::Reduction,
        nodes: vec![],
        shape: ShapeInfo::Tensor(vec![Some(cols)]),
        span: InstrSpan { start: 0, end: 0 },
        pattern: None,
    };

    let operations = vec![FusionOp::Primitive {
        op: runmat_accelerate::graph::PrimitiveOp::ElemMul,
        inputs: vec![vid_x, vid_x],
        output: Some(vid_mul),
    }];

    let mut constants: HashMap<usize, Value> = HashMap::new();
    constants.insert(0, Value::Num(1.0)); // dim=1 => reduce rows

    let plan = FusionGroupPlan {
        index: 0,
        group,
        operations,
        inputs: vec![vid_x],
        stack_pattern: vec![],
        constants,
        const_values: HashMap::new(),
        output: None,
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(vid_mul),
        reduction_dim: None,
        reduction_flavor: Some(ReductionFlavor::Sum),
        reduction_axes: Some(ReductionAxes::Explicit(vec![1])),
        pattern: None,
    };

    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x.clone())],
    };

    let result = execute_reduction(request, rows, cols, 0).expect("execute fused reduction");
    let out_handle = match result {
        Value::GpuTensor(h) => h,
        _ => panic!("expected GPU tensor"),
    };
    let out = AccelProvider::download(provider, &out_handle)
        .await
        .expect("download");
    assert_eq!(out.shape, vec![cols]);

    // Manual per-column sum of squares
    for c in 0..cols {
        let mut acc = 0.0f64;
        for r in 0..rows {
            let v = xh[r + c * rows];
            acc += v * v;
        }
        let got = out.data[c];
        assert!(
            (got - acc).abs() < 1e-6,
            "sumsq col={c} got={got} exp={acc}"
        );
    }
}
