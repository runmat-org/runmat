#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider_impl::WgpuProviderOptions;
use runmat_accelerate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::fusion_exec::{execute_reduction, FusionExecutionRequest};
use runmat_accelerate::graph::{InstrSpan, PrimitiveOp, ShapeInfo, ValueId};
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView, ReductionFlavor};
use runmat_builtins::Value;
use std::collections::HashMap;

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

// Minimal single-pass fused reduction repro: reduce_len=1, num_slices>1
// Mirrors the captured Metal binding layout: layout_tag=runmat-reduction-layout-1
#[test]
fn fused_single_pass_reduce_len_1_no_alias() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");

    let rows = 1usize;
    let cols = 3907usize;

    let mut x_data = vec![0.0f64; rows * cols];
    let mut w_data = vec![0.0f64; rows * cols];
    for i in 0..(rows * cols) {
        x_data[i] = (i as f64 * 0.001).sin();
        w_data[i] = (i as f64 * 0.002).cos();
    }

    let x = upload(provider, rows, cols, &x_data);
    let w = upload(provider, rows, cols, &w_data);

    let vid_x: ValueId = 0;
    let vid_w: ValueId = 1;
    let vid_mul: ValueId = 2;
    let vid_dim: ValueId = 3;
    let vid_sum: ValueId = 4;

    let group = FusionGroup {
        id: 0,
        kind: FusionKind::Reduction,
        nodes: vec![],
        shape: ShapeInfo::Tensor(vec![Some(cols)]),
        span: InstrSpan { start: 0, end: 0 },
        pattern: None,
    };

    let plan = FusionGroupPlan {
        index: 0,
        group,
        operations: vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemMul,
                inputs: vec![vid_x, vid_w],
                output: Some(vid_mul),
            },
            FusionOp::Builtin {
                name: "sum".to_string(),
                inputs: vec![vid_mul, vid_dim],
                output: Some(vid_sum),
            },
        ],
        inputs: vec![vid_x, vid_w],
        stack_pattern: vec![],
        constants: {
            let mut m = HashMap::new();
            m.insert(0, Value::Num(1.0)); // dim=1 => reduce rows
            m
        },
        const_values: {
            let mut cv = HashMap::new();
            cv.insert(vid_dim, Value::Num(1.0));
            cv
        },
        output: Some(vid_sum),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(vid_mul),
        reduction_dim: Some(vid_dim),
        reduction_flavor: Some(ReductionFlavor::Sum),
        reduction_axes: Some(ReductionAxes::Explicit(vec![1])),
        pattern: None,
    };

    let req = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x), Value::GpuTensor(w)],
    };
    // reduce_len=rows=1, num_slices=cols
    let _ = execute_reduction(req, rows, cols, 0).expect("fused sum(x.*W,1) should run");
}
