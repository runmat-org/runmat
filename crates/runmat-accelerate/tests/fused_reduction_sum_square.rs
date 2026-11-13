#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate::fusion::{FusionKind, FusionOp};
use runmat_accelerate::fusion_exec::FusionExecutionRequest;
use runmat_accelerate::graph::{FusionCallOrigin, ShapeInfo, ValueId};
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_value::Value;
#[path = "support/fusion_plan.rs"]
mod fixture;
#[path = "support/selected_reduction.rs"]
mod selected_reduction;

static TEST_MUTEX: once_cell::sync::Lazy<tokio::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| tokio::sync::Mutex::new(()));

fn upload(
    provider: &runmat_accelerate::backend::wgpu::provider::WgpuProvider,
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

    // A reduction flavor is issued by the real terminal declaration, never
    // inferred from legacy metadata when plan.output is absent.
    let axis: ValueId = 2;
    let output: ValueId = 3;
    let mut plan = fixture::plan(
        vec![
            FusionOp::Primitive {
                op: runmat_accelerate::graph::PrimitiveOp::ElemMul,
                inputs: vec![vid_x, vid_x],
                output: Some(vid_mul),
            },
            FusionOp::Builtin {
                name: "sum".into(),
                origin: FusionCallOrigin::named("sum"),
                inputs: vec![vid_mul, axis],
                output: Some(output),
            },
        ],
        output,
    );
    plan.group.kind = FusionKind::Reduction;
    plan.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(cols)]);
    plan.kernel.kind = FusionKind::Reduction;
    plan.inputs = vec![vid_x];
    plan.const_values.insert(axis, Value::Num(1.0));
    plan.reduction_data = Some(vid_mul);
    plan.reduction_axes = Some(ReductionAxes::Explicit(vec![1]));

    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x.clone())],
        placement: None,
        runtime: None,
    };

    // Preserve the canonical default accumulator contract, including its
    // explicit unsupported-provider refusal; do not substitute native Single.
    let result = selected_reduction::execute(request, rows, cols, &[1, cols], 0)
        .await
        .expect("execute canonical fused reduction");
    let Some(result) = result else {
        return;
    };
    let out_handle = match result {
        Value::GpuTensor(h) => h,
        _ => panic!("expected GPU tensor"),
    };
    let out = AccelProvider::download(provider, &out_handle)
        .await
        .expect("download");
    assert_eq!(out.shape, vec![1, cols]);

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
