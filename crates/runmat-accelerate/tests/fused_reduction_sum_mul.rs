#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider_impl::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp, ReductionMode,
};
use runmat_accelerate::fusion_exec::{execute_reduction, FusionExecutionRequest};
use runmat_accelerate::graph::{InstrSpan, ShapeInfo, ValueId};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::Value;
use std::collections::HashMap;

// Guard tests to avoid provider state races
static TEST_MUTEX: once_cell::sync::Lazy<std::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(()));

fn upload_matrix(
    provider: &WgpuProvider,
    rows: usize,
    cols: usize,
    data: &[f64],
) -> GpuTensorHandle {
    assert_eq!(data.len(), rows * cols);
    provider
        .upload(&HostTensorView {
            data,
            shape: &[rows, cols],
        })
        .expect("upload")
}

#[test]
fn fused_sum_mul_dim0_matches_manual() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");
    let rows = 5usize;
    let cols = 4usize;

    // X[r,c] = r+1, W[r,c] = (c+1)
    let mut xh = vec![0.0f64; rows * cols];
    let mut wh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + 1.0;
            wh[r + c * rows] = (c as f64) + 1.0;
        }
    }
    let x = upload_matrix(&provider, rows, cols, &xh);
    let w = upload_matrix(&provider, rows, cols, &wh);

    // Build a minimal FusionGroupPlan for reduction of (X .* W) along dim=1 (MATLAB '1' => reduce rows)
    // Inputs: v = X, v1 = W
    // Expr: tmp0 = v * v1; reduction_data = tmp0
    let vid_x: ValueId = 0;
    let vid_w: ValueId = 1;
    let vid_mul: ValueId = 2;

    let group = FusionGroup {
        id: 0,
        kind: FusionKind::Reduction,
        nodes: vec![],
        // Use a 1-D output shape placeholder (num_slices vector)
        shape: ShapeInfo::Tensor(vec![Some(cols)]),
        span: InstrSpan { start: 0, end: 0 },
        pattern: None,
    };

    let operations = vec![FusionOp::Primitive {
        op: runmat_accelerate::graph::PrimitiveOp::ElemMul,
        inputs: vec![vid_x, vid_w],
        output: Some(vid_mul),
    }];

    // Axis constant: MATLAB dim=1 => reduce rows (axis=0 in generator)
    let mut constants: HashMap<usize, Value> = HashMap::new();
    constants.insert(0, Value::Num(1.0));

    let plan = FusionGroupPlan {
        index: 0,
        group,
        operations,
        inputs: vec![vid_x, vid_w],
        stack_pattern: vec![],
        constants,
        const_values: HashMap::new(),
        output: None,
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(vid_mul),
        reduction_mode: Some(ReductionMode::Sum),
        pattern: None,
    };

    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x.clone()), Value::GpuTensor(w.clone())],
    };

    // reduce_len = rows, num_slices = cols, workgroup_size = default(0)
    let result = execute_reduction(request, rows, cols, 0).expect("execute fused reduction");
    let out_handle = match result {
        Value::GpuTensor(h) => h,
        _ => panic!("expected GPU tensor"),
    };
    let out = provider.download(&out_handle).expect("download");
    assert_eq!(out.shape, vec![cols], "output shape should be vector of cols");

    // Manual per-column dot
    for c in 0..cols {
        let mut acc = 0.0f64;
        for r in 0..rows {
            acc += xh[r + c * rows] * wh[r + c * rows];
        }
        let got = out.data[c];
        assert!(
            (got - acc).abs() < 1e-6,
            "dim0 col={c} got={got} exp={acc} rows={rows}"
        );
    }
}

#[test]
fn fused_mean_mul_dim0_matches_manual() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");
    let rows = 5usize;
    let cols = 4usize;

    let mut xh = vec![0.0f64; rows * cols];
    let mut wh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + 1.0;
            wh[r + c * rows] = (c as f64) + 1.0;
        }
    }
    let x = upload_matrix(&provider, rows, cols, &xh);
    let w = upload_matrix(&provider, rows, cols, &wh);

    let vid_x: ValueId = 0;
    let vid_w: ValueId = 1;
    let vid_mul: ValueId = 2;

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
        inputs: vec![vid_x, vid_w],
        output: Some(vid_mul),
    }];
    let mut constants: HashMap<usize, Value> = HashMap::new();
    constants.insert(0, Value::Num(1.0)); // dim=1 => reduce rows
    let plan = FusionGroupPlan {
        index: 0,
        group,
        operations,
        inputs: vec![vid_x, vid_w],
        stack_pattern: vec![],
        constants,
        const_values: HashMap::new(),
        output: None,
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(vid_mul),
        reduction_mode: Some(ReductionMode::Mean),
        pattern: None,
    };
    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x.clone()), Value::GpuTensor(w.clone())],
    };
    let result = execute_reduction(request, rows, cols, 0).expect("execute fused reduction");
    let out_handle = match result {
        Value::GpuTensor(h) => h,
        _ => panic!("expected GPU tensor"),
    };
    let out = provider.download(&out_handle).expect("download");
    assert_eq!(out.shape, vec![cols]);
    for c in 0..cols {
        let mut acc = 0.0f64;
        for r in 0..rows {
            acc += xh[r + c * rows] * wh[r + c * rows];
        }
        let mean = acc / (rows as f64);
        let got = out.data[c];
        assert!((got - mean).abs() < 1e-6, "dim0 mean col={c} got={got} exp={mean}");
    }
}

#[test]
fn fused_sum_mul_dim1_matches_manual() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");
    let rows = 5usize;
    let cols = 4usize;

    // X[r,c] = r+1, W[r,c] = (c+1)
    let mut xh = vec![0.0f64; rows * cols];
    let mut wh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + 1.0;
            wh[r + c * rows] = (c as f64) + 1.0;
        }
    }
    let x = upload_matrix(&provider, rows, cols, &xh);
    let w = upload_matrix(&provider, rows, cols, &wh);

    // Reduce (X .* W) over columns (MATLAB dim=2 => axis=1 in generator)
    let vid_x: ValueId = 0;
    let vid_w: ValueId = 1;
    let vid_mul: ValueId = 2;

    let group = FusionGroup {
        id: 0,
        kind: FusionKind::Reduction,
        nodes: vec![],
        shape: ShapeInfo::Tensor(vec![Some(rows)]),
        span: InstrSpan { start: 0, end: 0 },
        pattern: None,
    };

    let operations = vec![FusionOp::Primitive {
        op: runmat_accelerate::graph::PrimitiveOp::ElemMul,
        inputs: vec![vid_x, vid_w],
        output: Some(vid_mul),
    }];

    let mut constants: HashMap<usize, Value> = HashMap::new();
    constants.insert(0, Value::Num(2.0)); // MATLAB dim=2 → axis=1 (reduce cols)

    let plan = FusionGroupPlan {
        index: 0,
        group,
        operations,
        inputs: vec![vid_x, vid_w],
        stack_pattern: vec![],
        constants,
        const_values: HashMap::new(),
        output: None,
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(vid_mul),
        reduction_mode: Some(ReductionMode::Sum),
        pattern: None,
    };

    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x.clone()), Value::GpuTensor(w.clone())],
    };

    // reduce_len = cols, num_slices = rows
    let result = execute_reduction(request, cols, rows, 0).expect("execute fused reduction");
    let out_handle = match result {
        Value::GpuTensor(h) => h,
        _ => panic!("expected GPU tensor"),
    };
    let out = provider.download(&out_handle).expect("download");
    assert_eq!(out.shape, vec![rows], "output shape should be vector of rows");

    // Manual per-row sum across columns of X.*W
    for r in 0..rows {
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += xh[r + c * rows] * wh[r + c * rows];
        }
        let got = out.data[r];
        assert!(
            (got - acc).abs() < 1e-6,
            "dim1 row={r} got={got} exp={acc} cols={cols}"
        );
    }
}


