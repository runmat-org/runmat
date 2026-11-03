#![cfg(feature = "wgpu")]
#[path = "support/direct_reduction_runtime.rs"]
mod invocation;
#[path = "support/selected_reduction.rs"]
mod selected_reduction;

use runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::fusion_exec::FusionExecutionRequest;
use runmat_accelerate::graph::{
    InstrSpan, PrimitiveOp, ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarKind,
};
use runmat_accelerate::ReductionAxes;
use runmat_accelerate_api::{AccelProvider, HostTensorView};
use runmat_builtins::Type;
use runmat_value::Value;
use std::collections::HashMap;

#[test]
fn fused_single_pass_sum_mul_no_alias() {
    let runtime = invocation::runtime();
    let _invocation = runtime.enter();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");

    // Choose dimensions to force single-pass fused reduction (reduce_len <= threshold)
    let rows = 256usize; // reduce_len
    let cols = 64usize; // num_slices
    let len = rows * cols;
    let mut xh = vec![0.0f64; len];
    let mut wh = vec![0.0f64; len];
    for c in 0..cols {
        for r in 0..rows {
            let i = r + c * rows;
            xh[i] = (r as f64 + 1.0) * 0.001;
            wh[i] = (c as f64 + 1.0) * 0.01;
        }
    }
    let x = provider
        .upload(&HostTensorView {
            data: &xh,
            shape: &[rows, cols],
        })
        .expect("upload");
    let w = provider
        .upload(&HostTensorView {
            data: &wh,
            shape: &[rows, cols],
        })
        .expect("upload");

    // Build a minimal fused reduction plan for y = sum(x .* w, 1)
    let v_x: ValueId = 0;
    let v_w: ValueId = 1;
    let v_mul: ValueId = 2;
    let v_dim: ValueId = 3;
    let v_sum: ValueId = 4;

    let _values = vec![
        ValueInfo {
            id: v_x,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(rows), Some(cols)]),
            constant: None,
        },
        ValueInfo {
            id: v_w,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 1,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(rows), Some(cols)]),
            constant: None,
        },
        ValueInfo {
            id: v_mul,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(rows), Some(cols)]),
            constant: None,
        },
        ValueInfo {
            id: v_dim,
            origin: ValueOrigin::Constant,
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::Num(1.0)),
        },
        ValueInfo {
            id: v_sum,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(1), Some(cols)]),
            constant: None,
        },
    ];

    let group = FusionGroup {
        id: 0,
        kind: FusionKind::Reduction,
        nodes: vec![],
        shape: ShapeInfo::Tensor(vec![Some(1), Some(cols)]),
        span: InstrSpan { start: 0, end: 1 },
        pattern: None,
        stack_layout: None,
    };

    let plan = FusionGroupPlan {
        index: 0,
        group,
        operations: vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemMul,
                inputs: vec![v_x, v_w],
                output: Some(v_mul),
            },
            FusionOp::Builtin {
                origin: Default::default(),
                name: "sum".into(),
                inputs: vec![v_mul, v_dim],
                output: Some(v_sum),
            },
        ],
        inputs: vec![v_x, v_w],
        stack_pattern: vec![],
        constants: {
            let mut m = HashMap::new();
            m.insert(0, Value::Num(1.0));
            m
        },
        const_values: {
            let mut cv = HashMap::new();
            cv.insert(v_dim, Value::Num(1.0));
            cv
        },
        materialized_stores: Vec::new(),
        output: Some(v_sum),
        kernel: FusionKernelSpec {
            kind: FusionKind::Reduction,
            supported: true,
        },
        reduction_data: Some(v_mul),
        reduction_dim: Some(v_dim),
        reduction_flavor: None,
        reduction_axes: Some(ReductionAxes::Explicit(vec![1])),
        pattern: None,
    };

    let req = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(x), Value::GpuTensor(w)],
        placement: None,
        runtime: Some(runtime.clone()),
    };
    // reduce_len = rows, num_slices = cols; single-pass expected
    let Some(out) =
        futures::executor::block_on(selected_reduction::execute(req, rows, cols, &[1, cols], 0))
            .expect("canonical single-pass reduction admission")
    else {
        return;
    };
    match out {
        Value::GpuTensor(h) => {
            assert_eq!(
                h.shape(),
                vec![1, cols],
                "canonical sum retains its reduced axis"
            );
            let host =
                futures::executor::block_on(provider.download(&h)).expect("actual GPU download");
            for column in 0..cols {
                let expected: f64 = (0..rows)
                    .map(|row| xh[row + column * rows] * wh[row + column * rows])
                    .sum();
                assert!(
                    (host.data[column] - expected).abs() < 1e-6,
                    "column {column}: {} != {expected}",
                    host.data[column]
                );
            }
        }
        _ => panic!("expected GPU tensor"),
    }
}
