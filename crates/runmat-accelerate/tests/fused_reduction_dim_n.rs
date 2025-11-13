#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider_impl::WgpuProviderOptions;
use runmat_accelerate::fusion::{
    detect_fusion_groups, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp, ReductionMode,
};
use runmat_accelerate::fusion_exec::{execute_reduction, FusionExecutionRequest};
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, PrimitiveOp,
    ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{Type, Value};
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

#[test]
fn fused_sum_mul_dim_n_equals_manual_for_n1_and_n2() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");

    let rows = 6usize;
    let cols = 5usize;
    let mut xh = vec![0.0f64; rows * cols];
    let mut wh = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            xh[r + c * rows] = (r as f64) + 1.0;
            wh[r + c * rows] = (c as f64) + 2.0;
        }
    }
    let x = upload(&provider, rows, cols, &xh);
    let w = upload(&provider, rows, cols, &wh);

    for dim_val in [1.0f64, 2.0f64] {
        // Build graph: y = sum(x .* w, n)
        let v_x: ValueId = 0;
        let v_w: ValueId = 1;
        let v_mul: ValueId = 2;
        let v_dim: ValueId = 3;
        let v_y: ValueId = 4;

        let values = vec![
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
                constant: Some(Value::Num(dim_val)),
            },
            ValueInfo {
                id: v_y,
                origin: ValueOrigin::Unknown,
                ty: Type::tensor(),
                shape: if dim_val == 1.0 {
                    ShapeInfo::Tensor(vec![Some(cols)])
                } else {
                    ShapeInfo::Tensor(vec![Some(rows)])
                },
                constant: None,
            },
        ];

        let mul_node = AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![v_x, v_w],
            outputs: vec![v_mul],
            span: InstrSpan { start: 0, end: 0 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let sum_node = AccelNode {
            id: 1,
            label: AccelNodeLabel::Builtin { name: "sum".into() },
            category: AccelOpCategory::Reduction,
            inputs: vec![v_mul, v_dim],
            outputs: vec![v_y],
            span: InstrSpan { start: 1, end: 1 },
            tags: vec![AccelGraphTag::Reduction],
        };

        let mut var_bindings = HashMap::new();
        var_bindings.insert(
            v_x,
            VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        );
        var_bindings.insert(
            v_w,
            VarBinding {
                kind: VarKind::Global,
                index: 1,
            },
        );

        let graph = AccelGraph {
            nodes: vec![mul_node, sum_node],
            values,
            var_bindings,
        };
        let groups = detect_fusion_groups(&graph);
        let red_group = groups
            .iter()
            .find(|g| g.kind.is_reduction())
            .unwrap()
            .clone();

        let plan = FusionGroupPlan {
            index: red_group.id,
            group: red_group.clone(),
            operations: vec![
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemMul,
                    inputs: vec![v_x, v_w],
                    output: Some(v_mul),
                },
                FusionOp::Builtin {
                    name: "sum".to_string(),
                    inputs: vec![v_mul, v_dim],
                    output: Some(v_y),
                },
            ],
            inputs: vec![v_x, v_w],
            stack_pattern: vec![],
            constants: HashMap::new(),
            const_values: {
                let mut cv = HashMap::new();
                cv.insert(v_dim, Value::Num(dim_val));
                cv
            },
            output: Some(v_y),
            kernel: FusionKernelSpec {
                kind: FusionKind::Reduction,
                supported: true,
            },
            reduction_data: Some(v_mul),
            reduction_dim: Some(v_dim),
            reduction_mode: Some(ReductionMode::Sum),
            pattern: None,
        };

        let req = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(x.clone()), Value::GpuTensor(w.clone())],
        };
        let (reduce_len, num_slices) = if dim_val == 1.0 {
            (rows, cols)
        } else {
            (cols, rows)
        };
        let result =
            execute_reduction(req, reduce_len, num_slices, 0).expect("execute fused reduction");
        let out_handle = match result {
            Value::GpuTensor(h) => h,
            _ => panic!("expected GPU tensor"),
        };
        let out = provider.download(&out_handle).expect("download");

        if dim_val == 1.0 {
            assert_eq!(out.shape, vec![cols]);
            for c in 0..cols {
                let mut acc = 0.0f64;
                for r in 0..rows {
                    acc += xh[r + c * rows] * wh[r + c * rows];
                }
                let got = out.data[c];
                assert!(
                    (got - acc).abs() < 1e-6,
                    "dim=1 col={c} got={got} exp={acc}"
                );
            }
        } else {
            assert_eq!(out.shape, vec![rows]);
            for r in 0..rows {
                let mut acc = 0.0f64;
                for c in 0..cols {
                    acc += xh[r + c * rows] * wh[r + c * rows];
                }
                let got = out.data[r];
                assert!(
                    (got - acc).abs() < 1e-6,
                    "dim=2 row={r} got={got} exp={acc}"
                );
            }
        }
    }
}
