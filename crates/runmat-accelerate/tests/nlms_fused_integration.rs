#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider_impl::WgpuProviderOptions;
use runmat_accelerate::fusion::{detect_fusion_groups, FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
use runmat_accelerate::fusion_exec::{
    execute_elementwise, execute_reduction, FusionExecutionRequest,
};
use runmat_accelerate::graph::{
    AccelGraph, AccelNode, AccelNodeLabel, AccelOpCategory, AccelGraphTag, InstrSpan, PrimitiveOp,
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
fn nlms_two_fused_reductions_integration() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        WgpuProviderOptions::default(),
    )
    .expect("wgpu");
    let rows = 8usize;
    let cols = 16usize;
    let p = rows;
    let c = cols;

    // Create simple host data
    let mut xh = vec![0.0f64; rows * cols];
    let mut wh = vec![0.0f64; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            xh[row + col * rows] = (row as f64 + 1.0) * (col as f64 + 1.0) * 0.001;
            wh[row + col * rows] = (col as f64 + 1.0) * 0.01;
        }
    }
    let x = upload(&provider, rows, cols, &xh);
    let w = upload(&provider, rows, cols, &wh);

    // 1) d = sum(x .* x, 1)  (columns)
    {
        let v_x: ValueId = 0;
        let v_mul: ValueId = 1;
        let v_dim: ValueId = 2;
        let v_sum: ValueId = 3;

        let mut values: Vec<ValueInfo> = Vec::new();
        values.push(ValueInfo {
            id: v_x,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_mul,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_dim,
            origin: ValueOrigin::Constant,
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::Num(1.0)),
        });
        values.push(ValueInfo {
            id: v_sum,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(c)]),
            constant: None,
        });

        let mul_node = AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemMul),
            category: AccelOpCategory::Elementwise,
            inputs: vec![v_x, v_x],
            outputs: vec![v_mul],
            span: InstrSpan { start: 0, end: 0 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let sum_node = AccelNode {
            id: 1,
            label: AccelNodeLabel::Builtin {
                name: "sum".into(),
            },
            category: AccelOpCategory::Reduction,
            inputs: vec![v_mul, v_dim],
            outputs: vec![v_sum],
            span: InstrSpan { start: 1, end: 1 },
            tags: vec![AccelGraphTag::Reduction],
        };
        values[v_mul as usize].origin = ValueOrigin::NodeOutput { node: 0, output: 0 };
        values[v_sum as usize].origin = ValueOrigin::NodeOutput { node: 1, output: 0 };
        let mut var_bindings = HashMap::new();
        var_bindings.insert(
            v_x,
            VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        );
        let graph = AccelGraph {
            nodes: vec![mul_node, sum_node],
            values,
            var_bindings,
        };
        let groups = detect_fusion_groups(&graph);
        let red_group = groups.iter().find(|g| g.kind.is_reduction()).unwrap().clone();
        let plan = FusionGroupPlan {
            index: red_group.id,
            group: red_group.clone(),
            operations: vec![
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemMul,
                    inputs: vec![v_x, v_x],
                    output: Some(v_mul),
                },
                FusionOp::Builtin {
                    name: "sum".to_string(),
                    inputs: vec![v_mul, v_dim],
                    output: Some(v_sum),
                },
            ],
            inputs: vec![v_x],
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
            output: Some(v_sum),
            kernel: FusionKernelSpec {
                kind: FusionKind::Reduction,
                supported: true,
            },
            reduction_data: Some(v_mul),
            reduction_dim: Some(v_dim),
            reduction_mode: None,
            pattern: None,
        };
        let req = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(x.clone())],
        };
        let _d = execute_reduction(req, rows, cols, 0).expect("fused sum(x.*x,1)");
    }

    // 2) y = sum(x .* w, 1)  (two-input fused reduction)
    let y_handle = {
        let v_x: ValueId = 0;
        let v_w: ValueId = 1;
        let v_mul: ValueId = 2;
        let v_dim: ValueId = 3;
        let v_sum: ValueId = 4;

        let mut values: Vec<ValueInfo> = Vec::new();
        values.push(ValueInfo {
            id: v_x,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_w,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 1,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_mul,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(p), Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_dim,
            origin: ValueOrigin::Constant,
            ty: Type::Num,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::Num(1.0)),
        });
        values.push(ValueInfo {
            id: v_sum,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(c)]),
            constant: None,
        });

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
            label: AccelNodeLabel::Builtin {
                name: "sum".into(),
            },
            category: AccelOpCategory::Reduction,
            inputs: vec![v_mul, v_dim],
            outputs: vec![v_sum],
            span: InstrSpan { start: 1, end: 1 },
            tags: vec![AccelGraphTag::Reduction],
        };
        let mut values = values;
        values[v_mul as usize].origin = ValueOrigin::NodeOutput { node: 0, output: 0 };
        values[v_sum as usize].origin = ValueOrigin::NodeOutput { node: 1, output: 0 };
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
        let red_group = groups.iter().find(|g| g.kind.is_reduction()).unwrap().clone();
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
            output: Some(v_sum),
            kernel: FusionKernelSpec {
                kind: FusionKind::Reduction,
                supported: true,
            },
            reduction_data: Some(v_mul),
            reduction_dim: Some(v_dim),
            reduction_mode: None,
            pattern: None,
        };
        assert_eq!(plan.inputs.len(), 2, "two-input fused plan required");
        let req = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(x.clone()), Value::GpuTensor(w.clone())],
        };
        let y = execute_reduction(req, rows, cols, 0).expect("fused sum(x.*w,1)");
        match y {
            Value::GpuTensor(h) => h,
            _ => panic!("expected GPU tensor for y"),
        }
    };

    // 3) e = d - y  (we don't need actual d here; just reuse y for a shape-correct op)
    // To keep chain residency idea alive, do an elementwise e = y - y (zeros) with elementwise fusion
    let e_handle = {
        let v_a: ValueId = 0;
        let v_b: ValueId = 1;
        let v_out: ValueId = 2;
        // Build a minimal elementwise-chain plan directly (single op)
        let plan = FusionGroupPlan {
            index: 0,
            group: FusionGroup {
                id: 0,
                kind: FusionKind::ElementwiseChain,
                nodes: vec![],
                shape: ShapeInfo::Tensor(vec![Some(cols)]),
                span: InstrSpan { start: 0, end: 0 },
                pattern: None,
            },
            operations: vec![FusionOp::Primitive {
                op: PrimitiveOp::Sub,
                inputs: vec![v_a, v_b],
                output: Some(v_out),
            }],
            inputs: vec![v_a, v_b],
            stack_pattern: vec![],
            constants: HashMap::new(),
            const_values: HashMap::new(),
            output: Some(v_out),
            kernel: FusionKernelSpec {
                kind: FusionKind::ElementwiseChain,
                supported: true,
            },
            reduction_data: None,
            reduction_dim: None,
            reduction_mode: None,
            pattern: None,
        };
        let req = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(y_handle.clone()), Value::GpuTensor(y_handle.clone())],
        };
        let e_val = execute_elementwise(req).expect("fused elementwise sub");
        match e_val {
            Value::GpuTensor(h) => h,
            _ => panic!("expected GPU tensor for e"),
        }
    };

    // 4) mse = mean((e).^2, 'all')  (flattened fused reduction)
    {
        let v_e: ValueId = 0;
        let v_sq: ValueId = 1;
        let v_all: ValueId = 2;
        let v_mean: ValueId = 3;
        let mut values: Vec<ValueInfo> = Vec::new();
        values.push(ValueInfo {
            id: v_e,
            origin: ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            },
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_sq,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(c)]),
            constant: None,
        });
        values.push(ValueInfo {
            id: v_all,
            origin: ValueOrigin::Constant,
            ty: Type::String,
            shape: ShapeInfo::Scalar,
            constant: Some(Value::String("all".to_string())),
        });
        values.push(ValueInfo {
            id: v_mean,
            origin: ValueOrigin::Unknown,
            ty: Type::tensor(),
            shape: ShapeInfo::Tensor(vec![Some(1)]),
            constant: None,
        });
        let pow_node = AccelNode {
            id: 0,
            label: AccelNodeLabel::Primitive(PrimitiveOp::ElemPow),
            category: AccelOpCategory::Elementwise,
            inputs: vec![v_e, v_e], // Not exactly pow(e,2), but good enough for plan shape; squared done via elementwise multiply
            outputs: vec![v_sq],
            span: InstrSpan { start: 0, end: 0 },
            tags: vec![AccelGraphTag::Elementwise],
        };
        let mean_node = AccelNode {
            id: 1,
            label: AccelNodeLabel::Builtin {
                name: "mean".into(),
            },
            category: AccelOpCategory::Reduction,
            inputs: vec![v_sq, v_all],
            outputs: vec![v_mean],
            span: InstrSpan { start: 1, end: 1 },
            tags: vec![AccelGraphTag::Reduction],
        };
        let mut var_bindings = HashMap::new();
        var_bindings.insert(
            v_e,
            VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        );
        let values = values;
        let graph = AccelGraph {
            nodes: vec![pow_node, mean_node],
            values,
            var_bindings,
        };
        let groups = detect_fusion_groups(&graph);
        let red_group = groups.iter().find(|g| g.kind.is_reduction()).unwrap().clone();
        let plan = FusionGroupPlan {
            index: red_group.id,
            group: red_group.clone(),
            operations: vec![
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    inputs: vec![v_e, v_e], // used as a stand-in for squaring
                    output: Some(v_sq),
                },
                FusionOp::Builtin {
                    name: "mean".to_string(),
                    inputs: vec![v_sq, v_all],
                    output: Some(v_mean),
                },
            ],
            inputs: vec![v_e],
            stack_pattern: vec![],
            constants: HashMap::new(),
            const_values: {
                let mut cv = HashMap::new();
                cv.insert(v_all, Value::String("all".to_string()));
                cv
            },
            output: Some(v_mean),
            kernel: FusionKernelSpec {
                kind: FusionKind::Reduction,
                supported: true,
            },
            reduction_data: Some(v_sq),
            reduction_dim: None,
            reduction_mode: None,
            pattern: None,
        };
        // Reduce all elements: reduce_len=c, num_slices=1
        let req = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::GpuTensor(e_handle.clone())],
        };
        let mse = execute_reduction(req, cols, 1, 0).expect("fused mean(all)");
        match mse {
            Value::GpuTensor(h) => {
                assert_eq!(h.shape, vec![1], "mean(all) should return scalar shape [1]");
            }
            _ => panic!("expected GPU tensor for mse"),
        }
    }
}


