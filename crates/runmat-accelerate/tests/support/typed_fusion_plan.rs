use runmat_accelerate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::graph::{InstrSpan, ShapeInfo};
use std::collections::HashMap;

pub fn cast(name: &str, input: u32, output: u32) -> FusionOp {
    FusionOp::Builtin {
        origin: Default::default(),
        name: name.into(),
        inputs: vec![input],
        output: Some(output),
    }
}

pub fn plan(operations: Vec<FusionOp>, output: u32) -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Tensor(vec![Some(1), Some(2)]),
            span: InstrSpan {
                start: 0,
                end: operations.len(),
            },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0],
        stack_pattern: vec![0],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(output),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}
