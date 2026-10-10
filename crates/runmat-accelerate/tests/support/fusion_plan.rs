use runmat_accelerate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp,
};
use runmat_accelerate::graph::{InstrSpan, ShapeInfo};
use std::collections::HashMap;

pub fn initialize_registry() {
    // These fixtures invoke fusion directly, bypassing the runtime dispatcher
    // and VM entrypoints that normally initialize the browser registry.
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();

    // Verify real compiled registration, without selecting a provider route or
    // bypassing invocation-specific binding and operand admission below.
    for name in ["single", "double"] {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name)
            .expect("fixture conversion has a canonical catalog entry");
        let [binding] = entry.bindings else {
            panic!("fixture conversion {name} must have one canonical binding");
        };
        let identity = entry.binding_identity(binding);
        let declaration =
            runmat_runtime::builtins::common::spec::canonical_fusion_declaration(identity)
                .expect("fixture requires the compiled canonical fusion declaration");
        assert_eq!(declaration.binding, identity);
        assert_eq!(declaration.spec.name, name);
    }
}

pub fn plan(operations: Vec<FusionOp>, output: u32) -> FusionGroupPlan {
    initialize_registry();
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
