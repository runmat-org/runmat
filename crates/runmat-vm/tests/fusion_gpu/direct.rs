//! Exact selected Runtime bindings for direct graph-derived fusion fixtures.

use runmat_accelerate::fusion::{FusionGroupPlan, FusionOp};
use runmat_accelerate::graph::FusionCallOrigin;
use runmat_runtime::builtin::RuntimeBuiltinBinding;
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use std::rc::Rc;

struct Selected(Vec<RuntimeBuiltinBinding>);

impl RuntimeBuiltinService for Selected {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.0
            .iter()
            .filter(|binding| binding.identity.builtin.name == name)
            .copied()
            .collect()
    }
}

pub(super) fn selected_runtime_for_plan(plan: &FusionGroupPlan) -> RuntimeContext {
    let mut selected = Vec::<RuntimeBuiltinBinding>::new();
    for operation in &plan.operations {
        let FusionOp::Builtin { name, origin, .. } = operation else {
            continue;
        };
        let FusionCallOrigin::Named { binding: expected } = origin else {
            panic!("direct named-call fixture has no selected binding: {name}");
        };
        let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        let [binding] = bindings.as_slice() else {
            panic!("direct fixture requires one compiled binding for {name}");
        };
        assert_eq!(binding.identity, *expected, "compiled binding for {name}");
        if !selected
            .iter()
            .any(|existing| existing.identity == binding.identity)
        {
            selected.push(*binding);
        }
    }
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Selected(selected))))
}
