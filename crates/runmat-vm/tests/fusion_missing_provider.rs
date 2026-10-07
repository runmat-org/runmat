//! A separate integration-test process keeps the device registry empty. Calling
//! `clear_provider` in a shared provider test is not an equivalent fixture:
//! registered device owners intentionally survive that operation.

#[path = "support/mod.rs"]
mod test_helpers;

use runmat_accelerate::placement::{self, PlacementEventKind, PlacementVariant};
use runmat_accelerate::FusionKind;
use runmat_value::Value;
use runmat_vm::Instr;

#[test]
fn automatic_host_fusion_without_provider_resumes_ordinary_vm() {
    runmat_gc::gc_test_context(|| {
        assert!(runmat_accelerate_api::provider().is_none());
        let source = "x = [-5.5, -1.25, 5.5]; y = mod(x, 2) + rem(x, 2);";
        let bytecode = test_helpers::compile_source(source).expect("compile source");
        let groups = bytecode.runtime_fusion_groups();
        let graph = bytecode
            .runtime_accel_graph_for_fusion(&groups)
            .expect("runtime fusion graph");
        let detected = graph.detect_fusion_groups();
        assert!(detected
            .iter()
            .any(|group| matches!(group.kind, FusionKind::ElementwiseChain)));
        let plans = runmat_accelerate::FusionPlan::from_graph(&graph, &detected);
        assert!(plans
            .groups
            .iter()
            .any(|plan| plan.group.kind.is_elementwise() && plan.kernel.supported));

        placement::reset();
        let vars = test_helpers::interpret(&bytecode).expect("host fallback must complete");
        let output_slot = bytecode
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instr::StoreVar(slot) => Some(*slot),
                _ => None,
            })
            .next_back()
            .expect("output store");
        let Value::Tensor(output) = &vars[output_slot] else {
            panic!("ordinary VM must publish a host tensor");
        };
        assert_eq!(output.materialize_f64(), vec![-1.0, -0.5, 3.0]);
        assert!(placement::report().traces.iter().any(|trace| {
            trace.events.iter().any(|event| {
                event.kind == PlacementEventKind::Selected
                    && event.variant == Some(PlacementVariant::SharedRuntime)
                    && event.reason.as_deref() == Some("provider_unavailable")
            })
        }));
    });
}
