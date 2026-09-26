use super::*;
use crate::region::{RegionOutputSource, SiteIdentity};
use runmat_runtime::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimePlacementService,
    RuntimeServicePorts,
};
use runmat_runtime::{execution::RuntimeExecutionService, RuntimeError};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    rc::Rc,
};

#[derive(Default)]
pub(crate) struct Probe {
    pub requests: RefCell<Vec<PlacementPlanRequest>>,
    pub feedback: RefCell<Vec<PlacementFeedback>>,
    pub vectorized: Cell<bool>,
    pub memory: Cell<Option<u64>>,
    pub epoch: Cell<u64>,
}
impl RuntimeParallelService for Probe {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: self.memory.get(),
            epoch: self.epoch.get(),
            ..Default::default()
        }
    }
}
impl RuntimePlacementService for Probe {
    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
        self.requests.borrow_mut().push(request.clone());
        let node = &request.graph.nodes[0];
        let vector = &node.candidates[1];
        let admit = self.vectorized.get()
            && request
                .resources
                .memory_available_bytes
                .is_none_or(|bytes| {
                    bytes >= vector.resources.retained_bytes + vector.resources.scratch_bytes
                });
        let selected = &node.candidates[usize::from(admit)].descriptor;
        Ok(runmat_execution::PlacementDecision {
            signature: request.signature.clone(),
            selections: vec![runmat_execution::SelectedExecutionCandidate {
                node: node.identity.clone(),
                candidate: selected.identity.clone(),
                kind: selected.kind,
            }],
            predicted_total_ns: selected.cost.checked_total_ns().unwrap(),
            from_cache: false,
            used_local_fallback: false,
            explored_states: 2,
            pruned_states: 0,
        })
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }
    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
}
pub(crate) fn runtime(probe: &Rc<Probe>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(probe.clone())
            .with_parallel(probe.clone()),
    )
}
pub(crate) fn plan() -> OptimizedRegionPlan {
    let function = runmat_types::ProgramFunctionId(0);
    let region = runmat_types::RegionId {
        function,
        ordinal: 0,
    };
    OptimizedRegionPlan {
        region,
        entry: SiteIdentity {
            point: runmat_types::ProgramPointId {
                function,
                block: 0,
                position: 0,
            },
            phase: runmat_native_codegen::NativeSitePhase::Rvalue,
            ordinal: 0,
        },
        exit: runmat_types::ProgramPointId {
            function,
            block: 0,
            position: 1,
        },
        inputs: vec![runmat_types::RegionValueId { function, local: 0 }],
        outputs: vec![crate::region::plan::RegionOutput {
            value: None,
            ssa: runmat_native_codegen::NativeValueId(0),
            source: RegionOutputSource::Computed(0),
        }],
        program: runmat_runtime::numeric_region::NumericRegionProgram {
            nodes: vec![runmat_runtime::numeric_region::NumericRegionNode::Input(0)],
            outputs: vec![0],
        },
        skipped_sites: BTreeSet::new(),
        arithmetic_operations: 1,
    }
}
pub(crate) fn choose(
    runtime: &RuntimeContext,
    plan: &OptimizedRegionPlan,
    input: &Value,
) -> RegionPlacement {
    let inputs = [input];
    choose_region(
        runtime,
        plan,
        &inputs,
        runmat_runtime::numeric_region::workload(&inputs).unwrap(),
    )
}
