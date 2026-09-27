use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, EstimateConfidence, EstimateSource,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ExecutionCostComponents,
    ExecutionCostEstimate, PlacementFeedback, PlacementGraph, PlacementGraphCandidate,
    PlacementGraphLimits, PlacementGraphNode, PlacementPlanRequest, PlacementResourceSnapshot,
    PlacementRevision, PlacementSignature,
};
use runmat_runtime::context::RuntimeContext;
use runmat_value::Value;

use super::plan::OptimizedRegionPlan;

pub(crate) enum RegionPlacement {
    Unavailable,
    Generic(RegionObservation),
    Vectorized(RegionObservation),
}

/// One selected execution, completed explicitly or failed when its scope exits.
pub(crate) struct RegionObservation {
    runtime: RuntimeContext,
    signature: PlacementSignature,
    candidate: String,
    started: runmat_time::Instant,
    reported: bool,
}

impl RegionObservation {
    pub fn complete(mut self) {
        self.report(true);
    }

    fn report(&mut self, succeeded: bool) {
        if self.reported {
            return;
        }
        self.reported = true;
        let Some(service) = self.runtime.service_ports().placement() else {
            return;
        };
        let _ = service.observe(PlacementFeedback {
            signature: self.signature.clone(),
            candidate: self.candidate.clone(),
            total_elapsed_ns: runmat_time::duration_ns_saturating(self.started.elapsed()).max(1),
            succeeded,
        });
    }
}

impl Drop for RegionObservation {
    fn drop(&mut self) {
        self.report(false);
    }
}

pub(crate) fn choose_region(
    runtime: &RuntimeContext,
    plan: &OptimizedRegionPlan,
    inputs: &[&Value],
    workload: runmat_runtime::numeric_region::NumericRegionWorkload,
) -> RegionPlacement {
    let Some(service) = runtime.service_ports().placement() else {
        return RegionPlacement::Unavailable;
    };
    let signature = signature(runtime, plan, inputs);
    let elements = u64::try_from(workload.elements).unwrap_or(u64::MAX);
    let operations = u64::from(plan.arithmetic_operations);
    let generic_ns = elements
        .saturating_mul(operations)
        .saturating_mul(20)
        .saturating_add(10_000);
    let vectorized_ns = elements
        .saturating_mul(operations)
        .saturating_mul(3)
        .saturating_add(1_000);
    let generic = candidate(plan, false, generic_ns);
    let vectorized = candidate(plan, true, vectorized_ns);
    let resources = runtime
        .service_ports()
        .parallel()
        .map(|service| service.placement_resources())
        .unwrap_or_default();
    let output_bytes = workload
        .output_bytes_per_value
        .saturating_mul(u64::try_from(plan.program.outputs.len()).unwrap_or(u64::MAX));
    let request = PlacementPlanRequest {
        signature: signature.clone(),
        graph: PlacementGraph {
            nodes: vec![PlacementGraphNode {
                identity: format!("native.region.{}", plan.region.ordinal),
                candidates: vec![
                    PlacementGraphCandidate {
                        descriptor: generic.clone(),
                        resources: CandidateResourceDemand {
                            cpu_millicores: 1_000,
                            ..CandidateResourceDemand::default()
                        },
                        transactional_results: true,
                    },
                    PlacementGraphCandidate {
                        descriptor: vectorized.clone(),
                        resources: CandidateResourceDemand {
                            cpu_millicores: 1_000,
                            retained_bytes: output_bytes,
                            scratch_bytes: output_bytes,
                            queue_slots: 0,
                        },
                        transactional_results: true,
                    },
                ],
            }],
            edges: Vec::new(),
        },
        limits: PlacementGraphLimits::default(),
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: resources.cpu_millicores_available,
            memory_available_bytes: resources.memory_available_bytes,
            cancellation_requested: runtime
                .cancellation()
                .load(std::sync::atomic::Ordering::Relaxed),
            providers: Vec::new(),
            epoch: resources.epoch,
        },
        deterministic: false,
        require_transactional_results: true,
    };
    let decision = match service.plan(request) {
        Ok(decision) => decision,
        Err(_) => return RegionPlacement::Unavailable,
    };
    if decision.signature != signature || decision.selections.len() != 1 {
        return RegionPlacement::Unavailable;
    }
    let selected = &decision.selections[0];
    if selected.node != format!("native.region.{}", plan.region.ordinal) {
        return RegionPlacement::Unavailable;
    }
    let observation = |candidate| RegionObservation {
        runtime: runtime.clone(),
        signature: signature.clone(),
        candidate,
        started: runmat_time::Instant::now(),
        reported: false,
    };
    if selected.kind == generic.kind && selected.candidate == generic.identity {
        RegionPlacement::Generic(observation(generic.identity))
    } else if selected.kind == vectorized.kind && selected.candidate == vectorized.identity {
        RegionPlacement::Vectorized(observation(vectorized.identity))
    } else {
        RegionPlacement::Unavailable
    }
}

fn candidate(
    plan: &OptimizedRegionPlan,
    vectorized: bool,
    execution_ns: u64,
) -> ExecutionCandidateDescriptor {
    ExecutionCandidateDescriptor {
        identity: candidate_identity(plan, vectorized),
        region: Some(plan.region),
        kind: if vectorized {
            ExecutionCandidateKind::VectorizedNativeCpu
        } else {
            ExecutionCandidateKind::SpecializedNativeCpu
        },
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        cost: ExecutionCostEstimate {
            components: ExecutionCostComponents {
                execution_ns,
                ..ExecutionCostComponents::default()
            },
            scratch_bytes: 0,
            confidence: EstimateConfidence::Prior,
            source: EstimateSource::Compiler,
        },
        output_residency: CandidateOutputResidency::Host,
        guards: Vec::new(),
    }
}

fn candidate_identity(plan: &OptimizedRegionPlan, vectorized: bool) -> String {
    format!(
        "cpu.{}.f{}.r{}",
        if vectorized {
            "vectorized"
        } else {
            "specialized"
        },
        plan.region.function.0,
        plan.region.ordinal
    )
}

fn signature(
    runtime: &RuntimeContext,
    plan: &OptimizedRegionPlan,
    inputs: &[&Value],
) -> PlacementSignature {
    let facts = inputs
        .iter()
        .map(|value| runmat_runtime::value_fact::value_fact(value))
        .collect::<Vec<_>>();
    PlacementSignature {
        region: Some(plan.region),
        operation: "native.vectorized_numeric_region".into(),
        runtime_facts: Digest::sha256(
            serde_json::to_vec(&facts).expect("portable runtime facts must serialize"),
        ),
        revision: PlacementRevision {
            program: runtime.program_revision().cloned(),
            catalog: Digest::sha256(b"runmat-placement-catalog-v1"),
            compiler: Digest::sha256(b"runmat-native-vector-region-v1"),
            provider: Digest::sha256(b"host-only"),
            policy: Digest::sha256(b"shared-placement-v1"),
        },
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod test_support;
