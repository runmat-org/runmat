use runmat_accelerate_api::AccelProvider;
use runmat_execution::{Digest, PlacementResourceSnapshot};
use runmat_runtime::context::RuntimeContext;

/// One snapshot authority for both admitted provider and rejected CPU-only
/// placement decisions. The revision still records the actual provider whose
/// capabilities produced the feasibility decision.
pub(super) fn session_facts(
    provider: &dyn AccelProvider,
    runtime: &RuntimeContext,
) -> (PlacementResourceSnapshot, Digest, u64, bool) {
    let capability = provider.capability_snapshot();
    let provider_digest = Digest::sha256(
        serde_json::to_vec(&capability).expect("provider capability snapshot must serialize"),
    );
    let provider_resources = provider.placement_resources();
    let host_resources = runtime
        .service_ports()
        .parallel()
        .map(|service| service.placement_resources())
        .unwrap_or_default();
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: host_resources.cpu_millicores_available,
        memory_available_bytes: host_resources.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: vec![provider_resources.clone()],
        epoch: provider_resources.epoch ^ host_resources.epoch.rotate_left(32),
    };
    (
        resources,
        provider_digest,
        host_resources.epoch,
        capability.concurrency.transactional_results,
    )
}
