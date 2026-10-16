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
    physical_facts(provider, Some(runtime))
}

/// Physical resource/capability facts do not confer session or observation
/// authority. Context-free callers use the same provider snapshot and the
/// existing unspecified-host-resource defaults, without creating a runtime.
pub(super) fn physical_facts(
    provider: &dyn AccelProvider,
    runtime: Option<&RuntimeContext>,
) -> (PlacementResourceSnapshot, Digest, u64, bool) {
    // A provider with an immutable shared inventory does not need to clone
    // its nested capability vectors for each placement request. Legacy
    // providers retain the detached snapshot contract. Stream the exact
    // serialized bytes in either case instead of allocating a JSON mirror.
    let shared_capability = provider.capability_snapshot_ref();
    let owned_capability = shared_capability
        .is_none()
        .then(|| provider.capability_snapshot());
    let capability = shared_capability
        .or(owned_capability.as_ref())
        .expect("one capability snapshot route is selected");
    let provider_digest =
        Digest::sha256_serialized(capability).expect("provider capability snapshot must serialize");
    let provider_resources = provider.placement_resources();
    let host_resources = runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .map(|service| service.placement_resources())
        .unwrap_or_default();
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: host_resources.cpu_millicores_available,
        memory_available_bytes: host_resources.memory_available_bytes,
        cancellation_requested: runtime.is_some_and(RuntimeContext::is_cancelled),
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
