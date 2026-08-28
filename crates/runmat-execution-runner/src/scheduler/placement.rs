use runmat_execution::identity::WorkerId;
use runmat_execution::resource::ResourceRequest;

use crate::pool::WorkerRecord;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementCandidate {
    pub worker_id: WorkerId,
    pub remaining_cpu_millicores: u32,
    pub remaining_memory_bytes: u64,
}

pub fn choose_worker<'a>(
    workers: impl IntoIterator<Item = &'a WorkerRecord>,
    request: &ResourceRequest,
) -> Option<PlacementCandidate> {
    let mut candidates = workers
        .into_iter()
        .filter(|worker| worker.accepts_work())
        .filter(|worker| super::fits(&worker.spec.resources, &worker.allocated, request))
        .map(|worker| PlacementCandidate {
            worker_id: worker.spec.id,
            remaining_cpu_millicores: worker
                .spec
                .resources
                .cpu_millicores
                .saturating_sub(worker.allocated.cpu_millicores)
                .saturating_sub(request.cpu_millicores),
            remaining_memory_bytes: worker
                .spec
                .resources
                .memory_bytes
                .saturating_sub(worker.allocated.memory_bytes)
                .saturating_sub(request.memory_bytes),
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| {
        (
            candidate.remaining_cpu_millicores,
            candidate.remaining_memory_bytes,
            candidate.worker_id,
        )
    });
    candidates.into_iter().next()
}
