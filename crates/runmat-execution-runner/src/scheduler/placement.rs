use runmat_execution::identity::WorkerId;
use runmat_execution::resource::AcceleratorDevice;
use runmat_execution::task::TaskRequest;

use crate::pool::WorkerRecord;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementCandidate {
    pub worker_id: WorkerId,
    pub remaining_cpu_millicores: u32,
    pub remaining_memory_bytes: u64,
    pub accelerator_devices: Vec<AcceleratorDevice>,
}

pub fn choose_worker<'a>(
    workers: impl IntoIterator<Item = &'a WorkerRecord>,
    task: &TaskRequest,
) -> Option<PlacementCandidate> {
    let request = &task.resources;
    let mut candidates = workers
        .into_iter()
        .filter(|worker| worker.accepts_work())
        .filter(|worker| task.host.is_satisfied_by(&worker.spec.host).is_ok())
        .filter_map(|worker| {
            let accelerator_devices =
                super::select_devices(&worker.spec.resources, &worker.allocated, request)?;
            if !super::fits(&worker.spec.resources, &worker.allocated, request) {
                return None;
            }
            Some(PlacementCandidate {
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
                accelerator_devices,
            })
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
