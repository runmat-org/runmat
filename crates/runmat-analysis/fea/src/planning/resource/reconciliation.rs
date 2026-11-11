use std::collections::BTreeMap;

use runmat_execution::resource::ResourceRequest;

use super::{
    contract::ResourcePlanningError,
    usage::{AnalysisResourceUsage, ResourceOverrun},
};

pub(crate) fn reconcile_resource_request(
    request: &ResourceRequest,
    actual: &AnalysisResourceUsage,
) -> Result<Vec<ResourceOverrun>, ResourcePlanningError> {
    let mut overruns = Vec::new();
    if actual.peak_cpu_millicores > request.cpu_millicores {
        overruns.push(ResourceOverrun::CpuPeak {
            planned: request.cpu_millicores,
            actual: actual.peak_cpu_millicores,
        });
    }
    // Planned CPU time is cores x wall time. A study with no authored wall-time ceiling plans the
    // wire maximum (2^53 - 1 ms), so the product needs u128; it cannot overflow there.
    let planned_core_millis =
        (u128::from(request.cpu_millicores) * u128::from(request.max_wall_millis)).div_ceil(1000);
    if u128::from(actual.scheduler.cpu_core_millis) > planned_core_millis {
        overruns.push(ResourceOverrun::CpuTime {
            // Below a u64 actual, so it fits.
            planned_core_millis: u64::try_from(planned_core_millis)
                .map_err(|_| ResourcePlanningError::Overflow("planned CPU time"))?,
            actual_core_millis: actual.scheduler.cpu_core_millis,
        });
    }
    if actual.peak_memory_bytes.0 > request.memory_bytes {
        overruns.push(ResourceOverrun::Memory {
            planned: request.memory_bytes,
            actual: actual.peak_memory_bytes.0,
        });
    }
    let planned_memory_time = u128::from(request.memory_bytes)
        .checked_mul(u128::from(request.max_wall_millis))
        .ok_or(ResourcePlanningError::Overflow("planned memory byte time"))?;
    if actual.scheduler.memory_byte_millis > planned_memory_time {
        overruns.push(ResourceOverrun::MemoryTime {
            planned_byte_millis: planned_memory_time,
            actual_byte_millis: actual.scheduler.memory_byte_millis,
        });
    }
    if actual.peak_scratch_bytes.0 > request.scratch_bytes {
        overruns.push(ResourceOverrun::Scratch {
            planned: request.scratch_bytes,
            actual: actual.peak_scratch_bytes.0,
        });
    }
    if actual.artifact_bytes.0 > request.max_artifact_bytes {
        overruns.push(ResourceOverrun::Artifact {
            planned: request.max_artifact_bytes,
            actual: actual.artifact_bytes.0,
        });
    }
    if actual.scheduler.egress_bytes > request.max_egress_bytes {
        overruns.push(ResourceOverrun::Egress {
            planned: request.max_egress_bytes,
            actual: actual.scheduler.egress_bytes,
        });
    }
    if actual.scheduler.relay_bytes > request.max_relay_bytes {
        overruns.push(ResourceOverrun::Relay {
            planned: request.max_relay_bytes,
            actual: actual.scheduler.relay_bytes,
        });
    }
    if actual.scheduler.wall_millis > request.max_wall_millis {
        overruns.push(ResourceOverrun::WallTime {
            planned: request.max_wall_millis,
            actual: actual.scheduler.wall_millis,
        });
    }
    let planned_retained = u128::from(request.max_artifact_bytes)
        .checked_mul(u128::from(request.max_wall_millis))
        .ok_or(ResourcePlanningError::Overflow(
            "planned retained byte time",
        ))?;
    if actual.scheduler.retained_byte_millis > planned_retained {
        overruns.push(ResourceOverrun::RetainedTime {
            planned_byte_millis: planned_retained,
            actual_byte_millis: actual.scheduler.retained_byte_millis,
        });
    }
    let mut actual_accelerators = BTreeMap::<&str, u64>::new();
    for usage in &actual.scheduler.accelerator_millis {
        let total = actual_accelerators.entry(usage.class.as_str()).or_default();
        *total = total
            .checked_add(usage.device_millis)
            .ok_or(ResourcePlanningError::Overflow("actual accelerator time"))?;
    }
    let mut planned_accelerators = BTreeMap::<&str, u64>::new();
    for accelerator in &request.accelerators {
        let millis = u64::from(accelerator.count)
            .checked_mul(request.max_wall_millis)
            .ok_or(ResourcePlanningError::Overflow("planned accelerator time"))?;
        let total = planned_accelerators
            .entry(accelerator.class.as_str())
            .or_default();
        *total = total
            .checked_add(millis)
            .ok_or(ResourcePlanningError::Overflow("planned accelerator time"))?;
    }
    for (class, actual_device_millis) in actual_accelerators {
        let planned_device_millis = planned_accelerators.get(class).copied().unwrap_or(0);
        if actual_device_millis > planned_device_millis {
            overruns.push(ResourceOverrun::AcceleratorTime {
                class: class.to_owned(),
                planned_device_millis,
                actual_device_millis,
            });
        }
    }
    Ok(overruns)
}
