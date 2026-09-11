use std::collections::BTreeMap;

use runmat_execution::resource::{AcceleratorRequest, ResourceInventory, ResourceRequest};
use runmat_execution::Digest;

use super::{
    build_resource_plan,
    contract::{
        AnalysisResourceUsage, FeaResourcePlan, FeaResourcePlanningInput, PartitioningPolicy,
        ResourceAdmission, ResourceOverrun, ResourcePlanningError, ResourceUsageAttempt,
        ResourceUsageEvidence,
    },
};

pub fn record_resource_usage(
    plan: &FeaResourcePlan,
    workload_digest: Digest,
    result_digest: Digest,
    attempts: Vec<ResourceUsageAttempt>,
    graph_wall_millis: u64,
) -> Result<ResourceUsageEvidence, ResourcePlanningError> {
    plan.validate_identity()?;
    if attempts
        .iter()
        .any(|attempt| !allocation_fits_budget(&attempt.resources.allocation, &plan.request))
    {
        return Err(ResourcePlanningError::Identity);
    }
    ResourceUsageEvidence::build(
        plan.identity,
        workload_digest,
        result_digest,
        plan.request.clone(),
        attempts,
        graph_wall_millis,
    )
}

pub fn validate_resource_usage_evidence(
    plan: &FeaResourcePlan,
    evidence: &ResourceUsageEvidence,
) -> Result<(), ResourcePlanningError> {
    plan.validate_identity()?;
    if evidence.plan != plan.identity
        || evidence.budget != plan.request
        || evidence
            .attempts
            .iter()
            .any(|attempt| !allocation_fits_budget(&attempt.resources.allocation, &plan.request))
        || evidence.overruns != reconcile_resource_usage(plan, &evidence.actual)?
    {
        return Err(ResourcePlanningError::Identity);
    }
    Ok(())
}

pub(crate) fn allocation_fits_budget(
    allocation: &ResourceRequest,
    budget: &ResourceRequest,
) -> bool {
    allocation.cpu_millicores <= budget.cpu_millicores
        && allocation.memory_bytes <= budget.memory_bytes
        && allocation.scratch_bytes <= budget.scratch_bytes
        && allocation.max_wall_millis <= budget.max_wall_millis
        && allocation.max_artifact_bytes <= budget.max_artifact_bytes
        && allocation.max_egress_bytes <= budget.max_egress_bytes
        && allocation.max_relay_bytes <= budget.max_relay_bytes
        && budget
            .required_capabilities
            .is_subset(&allocation.required_capabilities)
        && allocation.accelerators.iter().all(|requested| {
            budget.accelerators.iter().any(|available| {
                available.class == requested.class
                    && requested.count <= available.count
                    && requested.memory_bytes_each <= available.memory_bytes_each
            })
        })
}

pub fn allocation_matches_plan(allocation: &ResourceRequest, planned: &ResourceRequest) -> bool {
    allocation.cpu_millicores == planned.cpu_millicores
        && allocation.memory_bytes == planned.memory_bytes
        && allocation.scratch_bytes == planned.scratch_bytes
        && allocation.max_wall_millis == planned.max_wall_millis
        && allocation.max_artifact_bytes == planned.max_artifact_bytes
        && allocation.max_egress_bytes == planned.max_egress_bytes
        && allocation.max_relay_bytes == planned.max_relay_bytes
        && allocation.accelerators == planned.accelerators
        && planned
            .required_capabilities
            .is_subset(&allocation.required_capabilities)
}

pub fn admit_resource_plan(
    input: FeaResourcePlanningInput,
    inventory: &ResourceInventory,
) -> ResourceAdmission {
    let plan = match build_resource_plan(input) {
        Ok(plan) => plan,
        Err(reason) => return ResourceAdmission::Reject { reason },
    };
    if let Err(reason) = validate_non_memory_inventory(&plan, inventory) {
        return ResourceAdmission::Reject { reason };
    }
    let memory_capacity = inventory
        .memory_bytes
        .min(plan.input.accuracy_ceiling.maximum_resident_bytes);
    if plan.request.memory_bytes <= memory_capacity
        && plan.request.scratch_bytes <= inventory.scratch_bytes
    {
        return ResourceAdmission::Admit { plan };
    }
    let PartitioningPolicy::OperatorDomain {
        maximum_partitions, ..
    } = plan.input.partitioning
    else {
        return ResourceAdmission::Reject {
            reason: ResourcePlanningError::Inventory("memory or scratch"),
        };
    };
    let minimum_partition_count = match minimum_partition_count(
        &plan,
        memory_capacity,
        inventory.scratch_bytes,
        maximum_partitions,
    ) {
        Some(count) => count,
        None => {
            return ResourceAdmission::Reject {
                reason: ResourcePlanningError::Inventory(
                    "partition limit cannot satisfy memory or scratch",
                ),
            };
        }
    };
    let Some(per_partition_request) = partition_request(&plan, minimum_partition_count) else {
        return ResourceAdmission::Reject {
            reason: ResourcePlanningError::Overflow("per-partition resource request"),
        };
    };
    ResourceAdmission::RequiresPartitioning {
        plan,
        minimum_partition_count,
        per_partition_request,
    }
}

fn validate_non_memory_inventory(
    plan: &FeaResourcePlan,
    inventory: &ResourceInventory,
) -> Result<(), ResourcePlanningError> {
    if plan.request.cpu_millicores > inventory.cpu_millicores {
        return Err(ResourcePlanningError::Inventory("CPU"));
    }
    for capability in &plan.request.required_capabilities {
        if !inventory.capabilities.contains(capability) {
            return Err(ResourcePlanningError::Capability(capability.clone()));
        }
    }
    for request in &plan.request.accelerators {
        if !accelerator_available(request, &inventory.accelerators) {
            return Err(ResourcePlanningError::Accelerator {
                class: request.class.clone(),
                count: request.count,
                memory_bytes_each: request.memory_bytes_each,
            });
        }
    }
    Ok(())
}

fn accelerator_available(request: &AcceleratorRequest, inventory: &[AcceleratorRequest]) -> bool {
    let available = inventory
        .iter()
        .filter(|candidate| {
            candidate.class == request.class
                && candidate.memory_bytes_each >= request.memory_bytes_each
        })
        .try_fold(0u16, |total, candidate| total.checked_add(candidate.count));
    available.is_some_and(|count| count >= request.count)
}

fn minimum_partition_count(
    plan: &FeaResourcePlan,
    memory_bytes: u64,
    scratch_bytes: u64,
    maximum_partitions: u32,
) -> Option<u32> {
    (2..=maximum_partitions).find(|partitions| {
        per_partition(
            plan.breakdown.fixed_memory_bytes.0,
            plan.breakdown.partitionable_memory_bytes.0,
            *partitions,
        )
        .is_some_and(|required| required <= memory_bytes)
            && per_partition(
                plan.breakdown.fixed_scratch_bytes.0,
                plan.breakdown.partitionable_scratch_bytes.0,
                *partitions,
            )
            .is_some_and(|required| required <= scratch_bytes)
    })
}

fn partition_request(plan: &FeaResourcePlan, partitions: u32) -> Option<ResourceRequest> {
    let mut request = plan.request.clone();
    request.memory_bytes = per_partition(
        plan.breakdown.fixed_memory_bytes.0,
        plan.breakdown.partitionable_memory_bytes.0,
        partitions,
    )?;
    request.scratch_bytes = per_partition(
        plan.breakdown.fixed_scratch_bytes.0,
        plan.breakdown.partitionable_scratch_bytes.0,
        partitions,
    )?;
    Some(request)
}

fn per_partition(fixed: u64, partitionable: u64, partitions: u32) -> Option<u64> {
    let partitions = u64::from(partitions);
    let divided = partitionable
        .checked_add(partitions.checked_sub(1)?)?
        .checked_div(partitions)?;
    fixed.checked_add(divided)
}

pub fn reconcile_resource_usage(
    plan: &FeaResourcePlan,
    actual: &AnalysisResourceUsage,
) -> Result<Vec<ResourceOverrun>, ResourcePlanningError> {
    reconcile_resource_request(&plan.request, actual)
}

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
    let planned_core_millis = u64::from(request.cpu_millicores)
        .checked_mul(request.max_wall_millis)
        .map(|millicore_millis| millicore_millis.div_ceil(1000))
        .ok_or(ResourcePlanningError::Overflow("planned CPU time"))?;
    if actual.scheduler.cpu_core_millis > planned_core_millis {
        overruns.push(ResourceOverrun::CpuTime {
            planned_core_millis,
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
        let total = actual_accelerators.entry(&usage.class).or_default();
        *total = total
            .checked_add(usage.device_millis)
            .ok_or(ResourcePlanningError::Overflow("actual accelerator time"))?;
    }
    let mut planned_accelerators = BTreeMap::<&str, u64>::new();
    for accelerator in &request.accelerators {
        let millis = u64::from(accelerator.count)
            .checked_mul(request.max_wall_millis)
            .ok_or(ResourcePlanningError::Overflow("planned accelerator time"))?;
        let total = planned_accelerators.entry(&accelerator.class).or_default();
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
