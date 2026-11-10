use runmat_execution::resource::{AcceleratorRequest, ResourceInventory, ResourceRequest};
use runmat_execution::Digest;

use super::{
    build_resource_plan,
    contract::{
        FeaResourcePlan, FeaResourcePlanningInput, PartitioningPolicy, ResourceAdmission,
        ResourcePlanningError,
    },
    reconciliation::reconcile_resource_request,
    usage::{AnalysisResourceUsage, ResourceOverrun, ResourceUsageAttempt, ResourceUsageEvidence},
};

pub fn record_resource_usage(
    plan: &FeaResourcePlan,
    workload_digest: Digest,
    result_digest: Digest,
    attempts: Vec<ResourceUsageAttempt>,
    graph_wall_millis: u64,
) -> Result<ResourceUsageEvidence, ResourcePlanningError> {
    record_resource_graph_usage(
        plan,
        Vec::new(),
        workload_digest,
        result_digest,
        attempts,
        graph_wall_millis,
    )
}

pub fn record_resource_graph_usage(
    primary_plan: &FeaResourcePlan,
    mut additional_plans: Vec<FeaResourcePlan>,
    workload_digest: Digest,
    result_digest: Digest,
    attempts: Vec<ResourceUsageAttempt>,
    graph_wall_millis: u64,
) -> Result<ResourceUsageEvidence, ResourcePlanningError> {
    primary_plan.validate_identity()?;
    additional_plans.push(primary_plan.clone());
    ResourceUsageEvidence::build(
        primary_plan.identity,
        additional_plans,
        workload_digest,
        result_digest,
        attempts,
        graph_wall_millis,
    )
}

pub fn validate_resource_usage_evidence(
    primary_plan: &FeaResourcePlan,
    evidence: &ResourceUsageEvidence,
) -> Result<(), ResourcePlanningError> {
    primary_plan.validate_identity()?;
    evidence.validate_identity()?;
    if evidence.primary_plan != primary_plan.identity
        || !evidence.plans.iter().any(|plan| plan == primary_plan)
    {
        return Err(ResourcePlanningError::Identity);
    }
    Ok(())
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
        .min(plan.input.accuracy_ceiling.resident_byte_budget());
    if plan.request.memory_bytes <= memory_capacity
        && plan.request.scratch_bytes <= inventory.scratch_bytes
    {
        return ResourceAdmission::Admit { plan };
    }
    let PartitioningPolicy::OperatorDomain {
        maximum_partitions, ..
    } = *plan.input.partitioning()
    else {
        return ResourceAdmission::Reject {
            reason: ResourcePlanningError::MemoryBudget {
                required_memory_bytes: plan.request.memory_bytes,
                available_memory_bytes: memory_capacity,
                required_scratch_bytes: plan.request.scratch_bytes,
                available_scratch_bytes: inventory.scratch_bytes,
            },
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
                class: request.class.to_string(),
                count: request.count,
                memory_bytes_each: request.minimum_allocation_bytes,
            });
        }
    }
    Ok(())
}

/// Distinct physical allocation domains are the leasable unit: two provider
/// views of one device cannot satisfy a two-device request.
fn accelerator_available(
    request: &AcceleratorRequest,
    inventory: &[runmat_execution::resource::AcceleratorDevice],
) -> bool {
    let domains = inventory
        .iter()
        .filter(|device| request.matches(device))
        .map(|device| &device.allocation_domain)
        .collect::<std::collections::BTreeSet<_>>();
    domains.len() >= usize::from(request.count)
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
