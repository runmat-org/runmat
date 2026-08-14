use runmat_accelerate_api::{
    AccelProvider, ProviderCostQuery, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRejectionCode,
    ProviderRepresentation, ProviderWorkload,
};
use runmat_execution::{
    CandidateOutputResidency, CandidatePreparationState, ExecutionCandidateDescriptor,
    ExecutionCandidateKind, ExecutionCostEstimate,
};
use runmat_value::Value;

use super::{select_candidate, summarize_values, tensor_representation, PlacementPolicy};

pub(crate) struct LocalPlacementRequest<'a> {
    pub(crate) operation: &'a str,
    pub(crate) family: ProviderOperationFamily,
    pub(crate) provider_kind: ExecutionCandidateKind,
    pub(crate) inputs: &'a [&'a Value],
    pub(crate) outputs: Vec<ProviderRepresentation>,
    pub(crate) workload: ProviderWorkload,
    pub(crate) preparation: CandidatePreparationState,
    pub(crate) cpu: ExecutionCandidateDescriptor,
    /// Provider execution prior without transfer components. It is used only
    /// when the provider has no trustworthy estimate of its own.
    pub(crate) provider_fallback: ExecutionCostEstimate,
    pub(crate) required_download_bytes: u64,
    pub(crate) downstream_materialization: bool,
    /// Direct provider APIs preserve caller intent after feasibility succeeds;
    /// correlated runtime requests compare against the CPU candidate.
    pub(crate) compare_profitability: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum LocalPlacementOutcome {
    Selected {
        selected: Box<ExecutionCandidateDescriptor>,
        cpu_cost: ExecutionCostEstimate,
        provider_cost: ExecutionCostEstimate,
    },
    ProviderRejected {
        code: ProviderRejectionCode,
        cpu_cost: ExecutionCostEstimate,
    },
}

/// Normalize provider feasibility and complete transfer costs before comparing
/// profitability. This is the one local authority shared by legacy automatic
/// offload and correlated fusion execution.
pub(crate) fn plan_local(
    provider: &dyn AccelProvider,
    request: LocalPlacementRequest<'_>,
    policy: PlacementPolicy,
) -> LocalPlacementOutcome {
    debug_assert!(request.provider_kind.is_provider());
    let residency = summarize_values(request.inputs);
    let provider_resident_bytes = residency
        .provider_bytes
        .values()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let mut cpu = request.cpu;
    if provider_resident_bytes > 0 {
        cpu.cost.components.synchronization_ns =
            cpu.cost.components.synchronization_ns.saturating_add(5_000);
        cpu.cost.components.download_ns = cpu
            .cost
            .components
            .download_ns
            .saturating_add(transfer_prior_ns(provider_resident_bytes));
    }
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(request.operation),
        family: request.family,
        inputs: request
            .inputs
            .iter()
            .filter_map(|value| tensor_representation(value, provider.precision()))
            .collect(),
        outputs: request.outputs,
        workload: request.workload,
    };
    if let ProviderFeasibility::Rejected { rejection } = provider.query_feasibility(&query) {
        return LocalPlacementOutcome::ProviderRejected {
            code: rejection.code,
            cpu_cost: cpu.cost,
        };
    }

    let upload_bytes = residency.required_upload_bytes(provider.device_id());
    let provider_cost = provider
        .estimate_cost(&ProviderCostQuery {
            operation: query,
            preparation: request.preparation,
            required_upload_bytes: upload_bytes,
            required_download_bytes: request.required_download_bytes,
            downstream_materialization: request.downstream_materialization,
        })
        .map(|estimate| estimate.cost)
        .unwrap_or_else(|| {
            let mut cost = request.provider_fallback;
            cost.components.upload_ns = transfer_prior_ns(upload_bytes);
            cost.components.download_ns = transfer_prior_ns(request.required_download_bytes);
            cost.components.synchronization_ns = if request.required_download_bytes == 0 {
                0
            } else {
                cost.components.synchronization_ns.max(5_000)
            };
            cost
        });
    let provider_candidate = ExecutionCandidateDescriptor {
        identity: format!("provider.{}.{}", provider.device_id(), request.operation),
        region: cpu.region,
        kind: request.provider_kind,
        preparation: request.preparation,
        cost: provider_cost,
        output_residency: CandidateOutputResidency::Provider {
            device_id: provider.device_id(),
        },
        guards: cpu.guards.clone(),
    };
    let cpu_cost = cpu.cost;
    let selected = if request.compare_profitability {
        select_candidate([cpu, provider_candidate], policy)
            .expect("CPU and provider placement candidates must contain one valid descriptor")
    } else {
        provider_candidate
    };
    LocalPlacementOutcome::Selected {
        selected: Box::new(selected),
        cpu_cost,
        provider_cost,
    }
}

pub(crate) fn transfer_prior_ns(bytes: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        20_000_u64.saturating_add(bytes.saturating_add(7) / 8)
    }
}
