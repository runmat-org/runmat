//! Effect-free provider geometry and feedback identity for a source loop.
//! Physical tile selection belongs to the provider; placement policy belongs
//! to the installed service. Neither step may draw or upload a value.

use super::resources::planning_metadata_plan;
use super::source_eligibility::{StochasticPreparedSource, StochasticTypedPattern};
use crate::builtins::common::random::gpu_like::{host_random_mode, ExactHostRandomMode};
use crate::builtins::common::random::RngAlgorithm;
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionSelection, ProviderStochasticEvolutionWorkRequest,
    StochasticEvolutionArithmetic, PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, ProviderElementType, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation,
    ProviderResidency, ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};
use runmat_types::NumericClass;
use runmat_value::Value;

#[derive(serde::Serialize)]
enum SourceResidency {
    Host,
    AutomaticProvider(u32),
    ExplicitProvider(u32),
    UnspecifiedProvider(u32),
}

/// Seed, stream position, drift and scale values do not define different
/// timing classes. Their numeric classes and the selected arithmetic do.
pub(super) fn profile(
    context: &RuntimeContext,
    prepared: &StochasticPreparedSource,
    algorithm: RngAlgorithm,
) -> Digest {
    let pattern = prepared.pattern();
    let residency = match prepared.state() {
        Value::GpuTensor(handle) => match runmat_accelerate_api::handle_provenance(handle) {
            Some(GpuHandleProvenance::Explicit) => {
                SourceResidency::ExplicitProvider(handle.device_id())
            }
            Some(GpuHandleProvenance::Automatic) => {
                SourceResidency::AutomaticProvider(handle.device_id())
            }
            None => SourceResidency::UnspecifiedProvider(handle.device_id()),
        },
        _ => SourceResidency::Host,
    };
    Digest::sha256_serialized(&(
        pattern.shape(),
        pattern.state_class(),
        pattern.state_storage(),
        pattern.noise_class(),
        pattern.noise_storage(),
        pattern.scale_noise(),
        pattern.drift_scaled(),
        pattern.first_state_factor(),
        pattern.continuing_state_factor(),
        pattern.steps(),
        algorithm,
        residency,
        host_random_mode(Some(context)) == ExactHostRandomMode::Managed,
    ))
    .expect("typed stochastic timing facts serialize")
}

fn invalid(message: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:StochasticSource:InvalidPlan")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn element(class: NumericClass) -> BuiltinResult<ProviderElementType> {
    match class {
        NumericClass::Single => Ok(ProviderElementType::F32),
        NumericClass::Double => Ok(ProviderElementType::F64),
        _ => Err(invalid(
            "stochastic provider planning requires real floating classes",
        )),
    }
}

fn representation(
    pattern: &StochasticTypedPattern,
    class: NumericClass,
) -> BuiltinResult<ProviderRepresentation> {
    // Match the rank-sized backing admitted by planning_metadata_plan.
    // Fallible iterator collection may grow a short Vec to a larger minimum
    // capacity, beyond that quote.
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(pattern.shape().len())
        .map_err(|_| invalid("stochastic provider shape allocation failed"))?;
    for extent in pattern.shape() {
        shape.push(
            u64::try_from(*extent)
                .map_err(|_| invalid("stochastic provider shape exceeds its representation"))?,
        );
    }
    Ok(ProviderRepresentation {
        element_type: element(class)?,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    })
}

/// The request's metadata reservation follows its owned vectors and query.
/// Existing payloads remain borrowed; this is not permission to upload them.
pub(super) struct StochasticProviderPlan {
    pub(super) provider: &'static dyn AccelProvider,
    pub(super) selected: ProviderStochasticEvolutionSelection,
    pub(super) query: ProviderFeasibilityQuery,
    pub(super) metadata: Option<HostAllocationReceipt>,
}

pub(super) fn prepare_provider(
    context: &RuntimeContext,
    prepared: &StochasticPreparedSource,
) -> BuiltinResult<Option<StochasticProviderPlan>> {
    let provider = match prepared.state() {
        Value::GpuTensor(handle) => Some(
            crate::builtins::common::gpu_helpers::require_resident_compute_owner(handle, || {
                invalid("stochastic state has no exact live provider owner")
            })?,
        ),
        _ => context.command_device_provider(),
    };
    let Some(provider) = provider else {
        return Ok(None);
    };
    let pattern = prepared.pattern();
    let planning_scratch =
        provider.stochastic_evolution_planning_host_workspace(pattern.shape().len());
    let scratch = match planning_scratch {
        ResourceByteBound::Known(bytes) => bytes,
        ResourceByteBound::Unavailable(_) => {
            // A third-party planner with no allocation certificate cannot
            // escape finite admission even on an otherwise legal device.
            if host_random_mode(Some(context)) == ExactHostRandomMode::Managed {
                return Ok(None);
            }
            0
        }
    };
    let base = planning_metadata_plan(pattern.shape().len())?;
    let plan = HostAllocationPlan::new(
        base.scratch_peak_bytes()
            .checked_add(scratch)
            .ok_or_else(|| invalid("stochastic planning workspace size overflow"))?,
        base.retained_output_bytes(),
    )
    .map_err(|_| invalid("stochastic planning metadata size overflow"))?;
    let metadata = match HostAllocationReceipt::admit(Some(context), plan) {
        Ok(receipt) => receipt,
        Err(source) if source.is_capacity_denial() => return Ok(None),
        Err(source) => {
            return Err(build_runtime_error(
                "stochastic provider planning metadata was not admitted",
            )
            .with_identifier(format!(
                "{}:{}",
                context.error_namespace(),
                source.runtime_identifier_suffix()
            ))
            .with_source(source)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build());
        }
    };
    StochasticPreparedSource::checkpoint_receipt(metadata.as_ref(), context)?;
    let first = StochasticEvolutionArithmetic::new(
        pattern.state_class(),
        pattern.noise_class(),
        pattern.drift(),
        pattern.scale(),
    )
    .ok_or_else(|| invalid("stochastic initial arithmetic has no provider representation"))?;
    let continuing = StochasticEvolutionArithmetic::new(
        pattern.result_class(),
        pattern.noise_class(),
        pattern.drift(),
        pattern.scale(),
    )
    .ok_or_else(|| invalid("stochastic continuing arithmetic has no provider representation"))?;
    let work = ProviderStochasticEvolutionWorkRequest {
        state: representation(pattern, pattern.state_class())?,
        output: representation(pattern, pattern.result_class())?,
        first_arithmetic: first,
        continuing_arithmetic: continuing,
        steps: pattern.steps(),
    };
    let selected = provider
        .plan_stochastic_evolution(&work)
        .map_err(|source| {
            let mut error = crate::runtime_error::with_provider_execution_source(
                invalid("stochastic provider planning failed"),
                source,
            );
            error.gpu_gather_retry = GpuGatherRetry::Never;
            error
        })?;
    StochasticPreparedSource::checkpoint_receipt(metadata.as_ref(), context)?;
    let Some(selected) = selected else {
        return Ok(None);
    };
    if !selected.validate_for(&work) {
        return Err(invalid(
            "stochastic provider selected a different semantic request",
        ));
    }
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION),
        family: ProviderOperationFamily::Library,
        inputs: vec![selected.plan.state.clone()],
        outputs: vec![selected.plan.output.clone()],
        workload: ProviderWorkload {
            stochastic_evolution: selected.plan.workload(),
            ..ProviderWorkload::default()
        },
    };
    StochasticPreparedSource::checkpoint_receipt(metadata.as_ref(), context)?;
    Ok(Some(StochasticProviderPlan {
        provider,
        selected,
        query,
        metadata,
    }))
}
