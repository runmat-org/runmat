//! An immutable Value generation retained across an executor suspension.
//!
//! The physical Value owner is root-free so it can also live inside a traced
//! lexical activation without creating a self-rooting cycle. This external
//! read lease owns the separate GC roots, and drops those roots before the
//! final shared generation owner can retire.

use runmat_value::{
    CallableGenerationInput, CallableGenerationPrepareError, ExactArrayClaimPromotionError,
    SharedValueGeneration, Value,
};

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::runtime_error::semantic_error_with_source;
use crate::RuntimeError;

use super::{RootedValueAdmissionError, RootedValueHandles, RootedValueLease};
#[path = "retained_value/callable_rooted.rs"]
mod callable_rooted;

#[derive(Clone, Debug)]
pub struct RetainedValueReadLease {
    // Field order is intentional: unregister before the final Value backing
    // may retire and run a Foreign release callback.
    roots: RootedValueHandles,
    generation: SharedValueGeneration,
}

impl RetainedValueReadLease {
    /// Move one already-owned Value into a shared generation. The quote funds
    /// this header and its root lease only; it does not retroactively certify
    /// arbitrary preexisting Value payload backing as finite-admitted.
    pub fn prepare_owned(value: Value, context: &RuntimeContext) -> Result<Self, RuntimeError> {
        let roots = RootedValueLease::for_value_in_context(&value, context)
            .map_err(|source| root_error(context, source))?;
        runmat_value::validate_no_transient_sequence(&value).map_err(|source| {
            semantic_error_with_source(
                "RunMat:Workspace:RetainedValue",
                "transient retained Value",
                source,
            )
        })?;
        let raw = u64::try_from(SharedValueGeneration::requested_bytes())
            .map_err(|_| host_error(context, HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(raw, 1)
            .map_err(|source| host_error(context, source))?;
        let plan =
            HostAllocationPlan::new(0, retained).map_err(|source| host_error(context, source))?;
        let generation = match HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|source| host_error(context, source))?
        {
            Some(receipt) => {
                let permit = receipt
                    .output_permit(context)
                    .map_err(|source| host_error(context, source))?;
                let generation = SharedValueGeneration::prepare(value, &permit)
                    .map_err(|source| host_error(context, HostAllocationError::Numeric(source)))?;
                receipt
                    .checkpoint(context)
                    .map_err(|source| host_error(context, source))?;
                generation
            }
            None => SharedValueGeneration::unbounded(value),
        };
        Ok(Self {
            roots: roots.shared_handles(),
            generation,
        })
    }

    /// Root an already-owned immutable generation without manufacturing a
    /// second Value header or copying any child payload. A finite caller
    /// supplies its exact operation authority for the new external root lease.
    pub(crate) fn retain_shared_generation(
        generation: SharedValueGeneration,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        let roots = match context {
            Some(context) => RootedValueLease::for_value_in_context(generation.value(), context)
                .map_err(|source| root_error(context, source))?
                .shared_handles(),
            None => RootedValueHandles::for_traced(generation.value()).map_err(|source| {
                semantic_error_with_source(
                    "RunMat:Workspace:RetainedValue",
                    "retained Value root admission failed",
                    source,
                )
            })?,
        };
        Ok(Self { roots, generation })
    }

    pub fn value(&self) -> &Value {
        self.generation.value()
    }

    pub(crate) fn shared_generation(&self) -> SharedValueGeneration {
        self.generation.clone()
    }

    pub(crate) fn shared_roots(&self) -> RootedValueHandles {
        self.roots.clone()
    }

    pub fn shares_generation_with(&self, other: &Self) -> bool {
        self.generation.ptr_eq(&other.generation)
    }
}

fn callable_prepare_error(
    context: &RuntimeContext,
    source: CallableGenerationPrepareError,
) -> RuntimeError {
    match source {
        CallableGenerationPrepareError::Allocation(source)
        | CallableGenerationPrepareError::CapturePromotion(
            ExactArrayClaimPromotionError::Allocation(source),
        ) => host_error(context, HostAllocationError::Numeric(source)),
        CallableGenerationPrepareError::CapturePromotion(
            source @ ExactArrayClaimPromotionError::UnsharedBacking,
        ) => semantic_error_with_source(
            "RunMat:Callable:RepeatableSelection",
            "repeatable callable capture backing is not shared",
            source,
        ),
        CallableGenerationPrepareError::NotCallable => crate::runtime_error::semantic_error(
            "RunMat:Callable:RepeatableSelection",
            "repeatable target is not a callable Value",
        ),
        CallableGenerationPrepareError::AlreadyConsumedOrReserved => {
            crate::runtime_error::semantic_error(
                "RunMat:Callable:RepeatableSelection",
                "repeatable callable preparation state was already consumed",
            )
        }
    }
}

fn host_error(context: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(context),
        "RunMat:Workspace:RetainedValue",
        "retained Value admission failed",
        source,
    )
}

fn root_error(context: &RuntimeContext, source: RootedValueAdmissionError) -> RuntimeError {
    match source {
        RootedValueAdmissionError::Host(source) => host_error(context, source),
        RootedValueAdmissionError::Gc(source) => semantic_error_with_source(
            "RunMat:Workspace:RetainedValue",
            "retained Value root admission failed",
            source,
        ),
    }
}
