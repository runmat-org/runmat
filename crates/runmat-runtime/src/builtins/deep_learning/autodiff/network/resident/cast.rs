//! Restore an executed resident adjoint to its traced parent's floating class.
//! The provider's general floating-conversion contract owns feasibility and
//! physical resource accounting; this module only binds that contract to tape.

use super::*;
use crate::builtins::common::gpu_helpers::{
    reject_gpu_output, unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, ProviderElementType, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderPrecision, ProviderPreparationRoute, ProviderRepresentation, ProviderResidency,
    ProviderResourceEstimate, ProviderStorage, ProviderWorkload,
};

/// An operation-owned bound, not a certificate for the whole reverse route.
/// The provider's full backward buffers and total available VRAM remain outside
/// this value. Every cast result stays live until the reverse result publishes.
#[derive(Clone, Copy, Default)]
pub(super) struct CastBudget {
    transient_bytes: u64,
    output_bytes: u64,
    dispatches: u64,
}

#[derive(Clone, Copy)]
pub(super) struct CastRequirement {
    pub(super) source: ProviderPrecision,
    pub(super) target: ProviderPrecision,
    transient_bytes: u64,
    output_bytes: u64,
    dispatches: u32,
}

impl CastRequirement {
    fn from_estimate(
        source: ProviderPrecision,
        target: ProviderPrecision,
        estimate: ProviderResourceEstimate,
    ) -> Option<Self> {
        Some(Self {
            source,
            target,
            transient_bytes: estimate.transient_bytes?,
            output_bytes: estimate.output_bytes?,
            dispatches: estimate.dispatches?,
        })
    }

    fn within(self, admitted: Self) -> bool {
        self.source == admitted.source
            && self.target == admitted.target
            && self.transient_bytes <= admitted.transient_bytes
            && self.output_bytes <= admitted.output_bytes
            && self.dispatches <= admitted.dispatches
    }
}

impl CastBudget {
    pub(super) fn include(
        &mut self,
        required: CastRequirement,
        function: &'static str,
    ) -> BuiltinResult<()> {
        let transient_bytes = self
            .transient_bytes
            .checked_add(required.transient_bytes)
            .ok_or_else(|| deep_learning_error(function, "resident cast resource sum overflows"))?;
        let output_bytes = self
            .output_bytes
            .checked_add(required.output_bytes)
            .ok_or_else(|| deep_learning_error(function, "resident cast output sum overflows"))?;
        let dispatches = self
            .dispatches
            .checked_add(u64::from(required.dispatches))
            .ok_or_else(|| deep_learning_error(function, "resident cast dispatch sum overflows"))?;
        *self = Self {
            transient_bytes,
            output_bytes,
            dispatches,
        };
        Ok(())
    }

    fn consume(
        &mut self,
        actual: CastRequirement,
        admitted: CastRequirement,
        function: &'static str,
    ) -> BuiltinResult<()> {
        let changed = || {
            model::gpu::mixed::capability_changed(
                function,
                model::gpu::GpuCapabilityReason::ProviderOperation,
            )
        };
        if !actual.within(admitted) {
            return Err(changed());
        }
        let transient_bytes = self
            .transient_bytes
            .checked_sub(actual.transient_bytes)
            .ok_or_else(changed)?;
        let output_bytes = self
            .output_bytes
            .checked_sub(actual.output_bytes)
            .ok_or_else(changed)?;
        let dispatches = self
            .dispatches
            .checked_sub(u64::from(actual.dispatches))
            .ok_or_else(changed)?;
        *self = Self {
            transient_bytes,
            output_bytes,
            dispatches,
        };
        Ok(())
    }
}

fn element_type(precision: ProviderPrecision) -> ProviderElementType {
    match precision {
        ProviderPrecision::F32 => ProviderElementType::F32,
        ProviderPrecision::F64 => ProviderElementType::F64,
    }
}

fn query(
    source: ProviderPrecision,
    target: ProviderPrecision,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<ProviderFeasibilityQuery> {
    let shape: Vec<u64> = shape
        .iter()
        .map(|&extent| u64::try_from(extent))
        .collect::<Result<_, _>>()
        .map_err(|_| {
            deep_learning_error(function, "resident adjoint shape exceeds provider range")
        })?;
    let representation = |dtype| ProviderRepresentation {
        element_type: element_type(dtype),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.clone(),
        residency: ProviderResidency::Device,
    };
    let identity = match target {
        ProviderPrecision::F32 => runmat_builtins::SINGLE_CATALOG_ENTRY.identity.name,
        ProviderPrecision::F64 => runmat_builtins::DOUBLE_CATALOG_ENTRY.identity.name,
    };
    Ok(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(identity),
        family: ProviderOperationFamily::Elementwise,
        inputs: vec![representation(source)],
        outputs: vec![representation(target)],
        workload: ProviderWorkload {
            preparation_route: Some(ProviderPreparationRoute::FloatingCast { target }),
            ..ProviderWorkload::default()
        },
    })
}

pub(super) fn admit(
    provider: &dyn AccelProvider,
    source: ProviderPrecision,
    target: ProviderPrecision,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<Option<CastRequirement>> {
    if source == target {
        return Ok(None);
    }
    let query = query(source, target, shape, function)?;
    match provider.query_feasibility(&query) {
        ProviderFeasibility::Supported { estimate } => {
            CastRequirement::from_estimate(source, target, estimate)
                .map(Some)
                .ok_or_else(|| {
                    unsupported_error(
                        function,
                        format!("{function}: resident gradient cast resource bound is unavailable"),
                    )
                })
        }
        ProviderFeasibility::Rejected { rejection } => Err(unsupported_error(
            function,
            format!(
                "{function}: resident gradient cast is unavailable ({:?})",
                rejection.code
            ),
        )),
    }
}

pub(super) async fn execute(
    provider: &'static dyn AccelProvider,
    gradient: GpuTensorHandle,
    target: ProviderPrecision,
    borrowed: &[GpuTensorHandle],
    admitted: CastRequirement,
    remaining: &mut CastBudget,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let source = resident_precision(&gradient)?;
    if source == target || source != admitted.source || target != admitted.target {
        return Err(model::gpu::mixed::capability_changed(
            function,
            model::gpu::GpuCapabilityReason::ProviderOperation,
        ));
    }
    let current =
        match provider.query_feasibility(&query(source, target, gradient.shape(), function)?) {
            ProviderFeasibility::Supported { estimate } => {
                CastRequirement::from_estimate(source, target, estimate)
            }
            ProviderFeasibility::Rejected { .. } => None,
        }
        .ok_or_else(|| {
            model::gpu::mixed::capability_changed(
                function,
                model::gpu::GpuCapabilityReason::ProviderOperation,
            )
        })?;
    remaining.consume(current, admitted, function)?;
    let casted = match target {
        ProviderPrecision::F32 => provider.unary_single(&gradient).await,
        ProviderPrecision::F64 => provider.unary_double(&gradient).await,
    }
    .map_err(|error| {
        model::gpu::provider_failure(
            function,
            "resident gradient precision restoration",
            error.into(),
        )
    })?;
    let valid = unary_gpu_output_matches_shape(
        &casted,
        &gradient,
        provider,
        gradient.shape(),
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(target),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) && casted.physical().requested_bytes() <= current.output_bytes
        && casted.view().layout() == gradient.view().layout()
        && !borrowed
            .iter()
            .any(|handle| casted.shares_allocation_with(handle));
    if valid {
        Ok(casted)
    } else {
        Err(reject_gpu_output(
            casted,
            deep_learning_error(
                function,
                "resident gradient cast returned an invalid class, owner, shape, or alias",
            ),
        ))
    }
}

#[cfg(test)]
#[path = "cast/tests.rs"]
mod tests;
