//! Current-operation ownership of reshape's parsed dimensions and new shape.
//! The source tensor's payload claim is never treated as reshape authority.

use super::{
    finish_dimensions_in_place, parse_size_scalar, reshape_dimension_error, reshape_error,
    reshape_size_error, DimToken,
};
use crate::builtins::common::tensor;
use crate::context::host_allocation::{
    AdmittedCollectionAllocator, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use allocator_api2::vec::Vec as AdmittedVec;
use runmat_value::{SharedShape, Value};
use std::alloc::Layout;

pub(super) struct ReshapeDimensions {
    dimensions: AdmittedVec<usize, AdmittedCollectionAllocator>,
    receipt: HostAllocationReceipt,
    auto_index: Option<usize>,
    duplicate_auto: bool,
}

fn admission_error(context: &RuntimeContext, cause: HostAllocationError) -> crate::RuntimeError {
    build_runtime_error(format!("reshape: shape admission: {cause}"))
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

fn rank_hint(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<usize> {
    if args.len() != 1 {
        return Ok(args.len());
    }
    let rank = match &args[0] {
        Value::Tensor(tensor) => {
            if tensor.is_empty() {
                return Err(reshape_size_error(
                    "reshape: size vector must contain at least one element",
                ));
            }
            let values = tensor::borrowed_host_shape_dimensions(&args[0]).ok_or_else(|| {
                reshape_size_error("reshape: size vector must be a row or column vector")
            })?;
            for value in values {
                value.map_err(reshape_dimension_error)?;
            }
            tensor.len()
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Err(reshape_size_error(
                    "reshape: size vector must contain at least one element",
                ));
            }
            let values = tensor::borrowed_host_shape_dimensions(&args[0]).ok_or_else(|| {
                reshape_size_error("reshape: size vector must be a row or column vector")
            })?;
            for value in values {
                value.map_err(reshape_dimension_error)?;
            }
            logical.data.len()
        }
        Value::GpuTensor(handle) => handle
            .view()
            .checked_element_count()
            .ok()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| reshape_size_error("reshape: size vector extent overflow"))?,
        _ => 1,
    };
    Ok(rank)
}

impl ReshapeDimensions {
    pub(super) fn prepare(
        context: &RuntimeContext,
        args: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Option<Self>> {
        if !context.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        }) {
            return Ok(None);
        }
        let rank = rank_hint(args)?;
        if rank == 0 {
            return Err(reshape_size_error(
                "reshape: size vector must contain at least one element",
            ));
        }
        // The collection allocates exactly this Layout with one retained
        // block claim. Its capacity is fixed before parsing; no growth path
        // can change the physical quote after source decoding starts.
        let layout = Layout::array::<usize>(rank)
            .map_err(|_| reshape_size_error("reshape: size vector extent overflow"))?;
        let dimensions = AdmittedCollectionAllocator::requested_bytes(layout)
            .map_err(|_| reshape_size_error("reshape: size vector extent overflow"))?;
        let shape = SharedShape::requested_bytes_for_rank(rank)
            .map_err(|cause| reshape_error(format!("reshape: shape extent: {cause}")))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(dimensions)
                .ok()
                .and_then(|bytes| bytes.checked_add(u64::try_from(shape).ok()?))
                .ok_or_else(|| reshape_size_error("reshape: size vector extent overflow"))?,
            AdmittedCollectionAllocator::retained_claim_count() + 1,
        )
        .map_err(|cause| admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| admission_error(context, cause))?;
        let Some(receipt) = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| admission_error(context, cause))?
        else {
            return Ok(None);
        };
        let allocator = AdmittedCollectionAllocator::new(
            receipt
                .output_permit(context)
                .map_err(|cause| admission_error(context, cause))?,
        );
        let mut dimensions = AdmittedVec::new_in(allocator);
        dimensions.try_reserve_exact(rank).map_err(|cause| {
            receipt
                .checkpoint(context)
                .err()
                .map(|error| admission_error(context, error))
                .unwrap_or_else(|| reshape_error(format!("reshape: dimension storage: {cause}")))
        })?;
        Ok(Some(Self {
            dimensions,
            receipt,
            auto_index: None,
            duplicate_auto: false,
        }))
    }

    pub(super) async fn parse(
        &mut self,
        args: crate::call::FunctionArgumentView<'_>,
        context: &RuntimeContext,
    ) -> BuiltinResult<()> {
        if args.len() == 1 {
            let value = &args[0];
            match value {
                Value::Tensor(_) | Value::LogicalArray(_) => {
                    return self.extend_vector(value);
                }
                Value::GpuTensor(handle) => {
                    let downloaded = super::readback::download_control(context, handle).await?;
                    return self.extend_vector(&downloaded.value);
                }
                _ => {}
            }
        }
        for value in args {
            let token = match value {
                Value::GpuTensor(handle) => {
                    let downloaded = super::readback::download_control(context, handle).await?;
                    parse_size_scalar(&downloaded.value).await?
                }
                _ => parse_size_scalar(value).await?,
            };
            let index = self.dimensions.len();
            match token {
                DimToken::Known(value) => self.dimensions.push(value),
                DimToken::Auto => {
                    self.duplicate_auto |= self.auto_index.replace(index).is_some();
                    self.dimensions.push(1);
                }
            }
        }
        Ok(())
    }

    fn extend_vector(&mut self, value: &Value) -> BuiltinResult<()> {
        let Some(values) = tensor::borrowed_host_shape_dimensions(value) else {
            return Err(reshape_size_error(
                "reshape: size vector must be a row or column vector",
            ));
        };
        if values.len() == 0 {
            return Err(reshape_size_error(
                "reshape: size vector must contain at least one element",
            ));
        }
        for value in values {
            self.dimensions
                .push(value.map_err(reshape_dimension_error)?);
        }
        Ok(())
    }

    pub(super) fn finalize(&mut self, numel: usize) -> BuiltinResult<()> {
        finish_dimensions_in_place(
            &mut self.dimensions,
            self.auto_index,
            self.duplicate_auto,
            numel,
        )
    }

    pub(super) fn publish(self, value: Value, context: &RuntimeContext) -> BuiltinResult<Value> {
        let shape = self
            .receipt
            .with_output_allocation(context, |permit| {
                SharedShape::new_admitted_from_slice(&self.dimensions, permit)
            })
            .map_err(|cause| admission_error(context, cause))?;
        match value {
            Value::Tensor(mut tensor) => {
                tensor
                    .reshape_with_shared_shape(shape)
                    .map_err(|cause| reshape_error(format!("reshape: {cause}")))?;
                Ok(Value::Tensor(tensor))
            }
            Value::ComplexTensor(mut tensor) => {
                tensor
                    .reshape_with_shared_shape(shape)
                    .map_err(|cause| reshape_error(format!("reshape: {cause}")))?;
                Ok(Value::ComplexTensor(tensor))
            }
            Value::LogicalArray(mut logical) => {
                logical
                    .reshape_with_shared_shape(shape)
                    .map_err(|cause| reshape_error(format!("reshape: {cause}")))?;
                Ok(Value::LogicalArray(logical))
            }
            _ => unreachable!("managed reshape applies only to existing shared host payloads"),
        }
    }
}
