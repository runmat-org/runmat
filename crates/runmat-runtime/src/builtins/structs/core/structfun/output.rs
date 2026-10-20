use crate::builtins::common::uniform_scalar_output::{
    UniformScalarCollector, UniformScalarError, UniformScalarFailure,
};
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::RootedValue;
use runmat_value::Value;

use super::error;
mod slots;
mod structure;
pub(super) use slots::Slots;
use structure::StructureCollector;

pub(super) struct Collector<'a> {
    // Structure payload/roots retire together before the remaining mode state.
    structure: Option<StructureCollector<'a>>,
    mode: CollectorMode<'a>,
}
enum CollectorMode<'a> {
    Uniform(UniformScalarCollector<'a>),
    Structure,
}
impl<'a> Collector<'a> {
    pub(super) fn new<'n, I>(
        uniform: bool,
        names: I,
        runtime: Option<&'a RuntimeContext>,
    ) -> crate::BuiltinResult<Self>
    where
        I: ExactSizeIterator<Item = &'n str> + Clone,
    {
        if uniform {
            return Ok(Self {
                structure: None,
                mode: CollectorMode::Uniform(match runtime {
                    Some(runtime) => UniformScalarCollector::in_context(runtime, names.len())
                        .map_err(map_uniform_error)?,
                    None => {
                        ordinary_policy()?;
                        UniformScalarCollector::new()
                    }
                }),
            });
        }
        Ok(Self {
            structure: Some(StructureCollector::new(names, runtime)?),
            mode: CollectorMode::Structure,
        })
    }
    pub(super) fn push(&mut self, field: &str, value: RootedValue) -> crate::BuiltinResult<()> {
        match &mut self.mode {
            CollectorMode::Uniform(collector) => {
                collector.push(value.value()).map_err(map_uniform_error)
            }
            CollectorMode::Structure => self
                .structure
                .as_mut()
                .expect("structure mode")
                .push(field, value),
        }
    }
    pub(super) fn finish(
        mut self,
        field_count: usize,
        runtime: Option<&RuntimeContext>,
    ) -> crate::BuiltinResult<RootedValue> {
        match self.mode {
            CollectorMode::Uniform(collector) => root(
                collector
                    .finish(&[field_count, 1])
                    .map_err(map_uniform_error)?,
                runtime,
            ),
            CollectorMode::Structure => self
                .structure
                .take()
                .expect("structure mode")
                .finish(runtime),
        }
    }
}
pub(super) fn root(
    value: Value,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => {
            ordinary_policy()?;
            RootedValue::new(value).map_err(|e| e.into_runtime_error())
        }
    }
}
pub(super) fn allocation(
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Structfun:Allocation",
        "structfun output allocation failed",
        source,
    )
}
pub(super) fn ordinary_policy() -> crate::BuiltinResult<()> {
    match HostAllocationReceipt::policy(None).map_err(|e| allocation(None, e))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => {
            Err(allocation(None, HostAllocationError::MissingManagedOwner))
        }
    }
}
fn map_uniform_error(value: UniformScalarError) -> crate::RuntimeError {
    match value.into_failure() {
        Err(error)=>error,
        Ok(UniformScalarFailure::Materialization(reason)) =>error::internal(format!("structfun: {reason}")),
        Ok(UniformScalarFailure::InvalidStorage(kind)) =>error::internal(format!("structfun: {kind} has no storage value")),
        Ok(UniformScalarFailure::CharacterRank) =>error::uniform("structfun: uniform character outputs must form a 2-D column"),
        Ok(UniformScalarFailure::CharacterCount) =>error::uniform("structfun: callback returned the wrong number of characters"),
        Ok(UniformScalarFailure::SizeOverflow) =>error::internal("structfun: output size exceeds platform limits"),
        Ok(UniformScalarFailure::NonScalar) =>error::uniform("structfun: callback must return scalar numeric, logical, character, or complex values when UniformOutput is true"),
        Ok(UniformScalarFailure::Heterogeneous) =>error::uniform("structfun: callback outputs with UniformOutput=true must have the same data type on every invocation"),
    }
}
