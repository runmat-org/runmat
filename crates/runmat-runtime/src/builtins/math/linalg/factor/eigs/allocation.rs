//! Route selection captures authority, never a blanket allocation grant.
//! The shared EVD, parser slots and published factors admit their own storage.
use crate::{
    context::{
        host_allocation::{
            host_allocation_runtime_error, HostAllocationCheckpoint, HostAllocationReceipt,
        },
        RuntimeContext,
    },
    BuiltinResult,
};
use runmat_value::Value;
pub(super) enum Selected<'a> {
    Iterative,
    Dense(DenseExecution<'a>),
}
impl<'a> Selected<'a> {
    pub fn admit(
        value: &Value,
        arguments: &crate::call::BuiltinArguments,
        resident: bool,
        runtime: Option<&'a RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if super::super::iterative::controls::is_callable(value) {
            if resident {
                return Err(super::error_with_detail(
                    "eigs: callable operands and controls require host storage",
                    &super::ERROR_UNSUPPORTED,
                ));
            }
            Ok(Self::Iterative)
        } else if matches!(value, Value::SparseTensor(_))
            && !super::is_nonfloating_supported_coefficient_matrix(value)
            && !resident
            && !arguments.first().is_some_and(|value| {
                super::is_generalized_b(value)
                    && super::is_nonfloating_supported_coefficient_matrix(value)
            })
        {
            Ok(Self::Iterative)
        } else {
            Ok(Self::Dense(DenseExecution::new(runtime)?))
        }
    }
}
pub(super) struct DenseExecution<'a> {
    runtime: Option<&'a RuntimeContext>,
    owner: Option<HostAllocationCheckpoint>,
}
impl<'a> DenseExecution<'a> {
    fn new(runtime: Option<&'a RuntimeContext>) -> BuiltinResult<Self> {
        let owner = HostAllocationReceipt::capture_owner(runtime).map_err(|source| {
            super::with_context(host_allocation_runtime_error(
                runtime,
                "RunMat:eigs:DenseWorkspace",
                "eigs: dense execution authority",
                source,
            ))
        })?;
        let result = Self { runtime, owner };
        result.check()?;
        Ok(result)
    }
    pub fn check(&self) -> BuiltinResult<()> {
        if let Some(owner) = &self.owner {
            owner
                .check(self.runtime.expect("managed dense context"))
                .map_err(|source| {
                    super::with_context(host_allocation_runtime_error(
                        self.runtime,
                        "RunMat:eigs:DenseWorkspace",
                        "eigs: dense execution authority changed",
                        source,
                    ))
                })
        } else {
            super::super::iterative::copy::unbounded_checkpoint(self.runtime, "eigs")
        }
    }
    pub fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result?;
        self.check()?;
        Ok(output)
    }
}
