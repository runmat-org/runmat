//! Dense/GPU/nonfloating compatibility EVD has no finite host owner yet.
//! This is an explicit L25 fence, never a Managed-to-ordinary fallback.
use crate::{
    context::{
        host_allocation::{
            host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
            HostAllocationReceipt,
        },
        RuntimeContext,
    },
    BuiltinResult,
};
use runmat_value::Value;
pub(super) enum Selected<'a> {
    Iterative,
    Legacy(DenseWork<'a>),
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
            Ok(Self::Legacy(DenseWork::new(runtime)?))
        }
    }
}
pub(super) struct DenseWork<'a> {
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> DenseWork<'a> {
    fn new(runtime: Option<&'a RuntimeContext>) -> BuiltinResult<Self> {
        let result = Self { runtime };
        result.check_with(HostAllocationError::MissingManagedOwner)?;
        Ok(result)
    }
    fn check_with(&self, managed: HostAllocationError) -> BuiltinResult<()> {
        let source = match HostAllocationReceipt::policy(self.runtime) {
            Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
            Ok(HostAllocationPolicy::Managed) => managed,
            Err(source) => source,
        };
        Err(super::with_context(host_allocation_runtime_error(
            self.runtime,"RunMat:eigs:DenseWorkspace",
            format!("eigs: exact dense compatibility EVD requires unrestricted allocation until its shared full-EVD owner is available; {source}"),
            source,
        )))
    }
    pub fn check(&self) -> BuiltinResult<()> {
        self.check_with(HostAllocationError::StaleAuthority)
    }
    pub fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result?;
        self.check()?;
        Ok(output)
    }
}
