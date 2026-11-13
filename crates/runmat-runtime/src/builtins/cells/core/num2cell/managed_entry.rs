use runmat_value::Value;

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub(super) fn selected<'a>(
    runtime: Option<&'a RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Option<&'a RuntimeContext>> {
    let (Some(runtime), Value::StringArray(_)) = (runtime, value) else {
        return Ok(None);
    };
    let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|cause| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:num2cell:HostAllocation",
            "num2cell: allocation policy failed",
            cause,
        )
    })?;
    Ok((policy == HostAllocationPolicy::Managed).then_some(runtime))
}
