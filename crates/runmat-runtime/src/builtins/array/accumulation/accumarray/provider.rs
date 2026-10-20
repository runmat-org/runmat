use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::error;

pub(super) struct MaterializedRequest {
    pub subscripts: Value,
    pub data: Value,
    pub options: crate::call::BuiltinArguments,
    pub roots: crate::execution::RootedValueLease,
    pub runtime: Option<crate::context::RuntimeContext>,
}

struct ScalarInputs {
    subscripts: Value,
    data: Value,
    // Retire source payloads before their original roots, including failure.
    _roots: crate::execution::RootedValueLease,
}

pub(super) async fn materialize(
    runtime: Option<crate::context::RuntimeContext>,
    subscripts: Value,
    data: Value,
    options: crate::call::BuiltinArguments,
) -> BuiltinResult<MaterializedRequest> {
    let roots = root_scalars(&subscripts, &data, runtime.as_ref())?;
    let inputs = ScalarInputs {
        subscripts,
        data,
        _roots: roots,
    };
    reject_resident_integer(&inputs.data, "input data")?;
    if let Some(fill) = options.get(2) {
        reject_resident_integer(fill, "fill value")?;
    }
    let subscripts = gather_if_needed_async(&inputs.subscripts).await?;
    let data = gather_if_needed_async(&inputs.data).await?;
    let options =
        crate::dispatcher::gather_builtin_arguments_async(options, runtime.as_ref()).await?;
    let roots = root_scalars(&subscripts, &data, runtime.as_ref())?;
    Ok(MaterializedRequest {
        subscripts,
        data,
        options,
        roots,
        runtime,
    })
}

fn reject_resident_integer(value: &Value, role: &str) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        return Err(error::invalid(format!(
            "accumarray: GPU {role} must be logical, single, or double"
        )));
    }
    Ok(())
}

fn root_scalars(
    subscripts: &Value,
    data: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::execution::RootedValueLease> {
    let values = [subscripts, data].into_iter();
    if let Some(runtime) = runtime {
        return crate::execution::RootedValueLease::for_values_iterator_in_context(values, runtime);
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    };
    let allocation = |source| {
        host_allocation_runtime_error(
            None,
            "RunMat:accumarray:Roots",
            "accumarray input root admission failed",
            source,
        )
    };
    if HostAllocationReceipt::policy(None).map_err(allocation)? != HostAllocationPolicy::Unbounded {
        return Err(allocation(HostAllocationError::MissingManagedOwner));
    }
    crate::execution::RootedValueLease::for_values(values)
        .map_err(|error| error.into_runtime_error())
}
