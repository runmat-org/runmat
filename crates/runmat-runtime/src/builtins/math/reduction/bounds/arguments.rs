//! One normalized final argument owner over the original bounds inputs.
use super::*;
use crate::call::BuiltinArguments;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;

pub(super) async fn normalize(
    runtime: Option<&RuntimeContext>,
    rest: BuiltinArguments,
) -> BuiltinResult<BuiltinArguments> {
    if rest.len() > 2 {
        return Err(bounds_invalid_argument(
            "bounds accepts at most a dimension selector and nanflag",
        ));
    }
    let mut has_nanflag = false;
    for value in rest.iter() {
        if is_nanflag(value)? {
            has_nanflag = true;
            break;
        }
    }
    let count = rest.len() + 1 + usize::from(!has_nanflag);
    // These new root-free Values have their own admitted final owners; the
    // argument-slot reservation is not a certificate for child allocations.
    let placeholder = empty_placeholder(runtime)?;
    let normalized = rest
        .try_map_preserving_roots_async(count, runtime, |index| {
            std::future::ready((|| {
                if index == 0 {
                    Ok(placeholder.clone())
                } else if let Some(value) = rest.get(index - 1) {
                    match keyword_of(value)?.as_deref() {
                        Some("omitmissing") => text("omitnan", runtime),
                        Some("includemissing") => text("includenan", runtime),
                        _ => Ok(value.clone()),
                    }
                } else {
                    text("omitnan", runtime)
                }
            })())
        })
        .await?;
    drop(rest);
    Ok(normalized)
}

fn text(value: &str, runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
    crate::call::argument_text::copy_str(value, runtime).map(|text| Value::String(text.into()))
}

fn empty_placeholder(runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
    use runmat_value::{NumericDType, Tensor};
    let error = |source| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:Bounds:Arguments",
            "bounds placeholder admission failed",
            source,
        )
    };
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[0, 0]).map_err(error)?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(error)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(error)?;
    let tensor = if let Some(receipt) = &receipt {
        let context = runtime.expect("managed placeholder has its selected context");
        let allocation = receipt
            .exact_output_tensor(context, NumericDType::F64, &[0, 0])
            .map_err(error)?;
        let tensor = allocation
            .into_tensor()
            .map_err(|source| error(HostAllocationError::Numeric(source)))?;
        receipt.checkpoint(context).map_err(error)?;
        tensor
    } else {
        Tensor::new(Vec::new(), vec![0, 0]).map_err(bounds_internal)?
    };
    Ok(Value::Tensor(tensor))
}
