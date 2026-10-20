//! A selected refinement creates one new immutable callable generation.
//! Definition authority remains owned by the existing source resolver; this
//! allocation plan covers only the new data header and copied display path.

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{user_functions, RuntimeError};
use runmat_value::{Closure, Utf8Text};

pub(crate) fn bind_selected_closure(
    closure: Closure,
    function: usize,
) -> Result<Closure, RuntimeError> {
    let runtime = crate::context::legacy::active();
    // These existing resolver-owned contracts are independent of the new
    // callable's physical header; a header claim is not their certificate.
    let binding = user_functions::bound_function_binding_for(function);
    let info = user_functions::source_function_for(function);
    let role = info
        .as_ref()
        .map_or(runmat_types::FunctionSourceRole::Unknown, |info| info.role);
    let source = if closure.source_file.is_none() {
        info.and_then(|info| info.source_file)
    } else {
        None
    };
    let error = |source| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Callable:Binding",
            "callable refinement allocation failed",
            source,
        )
    };
    let text_bytes = source
        .as_ref()
        .map_or(Ok(0), |text| {
            Utf8Text::requested_bytes_for_capacity(text.len())
        })
        .map_err(|source| error(HostAllocationError::Numeric(source)))?;
    let raw = Closure::resolved_header_bytes()
        .checked_add(text_bytes)
        .ok_or_else(|| error(HostAllocationError::Overflow))?;
    let raw = u64::try_from(raw).map_err(|_| error(HostAllocationError::Overflow))?;
    let retained =
        HostAllocationPlan::retained_exact_output_bytes(raw, 1 + usize::from(text_bytes != 0))
            .map_err(error)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(error)?;
    let Some(receipt) = HostAllocationReceipt::admit(runtime.as_ref(), plan).map_err(error)? else {
        return Ok(closure.bind_resolved_unbounded(binding, role, source));
    };
    let runtime = runtime
        .as_ref()
        .expect("managed allocation requires its selected context");
    let permit = receipt.output_permit(runtime).map_err(error)?;
    receipt.checkpoint(runtime).map_err(error)?;
    // The legacy source catalog returns owned String metadata. Its conversion
    // is an explicit admitted copy, not an implicit clone during Value retain.
    let source = source
        .as_ref()
        .map(|text| {
            Utf8Text::try_copy_str_exact_with(text, &permit, |_| receipt.checkpoint(runtime))
        })
        .transpose()
        .map_err(|source| match source {
            runmat_value::ExactArrayInitializationError::Allocation(source) => {
                error(HostAllocationError::Numeric(source))
            }
            runmat_value::ExactArrayInitializationError::Element(source) => error(source),
        })?;
    let resolved = closure
        .try_bind_resolved(binding, role, source, &permit)
        .map_err(|source| error(HostAllocationError::Numeric(source)))?;
    receipt.checkpoint(runtime).map_err(error)?;
    Ok(resolved)
}
