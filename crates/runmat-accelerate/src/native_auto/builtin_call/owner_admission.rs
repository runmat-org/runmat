//! The retained preparation provider must own every borrowed resident operand.
//! This does not choose another provider or materialize a cross-device value.
use super::*;

pub(super) fn matches_selected_owner(
    selected: &'static dyn AccelProvider,
    operands: &[Value],
) -> PreparationResult<bool> {
    let mut matches = true;
    for operand in operands {
        runmat_runtime::dispatcher::try_visit_gpu_handles(operand, &mut |handle| {
            let owner =
                runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle(handle);
            matches &= owner.is_some_and(|owner| std::ptr::eq(owner, selected));
        })?;
    }
    Ok(matches)
}
