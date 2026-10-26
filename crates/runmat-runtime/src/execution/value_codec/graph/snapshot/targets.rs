//! One target normalization for worker calls and portable callable sets.

use super::*;

pub(super) fn selected_input(target: &CallableTarget) -> Result<Input<'_>, ValueCodecError> {
    Ok(match target {
        CallableTarget::Scoped(_) => {
            return Err(ValueCodecError::unsupported(
                "$",
                "invocation-scoped callable is not portable",
            ));
        }
        CallableTarget::Bound(binding) => Input::BoundTarget(binding),
        CallableTarget::BoundLexical { closure, .. } => Input::Borrowed(closure.as_value()),
        CallableTarget::CapturedResolved { .. } => {
            return Err(ValueCodecError::unsupported(
                "$",
                "captured callable has no exact selected graph target authority",
            ));
        }
        CallableTarget::FevalForward(value) => Input::Borrowed(value),
        CallableTarget::RetainedFevalForward(value) => Input::Borrowed(value.value()),
        CallableTarget::Resolved { .. } | CallableTarget::RetainedBuiltin(_) => {
            return Err(ValueCodecError::unsupported(
                "$",
                "unbound resolved callable has no selected graph target authority",
            ));
        }
    })
}

/// Prepare every selected user-code callback together. One encoder owns shared
/// activation/cell identity across objective, output and plot callbacks.
/// Compiled builtin identities are recorded separately by the caller; text is
/// never promoted to selected user-code authority here.
pub fn prepare_callable_targets<'a>(
    descriptors: &'a [crate::call::descriptor::CallableDescriptor],
    issuer: &'a super::super::GraphSourceIssuer,
    limits: ValueLimits,
    workspace: Option<&crate::execution::value_codec::ValidationWorkspaceOwner>,
) -> Result<PreparedPayloadGraph<'a>, ValueCodecError> {
    issuer.ensure_live()?;
    if descriptors.is_empty()
        || descriptors.len() > runmat_execution::value::PortableSelectedCode::MAX_CALLABLE_ROOTS
    {
        return Err(invalid(
            "callable capture root count exceeds its portable ordinal contract",
        ));
    }
    let mut inputs = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        if !descriptor.args.is_empty() {
            return Err(invalid("callable capture requires target-only descriptors"));
        }
        if !matches!(
            descriptor.target,
            CallableTarget::Bound(_) | CallableTarget::BoundLexical { .. }
        ) {
            return Err(ValueCodecError::unsupported(
                "$",
                "callable capture requires an actual selected bound handle",
            ));
        }
        inputs.push(selected_input(&descriptor.target)?);
    }
    prepare_inputs(inputs, issuer, limits, workspace)
}
