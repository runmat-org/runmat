//! Preparation follows the dispatcher's selected implementation authority.
use crate::call::BuiltinArguments;
use crate::RuntimeError;

#[derive(Clone, Copy)]
pub(super) enum SelectedBindingOrigin {
    CanonicalOrScoped,
    LegacyInventory,
}

pub(super) async fn prepare_retained_selected(
    origin: SelectedBindingOrigin,
    name: &str,
    args: BuiltinArguments,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<BuiltinArguments, RuntimeError> {
    if !matches!(origin, SelectedBindingOrigin::LegacyInventory) {
        return Ok(args);
    }
    let Some(context) = context else {
        return Ok(args);
    };
    let Some(service) = context.service_ports().acceleration().cloned() else {
        return Ok(args);
    };
    // This selected historical port genuinely requires an ordinary Vec.
    // Its policy-checked bridge is never used by canonical/scoped bindings.
    let ordinary = args.materialize_legacy_contiguous(Some(context))?;
    let result = service
        .prepare_legacy(context.clone(), name.to_owned(), ordinary)
        .await?;
    let prepared = crate::execution::RootedValues::in_context(result, context)?;
    drop(args); // successor roots/publication exist before original retirement.
    Ok(prepared.into())
}
