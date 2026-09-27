//! Preparation follows the dispatcher's selected implementation authority.
use crate::RuntimeError;
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(super) enum SelectedBindingOrigin {
    CanonicalOrScoped,
    LegacyInventory,
}

pub(super) async fn prepare_selected(
    origin: SelectedBindingOrigin,
    name: &str,
    args: &[Value],
) -> Result<Option<Vec<Value>>, RuntimeError> {
    if !matches!(origin, SelectedBindingOrigin::LegacyInventory) {
        return Ok(None);
    }
    let Some(context) = crate::context::legacy::active() else {
        return Ok(None);
    };
    let Some(service) = context.service_ports().acceleration().cloned() else {
        return Ok(None);
    };
    service
        .prepare_legacy(context, name.to_owned(), args.to_vec())
        .await
        .map(Some)
}
