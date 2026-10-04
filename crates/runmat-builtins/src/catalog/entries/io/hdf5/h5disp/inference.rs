use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, EffectKind};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut result =
        super::super::inference::infer(request, entry, super::super::inference::Hdf5Form::Display);
    // Only an explicit single-result call captures text without writing to
    // the console. Statement and dynamic output contexts may still emit it.
    if request.outputs.requested.known_count() == Some(1) {
        result.effects.0.remove(&EffectKind::UserInterface);
        result.effects.0.remove(&EffectKind::HostCallback);
    }
    result
}
