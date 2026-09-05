mod magnitude;
mod phase;
mod sign;

#[cfg(test)]
mod tests;

use crate::{BuiltinCatalogEntry, MagnitudePhaseSignKind};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    kind: MagnitudePhaseSignKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match kind {
        MagnitudePhaseSignKind::Magnitude => magnitude::infer(request, entry),
        MagnitudePhaseSignKind::Phase => phase::infer(request, entry),
        MagnitudePhaseSignKind::Sign => sign::infer(request, entry),
    }
}
