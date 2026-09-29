//! Static inference for treelayout.

use super::super::shared::{arity, double_fact, parent_vector};
use crate::{catalog::inference::finish_fixed_outputs, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 1, 1);
    parent_vector(request, &mut diagnostics);
    finish_fixed_outputs(
        entry,
        request,
        vec![
            double_fact(ShapeFact::from(vec![Some(1), None])),
            double_fact(ShapeFact::from(vec![Some(1), None])),
        ],
        diagnostics,
    )
}
