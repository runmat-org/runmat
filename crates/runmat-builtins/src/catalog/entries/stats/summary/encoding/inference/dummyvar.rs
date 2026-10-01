//! Dummy-variable output facts.
use super::common::{error, finish, numeric};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DimensionFact, NumericClass, ShapeFact, ValueKindFact,
};

pub(super) fn dummyvar(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(error("ARITY", "dummyvar requires one grouping argument"));
    }
    let rows = request
        .arguments
        .first()
        .and_then(|value| match (&value.kind, &value.shape) {
            (ValueKindFact::Numeric(_) | ValueKindFact::Logical, ShapeFact::Shaped { dims })
                if dims.len() >= 2 =>
            {
                // Cell shape describes the container, not its grouping rows.
                if dims.first() == Some(&DimensionFact::Known(1)) {
                    dims.get(1).cloned()
                } else {
                    dims.first().cloned()
                }
            }
            _ => None,
        });
    let shape = ShapeFact::Shaped {
        dims: vec![
            rows.unwrap_or(DimensionFact::Unknown),
            DimensionFact::Unknown,
        ],
    };
    finish(
        request,
        entry,
        numeric(NumericClass::Double, shape),
        diagnostics,
    )
}
