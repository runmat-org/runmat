//! Dummy-variable output facts.
use super::common::{error, finish, numeric};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DimensionFact, NumericClass, ShapeFact, ValueKindFact,
};

pub(in super::super) fn dummyvar(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(error("ARITY", "dummyvar requires one grouping argument"));
    }
    let rows = request.arguments.first().and_then(|value| {
        let ShapeFact::Shaped { dims } = &value.shape else {
            return None;
        };
        match &value.kind {
            // Only a nonscalar numeric matrix is split into separate column
            // grouping variables. Every other numeric or logical array is
            // flattened into one grouping variable by the runtime.
            ValueKindFact::Numeric(_)
                if dims.len() == 2
                    && matches!(dims[0], DimensionFact::Known(n) if n > 1)
                    && matches!(dims[1], DimensionFact::Known(n) if n > 1) =>
            {
                dims.first().cloned()
            }
            ValueKindFact::Numeric(_) | ValueKindFact::Logical => {
                value.shape.element_count().map(DimensionFact::Known)
            }
            _ => None,
        }
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
