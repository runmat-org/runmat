use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    let count = request.arguments.len();
    if count > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-COLORMAP-ARITY",
            "colormap accepts a map with an optional target",
            2,
        ));
    }
    let mut rows = None;
    for (index, argument) in request.arguments.iter().enumerate() {
        let target = count == 2 && index == 0;
        let possibly_query = count == 1
            && argument.shape.element_count() != Some(0)
            && argument.shape.element_count().is_none_or(|n| n == 1);
        let invalid_kind = match argument.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Character => {
                target || !crate::catalog::inference::possible_character_row(&argument.shape)
            }
            ValueKindFact::String => {
                target || !matches!(argument.storage, StorageFact::Unknown | StorageFact::Scalar)
            }
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Complex
                    || argument.storage == StorageFact::Sparse
                    || (target && argument.shape.element_count().is_some_and(|n| n != 1))
            }
            _ => true,
        };
        if invalid_kind
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-COLORMAP-INPUT",
                "colormap requires a host numeric graphics target, name, or real RGB matrix",
                index,
            ));
        }
        if !target && !possibly_query && argument.numeric().is_some() {
            if let ShapeFact::Shaped { dims: dimensions } = &argument.shape {
                // Tensor's rank-one representation is a row (rows=1, cols=dims[0]).
                let column = if dimensions.len() == 1 {
                    dimensions.first()
                } else {
                    dimensions.get(1)
                };
                if column.is_some_and(|dim| matches!(dim, DimensionFact::Known(n) if *n != 3))
                    || dimensions
                        .iter()
                        .skip(2)
                        .any(|dim| matches!(dim, DimensionFact::Known(n) if *n != 1))
                    || dimensions
                        .first()
                        .is_some_and(|dim| matches!(dim, DimensionFact::Known(0)))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-COLORMAP-SHAPE",
                        "RGB maps must be m-by-3",
                        index,
                    ));
                } else if dimensions.len() == 1 {
                    rows = Some(1);
                } else if let Some(DimensionFact::Known(n)) = dimensions.first() {
                    rows = Some(*n);
                }
            }
        }
        if !target
            && request
                .literals
                .literal_string_at(index)
                .is_some_and(|name| name.trim().eq_ignore_ascii_case("default"))
        {
            rows = Some(256);
        }
    }
    // Palette lookup names and graphics identity validity belong to their domains.
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, Some(3)]),
        StorageFact::Dense,
    );
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
