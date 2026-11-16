use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SUBPLOT-ARITY",
            "subplot expects rows, columns, and one-based position",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let invalid_shape = if index < 2 {
            argument.shape.is_definitely_nonscalar()
        } else {
            matches!(&argument.shape, ShapeFact::Shaped { dims }
                if dims.iter().any(|dim| matches!(dim, DimensionFact::Known(0)))
                || dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(size) if *size > 1)).count() > 1
                || dims.iter().skip(2).any(|dim| matches!(dim, DimensionFact::Known(size) if *size != 1)))
        };
        if !matches!(
            argument.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || invalid_shape
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SUBPLOT-GRID",
                if index < 2 {
                    "subplot rows and columns must be host real numeric scalars"
                } else {
                    "subplot position must be a nonempty host real numeric scalar or vector"
                },
                index,
            ));
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
