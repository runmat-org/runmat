//! Conservative object/result facts. Symbolic values are not numerical
//! arrays, and runtime-selected property/index paths are not guessed.
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
fn object(class: StaticClassIdentity, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    )
}
pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut dimensions = Vec::new();
    if request.arguments.is_empty() {
        return finish_fixed(
            entry,
            request,
            object(
                standard::OPTIMIZATION_EXPRESSION,
                ShapeFact::from(vec![Some(0), Some(0)]),
            ),
            diagnostics,
        );
    }
    for index in 0..request.arguments.len() {
        let Some(number) = request.literals.numeric_at(index) else {
            break;
        };
        match decode_shape_dimension(number) {
            Ok(n) if n > 0 => dimensions.push(Some(n)),
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMEXPR-DIMENSIONS",
                "dimensions must be positive integers",
                index,
            )),
        }
    }
    let shape = if !dimensions.is_empty() && dimensions.len() == request.arguments.len() {
        if dimensions.len() == 1 {
            dimensions.push(Some(1));
        }
        ShapeFact::from(dimensions)
    } else {
        ShapeFact::Unknown
    };
    finish_fixed(
        entry,
        request,
        object(standard::OPTIMIZATION_EXPRESSION, shape),
        diagnostics,
    )
}
pub(super) fn expression(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object(standard::OPTIMIZATION_EXPRESSION, ShapeFact::Unknown),
        Vec::new(),
    )
}
pub(super) fn equality(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object(standard::OPTIMIZATION_EQUALITY, ShapeFact::Unknown),
        Vec::new(),
    )
}
pub(super) fn inequality(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        object(standard::OPTIMIZATION_INEQUALITY, ShapeFact::Unknown),
        Vec::new(),
    )
}
pub(super) fn read(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        Vec::new(),
    )
}
pub(super) fn numeric(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        request
            .arguments
            .first()
            .map_or(ShapeFact::Unknown, |value| value.shape.clone()),
        StorageFact::Unknown,
    );
    finish_fixed(entry, request, output, Vec::new())
}
