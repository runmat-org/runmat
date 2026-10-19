use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-COLORORDER-ARITY",
            "colororder accepts a palette with an optional graphics target",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let target =
            index == 0 && (count == 2 || matches!(argument.kind, ValueKindFact::Object(_)));
        let invalid = if target {
            (match &argument.kind {
                ValueKindFact::Unknown => false,
                ValueKindFact::Numeric(numeric) => {
                    numeric.domain != NumericDomain::Real || argument.storage == StorageFact::Sparse
                }
                ValueKindFact::Object(_) => {
                    crate::catalog::inference::known_invalid_scalar_object_handle(
                        argument,
                        &[standard::FIGURE, standard::AXES],
                    )
                }
                _ => true,
            }) || {
                argument
                    .shape
                    .element_count()
                    .is_some_and(|elements| elements != 1)
            }
        } else {
            match argument.kind {
                ValueKindFact::Unknown | ValueKindFact::Cell(_) => false,
                ValueKindFact::Numeric(numeric) => {
                    numeric.domain == NumericDomain::Complex
                        || matches!(argument.storage, StorageFact::Sparse)
                }
                ValueKindFact::String => false,
                ValueKindFact::Character => false,
                _ => true,
            }
        };
        if invalid
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-COLORORDER-INPUT",
                "colororder requires a host target and RGB matrix, palette name, or text colors",
                index,
            ));
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, Some(3)]),
        StorageFact::Dense,
    );
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
