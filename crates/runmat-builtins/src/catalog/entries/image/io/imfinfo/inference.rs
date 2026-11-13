use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-IMFINFO-ARITY",
            "imfinfo requires a filename and optional format",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !super::super::text::could_be_scalar_text(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IMFINFO-TEXT",
                "filename and format must be scalar text",
                index,
            ));
        }
    }
    let mut fields = BTreeMap::new();
    for name in ["Filename", "Format", "ColorType"] {
        fields.insert(
            name.to_string(),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), None]),
                StorageFact::Dense,
            ),
        );
    }
    for name in ["FileModDate", "FormatVersion"] {
        fields.insert(
            name.to_string(),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), Some(0)]),
                StorageFact::Dense,
            ),
        );
    }
    for name in ["FileSize", "Width", "Height", "BitDepth"] {
        fields.insert(
            name.to_string(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        );
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-IMFINFO-OUTPUTS",
            "imfinfo returns one metadata struct",
            1,
        ));
    }
    result.diagnostics.extend(diagnostics);
    result
}
