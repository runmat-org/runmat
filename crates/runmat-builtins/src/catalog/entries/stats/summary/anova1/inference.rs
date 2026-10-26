use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let array = ValueFact::proven(
        scalar.kind.clone(),
        ShapeFact::Unknown,
        StorageFact::Unknown,
    );
    let mut fields = BTreeMap::new();
    fields.insert(
        "source".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(6)]),
            StorageFact::Dense,
        ),
    );
    fields.insert("df".into(), scalar.clone());
    fields.insert("s".into(), scalar.clone());
    fields.insert("n".into(), array.clone());
    fields.insert("means".into(), array);
    fields.insert(
        "gnames".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    let stats = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let table = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(4), Some(6)]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![scalar, table, stats]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request
        .literals
        .literal_string_at(2)
        .is_some_and(|value| value.eq_ignore_ascii_case("off"))
    {
        for effect in [
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
        ] {
            result.effects.0.remove(&effect);
        }
    }
    if request.arguments.is_empty() || request.arguments.len() > 3 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "anova1 requires y and at most group and display",
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double | NumericClass::Single,
                domain: NumericDomain::Real
            }) | ValueKindFact::Unknown
        ) {
            result.diagnostics.push(InferenceDiagnostic::error(
                "INPUT",
                "anova1 y must be real single or double",
            ));
        }
    }
    result
}
