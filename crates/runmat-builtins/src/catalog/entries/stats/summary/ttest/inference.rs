use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;
fn geometry(request: &CallRequest) -> Option<(ShapeFact, ShapeFact)> {
    let mut shape = request.arguments.first()?.shape.known_dims()?;
    let start = match request.arguments.get(1) {
        None => 1,
        Some(value) if matches!(value.kind, ValueKindFact::Character | ValueKindFact::String) => 1,
        Some(value) if matches!(value.kind, ValueKindFact::Numeric(_)) => 2,
        _ => return None,
    };
    if !(request.arguments.len() - start).is_multiple_of(2) {
        return None;
    }
    let mut dim = None;
    for index in (start..request.arguments.len()).step_by(2) {
        let name = request.literals.literal_string_at(index)?;
        match super::TtestOptionName::from_public(&name)? {
            super::TtestOptionName::Dim => {
                let value = LiteralContext::numeric_from_literal(
                    request.literals.literal_args.get(index + 1)?,
                )?;
                dim = Some(decode_shape_dimension(value).ok()?.checked_sub(1)?);
            }
            super::TtestOptionName::Alpha | super::TtestOptionName::Tail => {}
        }
    }
    let dim = match dim {
        Some(dim) => dim,
        None => {
            let mut selected = 0;
            for (axis, extent) in shape.iter().enumerate() {
                match extent {
                    Some(1) => {}
                    Some(_) => {
                        selected = axis;
                        break;
                    }
                    None => return None,
                }
            }
            selected
        }
    };
    // Do not allocate compile-time metadata proportional to a dimension
    // literal. Runtime can admit a larger rank through its fallible owner.
    if dim >= shape.len().max(2) {
        return None;
    }
    shape.resize(shape.len().max(2), Some(1));
    shape[dim] = Some(1);
    let output = ShapeFact::from(shape.clone());
    shape[dim] = Some(2);
    Some((output, ShapeFact::from(shape)))
}
fn storage(shape: &ShapeFact) -> StorageFact {
    match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    }
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // Runtime option names and Dim can change the tested axis. Class and
    // scalar-structure schema remain invariant; no value-dependent shape lies.
    let (shape, interval) = geometry(request).unwrap_or((ShapeFact::Unknown, ShapeFact::Unknown));
    let number = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        storage(&shape),
    );
    let h = ValueFact::proven(ValueKindFact::Logical, shape.clone(), storage(&shape));
    let fields = BTreeMap::from([
        ("tstat".into(), number.clone()),
        ("df".into(), number.clone()),
        ("sd".into(), number.clone()),
    ]);
    let stats = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let mut ci = number.clone();
    ci.shape = interval;
    ci.storage = storage(&ci.shape);
    let mut contract = CallContract::fixed(vec![h, number, ci, stats]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.is_empty() {
        result.diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "ttest requires sample data",
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                domain: NumericDomain::Real
            }) | ValueKindFact::Unknown
        ) {
            result.diagnostics.push(InferenceDiagnostic::error(
                "INPUT",
                "ttest samples must be real single or double",
            ));
        }
    }
    result
}
