//! Representation facts shared by the existing binary64 callback solvers.
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

/// Descriptive admission only: finite payloads, extension policy and allocation
/// liveness remain with the runtime numerical and physical owners.
pub fn nonlinear_real_numeric_metadata(value: &ValueFact, empty: bool) -> bool {
    (matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
        || matches!(value.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real))
        && value.storage != StorageFact::Sparse
        && (empty || value.shape.element_count() != Some(0))
}

pub(super) fn numeric(
    value: Option<&ValueFact>,
    index: usize,
    empty: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    if !nonlinear_real_numeric_metadata(value, empty) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NONLINEAR-REAL",
            "solver data must be a supported real numeric array",
            index,
        ));
    }
}

pub(super) fn options(
    value: Option<&ValueFact>,
    index: usize,
    empty: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    if empty
        && value.shape.element_count() == Some(0)
        && matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    {
        return;
    }
    if !matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Struct(_)
    ) || value.shape.element_count().is_some_and(|n| n != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-NONLINEAR-OPTIONS",
            "options must be a scalar struct or an admitted empty placeholder",
            index,
        ));
    }
}

pub(super) fn bound(
    value: Option<&ValueFact>,
    index: usize,
    variables: Option<usize>,
    partial: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    numeric(value, index, true, diagnostics);
    if let Some((n, count)) = variables.zip(value.and_then(|v| v.shape.element_count())) {
        if count != 0
            && if partial {
                count > n
            } else {
                count != 1 && count != n
            }
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-NONLINEAR-BOUND",
                "bound cardinality does not match the admitted solver form",
                index,
            ));
        }
    }
}

pub(super) fn double(shape: ShapeFact, storage: StorageFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    fact.residency = ResidencyFact::Host;
    fact
}

pub(super) fn scalar() -> ValueFact {
    double(ShapeFact::Scalar, StorageFact::Scalar)
}

pub(super) fn like(source: Option<&ValueFact>) -> ValueFact {
    let shape = source
        .map(|v| v.shape.clone())
        .unwrap_or(ShapeFact::Unknown);
    let storage = match source {
        Some(v) if v.residency == ResidencyFact::Host => match v.storage {
            StorageFact::Scalar => match v.kind {
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    ..
                }) => StorageFact::Dense,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical => StorageFact::Scalar,
                _ => StorageFact::Unknown,
            },
            StorageFact::Dense => StorageFact::Dense,
            _ => StorageFact::Unknown,
        },
        Some(v) if v.shape.element_count().is_some_and(|n| n != 1) => StorageFact::Dense,
        _ => StorageFact::Unknown,
    };
    double(shape, storage)
}

pub(super) fn matrix(rows: Option<usize>, columns: Option<usize>) -> ValueFact {
    double(ShapeFact::from(vec![rows, columns]), StorageFact::Dense)
}

pub(super) fn diagnostics(numeric_fields: &[&str]) -> ValueFact {
    let mut fields = BTreeMap::new();
    for name in numeric_fields {
        fields.insert((*name).into(), scalar());
    }
    for name in ["algorithm", "message"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    )
}

pub(super) fn lambda(solution: &ValueFact) -> ValueFact {
    let fields = [
        ("lower".into(), solution.clone()),
        ("upper".into(), solution.clone()),
    ]
    .into_iter()
    .collect();
    ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    )
}

pub(super) fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    mut diagnostics: Vec<InferenceDiagnostic>,
    callback: Option<CallInference>,
) -> CallInference {
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > outputs.len())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-NONLINEAR-OUTPUTS",
            "too many requested solver outputs",
            outputs.len(),
        ));
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(callback) = callback {
        contract.effects.0.extend(callback.effects.0);
        contract.capabilities.0.extend(callback.capabilities.0);
        diagnostics.extend(callback.diagnostics);
    }
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    if !diagnostics.is_empty() {
        result
            .outputs
            .iter_mut()
            .for_each(|v| *v = ValueFact::unknown(DynamicReason::UnsupportedRepresentation));
    }
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn field<'a>(fields: &'a StructFact, name: &str) -> Option<&'a ValueFact> {
    fields
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}
