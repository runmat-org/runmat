use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SINC-ARITY",
            "sinc expects one numeric input",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let output = match request.arguments.first() {
        Some(input) => output_for(input, &mut diagnostics),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let output = if diagnostics.is_empty() {
        output
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn output_for(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    if matches!(input.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SINC-STORAGE",
            "sinc requires dense input",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let (class, domain) = match input.kind {
        ValueKindFact::Numeric(NumericFact { class, domain }) => {
            if domain == NumericDomain::Complex && class.integer_class().is_some() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SINC-COMPLEX-INTEGER",
                    "sinc does not accept complex integer input",
                    0,
                ));
                return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
            let output_class = match class {
                NumericClass::Single => NumericClass::Single,
                NumericClass::Double => NumericClass::Double,
                NumericClass::Int8
                | NumericClass::UInt8
                | NumericClass::Int16
                | NumericClass::UInt16
                | NumericClass::Int32
                | NumericClass::UInt32
                | NumericClass::Int64
                | NumericClass::UInt64 => NumericClass::Double,
            };
            (output_class, domain)
        }
        ValueKindFact::Logical => (NumericClass::Double, NumericDomain::Real),
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SINC-INPUT",
                "sinc expects real or complex numeric input",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    };
    let storage = match input.shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        input.shape.clone(),
        storage,
    );
    output.residency = ResidencyFact::Unknown;
    output
}
