use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-METACLASS-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    let runtime_class = request.arguments.first().and_then(|value| {
        if !matches!(value.residency, ResidencyFact::Host) {
            return None;
        }
        Some(match &value.kind {
            ValueKindFact::Object(object) => return object.runtime_class.clone(),
            ValueKindFact::Numeric(numeric) => match numeric.class {
                NumericClass::Double => standard::DOUBLE.owned(),
                NumericClass::Single => standard::SINGLE.owned(),
                NumericClass::Int8 => standard::INT8.owned(),
                NumericClass::UInt8 => standard::UINT8.owned(),
                NumericClass::Int16 => standard::INT16.owned(),
                NumericClass::UInt16 => standard::UINT16.owned(),
                NumericClass::Int32 => standard::INT32.owned(),
                NumericClass::UInt32 => standard::UINT32.owned(),
                NumericClass::Int64 => standard::INT64.owned(),
                NumericClass::UInt64 => standard::UINT64.owned(),
            },
            ValueKindFact::Logical => standard::LOGICAL.owned(),
            ValueKindFact::String => standard::STRING.owned(),
            ValueKindFact::Character => standard::CHAR.owned(),
            ValueKindFact::Cell(_) => standard::CELL.owned(),
            ValueKindFact::Struct(_) => standard::STRUCT.owned(),
            ValueKindFact::Callable(_) => standard::FUNCTION_HANDLE.owned(),
            ValueKindFact::ClassReference(_) => standard::META_CLASS.owned(),
            ValueKindFact::Exception(_) => standard::MEXCEPTION.owned(),
            ValueKindFact::Symbolic => standard::SYMBOLIC.owned(),
            _ => return None,
        })
    });
    let outputs = vec![ValueFact::scalar(ValueKindFact::ClassReference(
        ClassReferenceFact {
            class: None,
            runtime_class,
        },
    ))];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
