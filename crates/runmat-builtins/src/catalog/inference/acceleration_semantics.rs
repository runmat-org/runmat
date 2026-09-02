use super::{argument_error, finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_gather(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-GATHER-ARITY",
            "gather requires at least one input",
            0,
        ));
    }
    let outputs = request
        .arguments
        .iter()
        .cloned()
        .map(|mut fact| {
            fact.residency = ResidencyFact::Host;
            fact
        })
        .collect::<Vec<_>>();
    if let Some(requested) = request.outputs.requested.known_count() {
        let valid = requested == 0
            || (request.arguments.len() == 1 && requested == 1)
            || (request.arguments.len() > 1 && requested == request.arguments.len());
        if !valid {
            diagnostics.push(InferenceDiagnostic::error(
                "RM-CATALOG-GATHER-OUTPUTS",
                "gather output count must match its input count",
            ));
        }
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

pub(super) fn infer_gpu_array(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(source) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-GPUARRAY-ARITY",
            "gpuArray requires an input value",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let literals = &request.literals.literal_args;
    let like_index = literals
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(index, literal)| {
            literal_text(literal)
                .is_some_and(|value| value.eq_ignore_ascii_case("like"))
                .then_some(index)
        });
    let mut output = like_index
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(|| source.clone());

    let explicit_class = literals
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(index, _)| like_index.is_none_or(|like| *index != like && *index != like + 1))
        .filter_map(|(_, literal)| literal_text(literal))
        .find_map(|value| numeric_class_tag(&value));

    if let Some(kind) = explicit_class {
        output.kind = kind;
    } else if like_index.is_none() {
        match &output.kind {
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown => {}
            ValueKindFact::Character | ValueKindFact::String => {
                output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
                output.storage = StorageFact::Dense;
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GPUARRAY-INPUT",
                    "gpuArray requires a numeric, logical, character, or string input",
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
        }
    }

    if let Some(shape) = gpu_array_shape(request) {
        output.shape = shape;
        output.storage = StorageFact::Dense;
    }
    output.residency = ResidencyFact::Device { provider: None };
    finish_fixed(entry, request, output, diagnostics)
}

fn gpu_array_shape(request: &CallRequest) -> Option<ShapeFact> {
    let literals = &request.literals.literal_args;
    if literals.len() <= 1 {
        return None;
    }
    if let Some(dimensions) = request.literals.numeric_vector_at(1) {
        return Some(ShapeFact::from(dimensions));
    }

    let mut dimensions = Vec::new();
    for index in 1..request.arguments.len() {
        let literal = literals.get(index).unwrap_or(&LiteralValue::Unknown);
        if literal_text(literal).is_some() {
            break;
        }
        let dimension = LiteralContext::numeric_from_literal(literal).and_then(|value| {
            (value.is_finite() && value >= 0.0 && value.fract() == 0.0).then_some(value as usize)
        });
        let scalar_numeric = matches!(
            request.arguments[index].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) && request.arguments[index].is_scalar();
        if dimension.is_none() && !scalar_numeric {
            break;
        }
        dimensions.push(dimension);
    }
    (!dimensions.is_empty()).then(|| ShapeFact::from(dimensions))
}

fn numeric_class_tag(value: &str) -> Option<ValueKindFact> {
    let kind = match value.to_ascii_lowercase().as_str() {
        "double" => numeric_kind(NumericClass::Double, NumericDomain::Real),
        "single" | "float32" => numeric_kind(NumericClass::Single, NumericDomain::Real),
        "logical" | "bool" | "boolean" => ValueKindFact::Logical,
        "int8" => numeric_kind(NumericClass::Int8, NumericDomain::Real),
        "int16" => numeric_kind(NumericClass::Int16, NumericDomain::Real),
        "int32" | "int" => numeric_kind(NumericClass::Int32, NumericDomain::Real),
        "int64" => numeric_kind(NumericClass::Int64, NumericDomain::Real),
        "uint8" => numeric_kind(NumericClass::UInt8, NumericDomain::Real),
        "uint16" => numeric_kind(NumericClass::UInt16, NumericDomain::Real),
        "uint32" => numeric_kind(NumericClass::UInt32, NumericDomain::Real),
        "uint64" => numeric_kind(NumericClass::UInt64, NumericDomain::Real),
        _ => return None,
    };
    Some(kind)
}
