use super::{BuiltinCatalogEntry, BuiltinContractMaturity};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub fn infer_catalog_call(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule.0 {
        "array.full" => infer_full(request, entry),
        "array.zeros" => infer_zeros(request, entry),
        "math.abs" => infer_abs(request, entry),
        _ => unavailable_rule(entry, request),
    }
}

fn infer_abs(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = request.arguments.first().cloned().unwrap_or_else(|| {
        diagnostics.push(argument_error(
            "RM-CATALOG-ABS-ARITY",
            "abs requires exactly one input",
            0,
        ));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ABS-ARITY",
            "abs accepts exactly one input",
            1,
        ));
    }
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain = NumericDomain::Real,
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
        }
        ValueKindFact::Symbolic | ValueKindFact::Unknown => {}
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ABS-INPUT",
                "abs requires numeric, logical, character, or symbolic input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_zeros(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let literals = &request.literals.literal_args;
    let like_index = literals.iter().position(
        |literal| matches!(literal, LiteralValue::String(value) | LiteralValue::Character(value) | LiteralValue::Keyword(value) if value.eq_ignore_ascii_case("like")),
    );
    let trailing_class = literals
        .last()
        .and_then(literal_text)
        .filter(|value| !value.eq_ignore_ascii_case("like"));
    let dimension_end = like_index.unwrap_or_else(|| {
        trailing_class
            .as_ref()
            .map_or(literals.len(), |_| literals.len().saturating_sub(1))
    });

    let mut output = like_index
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);

    if let Some(class) = trailing_class.as_deref() {
        match class.to_ascii_lowercase().as_str() {
            "double" => output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real),
            "single" => output.kind = numeric_kind(NumericClass::Single, NumericDomain::Real),
            "logical" => output.kind = ValueKindFact::Logical,
            "int8" => output.kind = numeric_kind(NumericClass::Int8, NumericDomain::Real),
            "int16" => output.kind = numeric_kind(NumericClass::Int16, NumericDomain::Real),
            "int32" => output.kind = numeric_kind(NumericClass::Int32, NumericDomain::Real),
            "int64" => output.kind = numeric_kind(NumericClass::Int64, NumericDomain::Real),
            "uint8" => output.kind = numeric_kind(NumericClass::UInt8, NumericDomain::Real),
            "uint16" => output.kind = numeric_kind(NumericClass::UInt16, NumericDomain::Real),
            "uint32" => output.kind = numeric_kind(NumericClass::UInt32, NumericDomain::Real),
            "uint64" => output.kind = numeric_kind(NumericClass::UInt64, NumericDomain::Real),
            "gpuarray" => {
                output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
                output.residency = ResidencyFact::Device { provider: None };
            }
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-ZEROS-CLASS",
                "zeros class specifier is not recognized",
                literals.len().saturating_sub(1),
            )),
        }
    }

    if dimension_end > 0 {
        output.shape = zeros_shape(request, dimension_end);
        output.storage = StorageFact::Dense;
    } else if like_index.is_none() {
        output = default_double_scalar();
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn zeros_shape(request: &CallRequest, dimension_end: usize) -> ShapeFact {
    if dimension_end == 1 {
        if let Some(dims) = request.literals.numeric_vector_at(0) {
            return ShapeFact::from(dims);
        }
        if let Some(size) = request.literals.numeric_dims().first().copied().flatten() {
            return ShapeFact::from(vec![Some(size), Some(size)]);
        }
        return match request
            .arguments
            .first()
            .and_then(|argument| argument.shape.rank())
        {
            Some(2) if !request.arguments[0].is_scalar() => {
                let rank = request.arguments[0]
                    .shape
                    .element_count()
                    .unwrap_or(2)
                    .max(2);
                ShapeFact::Ranked { rank }
            }
            _ => ShapeFact::Ranked { rank: 2 },
        };
    }
    ShapeFact::from(
        request
            .literals
            .numeric_dims()
            .into_iter()
            .take(dimension_end)
            .collect::<Vec<_>>(),
    )
}

fn literal_text(literal: &LiteralValue) -> Option<String> {
    match literal {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => Some(value.clone()),
        _ => None,
    }
}

fn numeric_kind(class: NumericClass, domain: NumericDomain) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact { class, domain })
}

fn default_double_scalar() -> ValueFact {
    ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real))
}

fn infer_full(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = request.arguments.first().cloned().unwrap_or_else(|| {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULL-ARITY",
            "full requires exactly one input",
            0,
        ));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULL-ARITY",
            "full accepts exactly one input",
            1,
        ));
    }
    match output.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Character => {
            if matches!(output.storage, StorageFact::Sparse) {
                output.storage = StorageFact::Dense;
            }
        }
        ValueKindFact::Unknown => {}
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FULL-INPUT",
                "full requires a numeric, logical, or character array",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn unavailable_rule(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::UnsupportedRepresentation);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    if matches!(entry.contract.maturity, BuiltinContractMaturity::Complete) {
        inference.diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-INFERENCE-RULE",
            format!(
                "complete builtin contract `{}` has no registered inference rule",
                entry.contract.inference_rule.0
            ),
        ));
    }
    inference
}

fn finish_fixed(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    output: ValueFact,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn argument_error(code: &str, message: &str, argument: usize) -> InferenceDiagnostic {
    let mut diagnostic = InferenceDiagnostic::error(code, message);
    diagnostic.argument = Some(argument);
    diagnostic
}
