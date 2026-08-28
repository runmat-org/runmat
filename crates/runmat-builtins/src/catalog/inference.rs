use super::{BuiltinCatalogEntry, BuiltinContractMaturity};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ExecutionFact,
    FutureStateFact, InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputListFact, ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub fn infer_catalog_call(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule.0 {
        "array.full" => infer_full(request, entry),
        "array.zeros" => infer_zeros(request, entry),
        "math.abs" => infer_abs(request, entry),
        "acceleration.gather" => infer_gather(request, entry),
        "aggregate.struct" => infer_struct_builtin(request, entry),
        "introspection.feval" => infer_feval(request, entry),
        "parallel.parpool" => infer_parallel_pool(request, entry, false),
        "parallel.gcp" => infer_parallel_pool(request, entry, true),
        "parallel.parfeval" | "parallel.parfeval-on-all" => infer_parallel_future(request, entry),
        "parallel.fetch-outputs" => infer_parallel_fetch(request, entry, false),
        "parallel.fetch-next" => infer_parallel_fetch(request, entry, true),
        "parallel.distributed"
        | "parallel.local-part"
        | "parallel.barrier"
        | "parallel.broadcast"
        | "parallel.send"
        | "parallel.send-receive"
        | "parallel.receive"
        | "parallel.probe"
        | "parallel.gplus"
        | "parallel.spmd-index"
        | "parallel.spmd-size" => infer_parallel_data(request, entry),
        _ => unavailable_rule(entry, request),
    }
}

fn infer_parallel_data(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = match entry.contract.inference_rule.0 {
        "parallel.barrier" | "parallel.send" => ValueFact::scalar(ValueKindFact::Void),
        "parallel.probe" => ValueFact::scalar(ValueKindFact::Logical),
        "parallel.spmd-index" | "parallel.spmd-size" => {
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
        "parallel.local-part" => match request.arguments.first().map(|fact| &fact.kind) {
            Some(ValueKindFact::Distributed(distributed)) => distributed.value.as_ref().clone(),
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        "parallel.broadcast" => request
            .arguments
            .get(1)
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        "parallel.send-receive" => request
            .arguments
            .get(2)
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        "parallel.gplus" => request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        "parallel.distributed" | "parallel.receive" => {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        _ => unreachable!("parallel data inference is dispatched by a closed rule set"),
    };
    finish_fixed(entry, request, output, Vec::new())
}

fn infer_parallel_pool(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    current_pool: bool,
) -> CallInference {
    let nocreate = current_pool
        && request
            .literals
            .literal_args
            .first()
            .and_then(literal_text)
            .is_some_and(|value| value.eq_ignore_ascii_case("nocreate"));
    let output = if nocreate {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Pool))
    };
    finish_fixed(entry, request, output, Vec::new())
}

fn infer_parallel_future(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let explicit_pool = matches!(
        request.arguments.first().map(|fact| &fact.kind),
        Some(ValueKindFact::Execution(ExecutionFact::Pool))
    );
    let callable_index = usize::from(explicit_pool);
    let output_count_index = callable_index + 1;
    let output = scheduled_callable_output(request, callable_index, output_count_index);
    let future = ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Future {
        output: Box::new(output),
        state: FutureStateFact::Unknown,
    }));
    finish_fixed(entry, request, future, Vec::new())
}

fn scheduled_callable_output(
    request: &CallRequest,
    callable_index: usize,
    output_count_index: usize,
) -> ValueFact {
    let Some(ValueKindFact::Callable(callable)) =
        request.arguments.get(callable_index).map(|fact| &fact.kind)
    else {
        return ValueFact::unknown(DynamicReason::UnresolvedCallable);
    };
    let Some(output_count) = request
        .literals
        .literal_args
        .get(output_count_index)
        .and_then(literal_output_count)
    else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if output_count == 0 {
        return ValueFact::scalar(ValueKindFact::Void);
    }
    let outputs = (0..output_count)
        .map(|index| {
            callable
                .outputs
                .get(index)
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
        })
        .collect::<Vec<_>>();
    if output_count == 1 {
        outputs
            .into_iter()
            .next()
            .expect("one requested output was materialized")
    } else {
        ValueFact::scalar(ValueKindFact::OutputList(OutputListFact {
            outputs,
            variadic: !callable.outputs_complete || callable.variadic_outputs,
        }))
    }
}

fn literal_output_count(literal: &LiteralValue) -> Option<usize> {
    let LiteralValue::Number(value) = literal else {
        return None;
    };
    (value.is_finite() && *value >= 0.0 && value.fract() == 0.0 && *value <= usize::MAX as f64)
        .then_some(*value as usize)
}

fn infer_parallel_fetch(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    include_index: bool,
) -> CallInference {
    let payload = match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Execution(ExecutionFact::Future { output, .. })) => {
            output.as_ref().clone()
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut outputs = Vec::new();
    if include_index {
        outputs.push(default_double_scalar());
    }
    match payload.kind {
        ValueKindFact::OutputList(OutputListFact {
            outputs: payload_outputs,
            ..
        }) => outputs.extend(payload_outputs),
        ValueKindFact::Void => {}
        _ => outputs.push(payload),
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    infer_call(&contract, request)
}

fn infer_feval(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut contract = match request.arguments.first().map(|argument| &argument.kind) {
        Some(ValueKindFact::Callable(callable)) => CallContract {
            outputs: callable.outputs.clone(),
            variadic_output: (callable.variadic_outputs || !callable.outputs_complete)
                .then(|| Box::new(ValueFact::unknown(DynamicReason::RuntimeValue))),
            maximum_outputs: (callable.outputs_complete && !callable.variadic_outputs)
                .then_some(callable.outputs.len()),
            effects: Default::default(),
            capabilities: Default::default(),
            dynamic_reason: (!callable.outputs_complete || callable.variadic_outputs)
                .then_some(DynamicReason::RuntimeValue),
        },
        Some(_) => CallContract::dynamic(DynamicReason::RuntimeValue),
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEVAL-ARITY",
                "feval requires a function target",
                0,
            ));
            CallContract::dynamic(DynamicReason::RuntimeValue)
        }
    };
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn infer_struct_builtin(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [] => struct_fact(BTreeMap::new(), true),
        [template] => match template.kind {
            ValueKindFact::Struct(_) => template.clone(),
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ if template.shape.element_count() == Some(0) => struct_fact(BTreeMap::new(), true),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRUCT-TEMPLATE",
                    "single-argument struct requires a struct or empty array template",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        arguments if arguments.len() % 2 == 0 => {
            let mut fields = BTreeMap::new();
            let mut complete = true;
            for index in (0..arguments.len()).step_by(2) {
                if let Some(name) = request
                    .literals
                    .literal_args
                    .get(index)
                    .and_then(literal_text)
                {
                    fields.insert(name, arguments[index + 1].clone());
                } else {
                    complete = false;
                }
            }
            struct_fact(fields, complete)
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRUCT-PAIRS",
                "struct requires complete field/value pairs",
                request.arguments.len().saturating_sub(1),
            ));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn struct_fact(fields: BTreeMap<String, ValueFact>, fields_complete: bool) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete,
    }))
}

fn infer_gather(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
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
