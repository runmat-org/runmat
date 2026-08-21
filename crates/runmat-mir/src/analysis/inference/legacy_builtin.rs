//! Bounded bridge for builtin identities that have not reached the canonical
//! catalog cohorts yet.
//!
//! The identity lookup is the inventory: only a registered builtin's own
//! legacy resolver/declared return participates. No semantic name category is
//! reinterpreted here. C00-C07 replace these identity-owned legacy contracts
//! with canonical catalog entries, and R29 removes this module.

use std::collections::BTreeMap;

use runmat_builtins::Type;
use runmat_types::{
    infer_call, infer_concatenate, infer_reduction, infer_repmat, infer_reshape, permute_shape,
    CallContract, CallInference, CallRequest, CellFact, DimensionFact, DynamicReason,
    FactInference, FactJoin, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputListFact, QualifiedName, ShapeFact, StorageFact, StructFact, SymbolName,
    ValueFact, ValueKindFact,
};

pub(crate) fn infer_legacy_builtin(name: &str, request: &CallRequest) -> Option<CallInference> {
    let function = runmat_builtins::builtin_functions()
        .into_iter()
        .find(|function| function.name.eq_ignore_ascii_case(name))?;

    if name.eq_ignore_ascii_case("linspace") {
        return Some(infer_call(
            &CallContract::fixed(vec![linspace_fact(request)]),
            request,
        ));
    }
    if name.eq_ignore_ascii_case("meshgrid") {
        return Some(infer_call(
            &repeatable_contract(meshgrid_fact(request)),
            request,
        ));
    }
    if name.eq_ignore_ascii_case("reshape") {
        return request.arguments.first().map(|source| {
            let (dimensions, mut diagnostics) = dimensions_from(request, 1, true);
            let mut inference = infer_reshape(source, dimensions);
            inference.diagnostics.append(&mut diagnostics);
            call_from_fact(inference, request)
        });
    }
    if name.eq_ignore_ascii_case("repmat") {
        return request.arguments.first().map(|source| {
            let (dimensions, diagnostics) = dimensions_from(request, 1, false);
            let mut inference = infer_repmat(source, &dimensions);
            inference.diagnostics.extend(diagnostics);
            call_from_fact(inference, request)
        });
    }
    if name.eq_ignore_ascii_case("permute") {
        return request.arguments.first().map(|source| {
            let order = request
                .literals
                .numeric_vector_at(1)
                .unwrap_or_default()
                .into_iter()
                .collect::<Option<Vec<_>>>();
            let inference = match order {
                Some(order) => match permute_shape(&source.shape, &order) {
                    Ok(shape) => FactInference::exact(with_shape(source, shape)),
                    Err(diagnostic) => FactInference {
                        fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
                        diagnostics: vec![diagnostic],
                    },
                },
                None => FactInference::exact(with_shape(source, ShapeFact::Unknown)),
            };
            call_from_fact(inference, request)
        });
    }
    if name.eq_ignore_ascii_case("horzcat") || name.eq_ignore_ascii_case("vertcat") {
        let dimension = if name.eq_ignore_ascii_case("horzcat") {
            2
        } else {
            1
        };
        return Some(call_from_fact(
            infer_concatenate(dimension, &request.arguments),
            request,
        ));
    }
    if name.eq_ignore_ascii_case("cat") {
        let dimension = request
            .literals
            .numeric_at(0)
            .and_then(nonnegative_integer)
            .unwrap_or(0);
        return Some(call_from_fact(
            infer_concatenate(dimension, request.arguments.get(1..).unwrap_or(&[])),
            request,
        ));
    }
    if name.eq_ignore_ascii_case("dot") {
        let mut inference = FactInference::exact(numeric_array(ShapeFact::Scalar));
        if let [left, right, ..] = request.arguments.as_slice() {
            if left
                .shape
                .element_count()
                .zip(right.shape.element_count())
                .is_some_and(|(left, right)| left != right)
            {
                inference.diagnostics.push(InferenceDiagnostic::error(
                    "RM-TYPE-DOT",
                    "dot product operand element counts do not agree",
                ));
            }
        }
        return Some(call_from_fact(inference, request));
    }
    if matches!(
        name.to_ascii_lowercase().as_str(),
        "sum" | "prod" | "mean" | "median" | "min" | "max" | "all" | "any"
    ) {
        return request.arguments.first().map(|source| {
            let dimension = request.literals.numeric_at(1).and_then(nonnegative_integer);
            let mut inference = infer_reduction(source, dimension);
            if name.eq_ignore_ascii_case("all") || name.eq_ignore_ascii_case("any") {
                inference.fact.kind = ValueKindFact::Logical;
                inference.fact.storage = if inference.fact.shape.element_count() == Some(1) {
                    StorageFact::Scalar
                } else {
                    StorageFact::Dense
                };
                call_from_fact(inference, request)
            } else if name.eq_ignore_ascii_case("min") || name.eq_ignore_ascii_case("max") {
                let index = numeric_array(inference.fact.shape.clone());
                let mut call =
                    infer_call(&CallContract::fixed(vec![inference.fact, index]), request);
                call.diagnostics.append(&mut inference.diagnostics);
                call
            } else {
                call_from_fact(inference, request)
            }
        });
    }

    let arguments = request
        .arguments
        .iter()
        .map(fact_to_type)
        .collect::<Vec<_>>();
    let resolved = function.infer_return_type_with_context(&arguments, &request.literals);
    let contract = match resolved {
        Type::OutputList(outputs) => {
            CallContract::fixed(outputs.iter().map(type_to_fact).collect::<Vec<_>>())
        }
        output => conservative_legacy_contract(type_to_fact(&output)),
    };
    Some(infer_call(&contract, request))
}

fn call_from_fact(inference: FactInference, request: &CallRequest) -> CallInference {
    let mut call = infer_call(&CallContract::fixed(vec![inference.fact]), request);
    call.diagnostics.extend(inference.diagnostics);
    call
}

fn with_shape(source: &ValueFact, shape: ShapeFact) -> ValueFact {
    let mut output = source.clone();
    output.shape = shape;
    output
}

fn dimensions_from(
    request: &CallRequest,
    start: usize,
    reshape: bool,
) -> (Vec<DimensionFact>, Vec<InferenceDiagnostic>) {
    let literals = request.literals.literal_args.get(start..).unwrap_or(&[]);
    let flattened = match literals.first() {
        Some(runmat_types::LiteralValue::Vector(values)) if literals.len() == 1 => {
            values.as_slice()
        }
        _ => literals,
    };
    let mut inferred_dimensions = 0usize;
    let mut diagnostics = Vec::new();
    let dimensions = flattened
        .iter()
        .map(
            |literal| match runmat_types::LiteralContext::numeric_from_literal(literal) {
                Some(value) if reshape && value == -1.0 => {
                    inferred_dimensions += 1;
                    DimensionFact::Unknown
                }
                Some(value) if value.is_finite() && value >= 0.0 && value.fract() == 0.0 => {
                    DimensionFact::Known(value as usize)
                }
                Some(_) => {
                    diagnostics.push(InferenceDiagnostic::error(
                    if reshape { "RM-TYPE-RESHAPE" } else { "RM-TYPE-REPMAT" },
                    if reshape {
                        "reshape dimensions must be non-negative integers or one inferred dimension"
                    } else {
                        "repmat dimensions must be non-negative integers"
                    },
                ));
                    DimensionFact::Unknown
                }
                None => DimensionFact::Unknown,
            },
        )
        .collect();
    if reshape && inferred_dimensions > 1 {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-TYPE-RESHAPE",
            "reshape accepts at most one inferred dimension",
        ));
    }
    (dimensions, diagnostics)
}

fn repeatable_contract(output: ValueFact) -> CallContract {
    CallContract {
        outputs: vec![output.clone()],
        variadic_output: Some(Box::new(output)),
        maximum_outputs: None,
        effects: Default::default(),
        capabilities: Default::default(),
        dynamic_reason: Some(DynamicReason::UnsupportedRepresentation),
    }
}

fn conservative_legacy_contract(output: ValueFact) -> CallContract {
    CallContract {
        outputs: vec![output],
        variadic_output: Some(Box::new(ValueFact::unknown(
            DynamicReason::UnsupportedRepresentation,
        ))),
        maximum_outputs: None,
        effects: Default::default(),
        capabilities: Default::default(),
        dynamic_reason: Some(DynamicReason::UnsupportedRepresentation),
    }
}

fn linspace_fact(request: &CallRequest) -> ValueFact {
    let count = request
        .literals
        .numeric_at(2)
        .and_then(nonnegative_integer)
        .or_else(|| (request.arguments.len() == 2).then_some(100));
    numeric_array(ShapeFact::from(vec![Some(1), count]))
}

fn meshgrid_fact(request: &CallRequest) -> ValueFact {
    let axis_count = request.arguments.len().min(3);
    let x = request.arguments.first().and_then(vector_length);
    let y = request.arguments.get(1).and_then(vector_length).or(x);
    let z = request.arguments.get(2).and_then(vector_length);
    let dimensions = if axis_count >= 3 {
        vec![y, x, z]
    } else {
        vec![y, x]
    };
    let mut output = request
        .arguments
        .iter()
        .take(axis_count)
        .cloned()
        .reduce(|left, right| left.join(&right))
        .unwrap_or_else(|| numeric_array(ShapeFact::Unknown));
    output.shape = ShapeFact::from(dimensions);
    output.storage = StorageFact::Dense;
    output
}

fn vector_length(fact: &ValueFact) -> Option<usize> {
    let dimensions = fact.shape.known_dims()?;
    if dimensions.len() == 1
        || (dimensions.len() == 2
            && (dimensions.first() == Some(&Some(1)) || dimensions.get(1) == Some(&Some(1))))
    {
        dimensions
            .into_iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(dimension?))
    } else {
        None
    }
}

fn nonnegative_integer(value: f64) -> Option<usize> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0).then_some(value as usize)
}

fn numeric_array(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn fact_to_type(fact: &ValueFact) -> Type {
    let shape = fact.shape.known_dims();
    match &fact.kind {
        ValueKindFact::Numeric(numeric) if fact.is_scalar() => match numeric.class {
            NumericClass::Int8
            | NumericClass::Int16
            | NumericClass::Int32
            | NumericClass::Int64
            | NumericClass::UInt8
            | NumericClass::UInt16
            | NumericClass::UInt32
            | NumericClass::UInt64 => Type::Int,
            NumericClass::Single | NumericClass::Double => Type::Num,
        },
        ValueKindFact::Numeric(_) => Type::Tensor { shape },
        ValueKindFact::Logical if fact.is_scalar() => Type::Bool,
        ValueKindFact::Logical => Type::Logical { shape },
        ValueKindFact::String | ValueKindFact::Character => Type::String,
        ValueKindFact::Symbolic if fact.is_scalar() => Type::Symbolic,
        ValueKindFact::Symbolic => Type::SymbolicArray { shape },
        ValueKindFact::Cell(cell) => Type::Cell {
            element_type: Some(Box::new(fact_to_type(&cell.element))),
            length: fact.shape.element_count(),
        },
        ValueKindFact::Struct(structure) => Type::Struct {
            known_fields: structure
                .fields_complete
                .then(|| structure.fields.keys().cloned().collect()),
        },
        ValueKindFact::Object(object) => Type::Object {
            class_name: object
                .runtime_class
                .as_ref()
                .and_then(QualifiedName::display_name),
            shape,
        },
        ValueKindFact::Callable(callable) => Type::Function {
            params: callable.parameters.iter().map(fact_to_type).collect(),
            returns: Box::new(Type::OutputList(
                callable.outputs.iter().map(fact_to_type).collect(),
            )),
        },
        ValueKindFact::OutputList(outputs) => {
            Type::OutputList(outputs.outputs.iter().map(fact_to_type).collect())
        }
        ValueKindFact::Void => Type::Void,
        _ => Type::Unknown,
    }
}

fn type_to_fact(value: &Type) -> ValueFact {
    match value {
        Type::Int => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        })),
        Type::Num => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        Type::Bool => ValueFact::scalar(ValueKindFact::Logical),
        Type::Logical { shape } => ValueFact::proven(
            ValueKindFact::Logical,
            shape_fact(shape),
            StorageFact::Dense,
        ),
        Type::String => ValueFact::scalar(ValueKindFact::String),
        Type::Tensor { shape } => numeric_array(shape_fact(shape)),
        Type::Symbolic => ValueFact::scalar(ValueKindFact::Symbolic),
        Type::SymbolicArray { shape } => ValueFact::proven(
            ValueKindFact::Symbolic,
            shape_fact(shape),
            StorageFact::Dense,
        ),
        Type::Cell {
            element_type,
            length,
        } => {
            let element = element_type
                .as_deref()
                .map(type_to_fact)
                .unwrap_or_else(dynamic_value);
            ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(element),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                ShapeFact::from(vec![Some(1), *length]),
                StorageFact::Dense,
            )
        }
        Type::Function { params, returns } => {
            ValueFact::scalar(ValueKindFact::Callable(runmat_types::CallableFact {
                identity: None,
                parameters: params.iter().map(type_to_fact).collect(),
                parameters_complete: true,
                outputs: match returns.as_ref() {
                    Type::OutputList(outputs) => outputs.iter().map(type_to_fact).collect(),
                    output => vec![type_to_fact(output)],
                },
                outputs_complete: true,
                variadic_inputs: false,
                variadic_outputs: false,
                captures: Vec::new(),
                captures_complete: false,
            }))
        }
        Type::Void => ValueFact::scalar(ValueKindFact::Void),
        Type::Unknown => dynamic_value(),
        Type::Union(values) => values
            .iter()
            .map(type_to_fact)
            .reduce(|left, right| left.join(&right))
            .unwrap_or_else(dynamic_value),
        Type::Struct { known_fields } => ValueFact::scalar(ValueKindFact::Struct(StructFact {
            fields: known_fields
                .as_ref()
                .map(|fields| {
                    fields
                        .iter()
                        .cloned()
                        .map(|field| (field, dynamic_value()))
                        .collect::<BTreeMap<_, _>>()
                })
                .unwrap_or_default(),
            fields_complete: known_fields.is_some(),
        })),
        Type::Object { class_name, shape } => ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: class_name.as_ref().map(|name| {
                    QualifiedName(
                        name.split('.')
                            .map(|part| SymbolName(part.to_string()))
                            .collect(),
                    )
                }),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: None,
            }),
            shape_fact(shape),
            StorageFact::Opaque,
        ),
        Type::OutputList(outputs) => ValueFact::scalar(ValueKindFact::OutputList(OutputListFact {
            outputs: outputs.iter().map(type_to_fact).collect(),
            variadic: false,
        })),
    }
}

fn shape_fact(shape: &Option<Vec<Option<usize>>>) -> ShapeFact {
    shape.clone().map_or(ShapeFact::Unknown, ShapeFact::from)
}

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}
