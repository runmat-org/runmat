use crate::catalog::inference::{argument_error, finish_fixed, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn map_fact() -> ValueFact {
    let mut properties = BTreeMap::new();
    properties.insert(
        "Count".into(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
    );
    for name in ["KeyType", "ValueType"] {
        properties.insert(
            name.into(),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), None]),
                StorageFact::Dense,
            ),
        );
    }
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("containers.Map").owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

fn receiver(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(value) = request.arguments.first() {
        let admissible = match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class == &StaticClassIdentity::new("containers.Map").owned()),
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-CONTAINERS-MAP-RECEIVER",
                "expected a containers.Map handle",
                0,
            ));
        }
    }
}

fn arity(request: &CallRequest, minimum: usize, maximum: usize) -> Vec<InferenceDiagnostic> {
    let count = request.arguments.len();
    if (minimum..=maximum).contains(&count) {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-CONTAINERS-MAP-ARITY",
            format!("expected {minimum} to {maximum} inputs"),
            0,
        )]
    }
}

fn cell_output(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count == 1 || count == 3 || count > 4 {
        diagnostics.push(argument_error(
            "RM-CONTAINERS-MAP-CONSTRUCTOR-ARITY",
            "expected an empty map, key/value pairs, or supported name/value options",
            0,
        ));
    }
    finish_fixed(entry, request, map_fact(), diagnostics)
}

pub(super) fn keys(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 1, 1);
    receiver(request, &mut diagnostics);
    finish_fixed(
        entry,
        request,
        cell_output(ShapeFact::from(vec![Some(1), None])),
        diagnostics,
    )
}

pub(super) fn values(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 1, 2);
    receiver(request, &mut diagnostics);
    if let Some(key_set) = request.arguments.get(1) {
        if !matches!(
            key_set.kind,
            ValueKindFact::Unknown | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CONTAINERS-MAP-KEY-SET",
                "selected values require a cell array of keys",
                1,
            ));
        }
    }
    let shape = request.arguments.get(1).map_or_else(
        || ShapeFact::from(vec![Some(1), None]),
        |key_set| key_set.shape.clone(),
    );
    finish_fixed(entry, request, cell_output(shape), diagnostics)
}

pub(super) fn is_key(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2, 2);
    receiver(request, &mut diagnostics);
    let Some(keys) = request.arguments.get(1) else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let shape = match &keys.kind {
        ValueKindFact::Cell(_) => {
            if keys.shape.element_count() == Some(1) {
                ShapeFact::Scalar
            } else {
                keys.shape.clone()
            }
        }
        ValueKindFact::Character => {
            let rows = keys
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten());
            match rows {
                Some(1) => ShapeFact::Scalar,
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CONTAINERS-MAP-SCALAR-KEY",
                        "character keys must be a single-row character vector",
                        1,
                    ));
                    ShapeFact::Unknown
                }
                None => ShapeFact::Unknown,
            }
        }
        ValueKindFact::String | ValueKindFact::Numeric(_) => match keys.shape.element_count() {
            Some(1) => ShapeFact::Scalar,
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-CONTAINERS-MAP-SCALAR-KEY",
                    "multiple keys must be supplied in a cell array",
                    1,
                ));
                ShapeFact::Unknown
            }
            None => ShapeFact::Unknown,
        },
        ValueKindFact::Unknown => ShapeFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                "RM-CONTAINERS-MAP-KEY-TYPE",
                "expected a character, string, numeric, or cell key",
                1,
            ));
            ShapeFact::Unknown
        }
    };
    if shape == ShapeFact::Unknown {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if shape == ShapeFact::Unknown {
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    };
    finish_fixed(
        entry,
        request,
        ValueFact::proven(ValueKindFact::Logical, shape, storage),
        diagnostics,
    )
}

pub(super) fn remove(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2, 2);
    receiver(request, &mut diagnostics);
    finish_fixed_outputs(entry, request, Vec::new(), diagnostics)
}

pub(super) fn subsref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2, 2);
    receiver(request, &mut diagnostics);
    if let Some(path) = request.arguments.get(1) {
        if !matches!(path.kind, ValueKindFact::Unknown | ValueKindFact::Struct(_)) {
            diagnostics.push(argument_error(
                "RM-CONTAINERS-MAP-SUBSCRIPT",
                "expected a standard subscript path",
                1,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::DynamicDispatch),
        diagnostics,
    )
}

pub(super) fn subsasgn(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 3, 3);
    receiver(request, &mut diagnostics);
    let output = request
        .arguments
        .first()
        .and_then(|value| match &value.kind {
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref()
                    == Some(&StaticClassIdentity::new("containers.Map").owned()) =>
            {
                Some(value.clone())
            }
            _ => None,
        });
    finish_fixed(
        entry,
        request,
        output.unwrap_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch)),
        diagnostics,
    )
}
