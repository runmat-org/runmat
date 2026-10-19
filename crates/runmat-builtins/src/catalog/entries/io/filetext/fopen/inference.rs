use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Form {
    Open,
    Query,
    List,
    Unknown,
    Invalid,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let form = classify(request);
    let outputs = match form {
        Form::Open => vec![double(), character_row(), character_row(), character_row()],
        Form::Query => vec![character_row(); 4],
        Form::List => vec![
            handle_column(),
            character_cell_column(),
            character_cell_column(),
            character_cell_column(),
        ],
        Form::Unknown | Form::Invalid => {
            vec![ValueFact::unknown(DynamicReason::DynamicDispatch); 4]
        }
    };
    let mut contract = CallContract::fixed(outputs);
    // The runtime pads requested slots beyond the four named results with
    // double zero; this is not a four-output maximum.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(double()));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if matches!(form, Form::Query | Form::List) {
        result.effects.0.remove(&EffectKind::FilesystemRead);
        result.effects.0.remove(&EffectKind::FilesystemWrite);
        result.effects.0.remove(&EffectKind::EnvironmentWrite);
        result
            .capabilities
            .0
            .remove(&CapabilityRequirement::Filesystem);
    }
    if form == Form::Invalid {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FOPEN-FORM",
            "fopen expects a scalar filename, file identifier, or gated all-list form",
            0,
        ));
    }
    if request.arguments.len() > 4
        || form == Form::Query && request.arguments.len() != 1
        || form == Form::List && request.arguments.len() > 2
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FOPEN-ARITY",
            "too many arguments for the selected fopen form",
            0,
        ));
    }
    for index in 1..request.arguments.len() {
        if invalid_text(&request.arguments[index]) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-FOPEN-CONTROL",
                "permission, machine format, and encoding must be scalar text",
                index,
            ));
        }
    }
    result
}

fn classify(request: &CallRequest) -> Form {
    let Some(first) = request.arguments.first() else {
        return Form::List;
    };
    match &first.kind {
        ValueKindFact::Numeric(fact)
            if fact.domain == NumericDomain::Real && !first.shape.is_definitely_nonscalar() =>
        {
            Form::Query
        }
        ValueKindFact::Numeric(_) => Form::Invalid,
        ValueKindFact::Character | ValueKindFact::String if !invalid_text(first) => {
            match request.literals.literal_string_at(0).as_deref() {
                Some(text) if text.eq_ignore_ascii_case("all") => Form::List,
                Some(_) => Form::Open,
                None => Form::Unknown,
            }
        }
        ValueKindFact::Unknown => Form::Unknown,
        _ => Form::Invalid,
    }
}

fn invalid_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|dim| dim != 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

fn handle_column() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    )
}

fn character_cell_column() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(character_row()),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    )
}
