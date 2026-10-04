use crate::catalog::inference::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, PreferenceInferenceRule};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(in crate::catalog::entries::io::repl_fs::preferences) fn infer(
    rule: PreferenceInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    let valid_arity = match rule {
        PreferenceInferenceRule::Read => count <= 3,
        PreferenceInferenceRule::Exists => (1..=2).contains(&count),
        PreferenceInferenceRule::Set => count == 3,
    };
    if !valid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-PREFERENCE-ARITY",
            "invalid preference argument count",
            count.saturating_sub(1),
        ));
    }
    if let Some(group) = request.arguments.first() {
        if !possible_text_scalar(group) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PREFERENCE-GROUP",
                "preference group must be a text scalar or character row",
                0,
            ));
        }
    }
    let names_shape = request.arguments.get(1).and_then(|names| {
        let shape = possible_names(names);
        if shape.is_none() {
            diagnostics.push(argument_error(
                "RM-CATALOG-PREFERENCE-NAME",
                "preference names must be text or a supported text container",
                1,
            ));
        }
        shape
    });
    if matches!(
        rule,
        PreferenceInferenceRule::Set | PreferenceInferenceRule::Read
    ) && count == 3
    {
        if let (Some(Some(name_shape)), Some(value)) =
            (names_shape.as_ref(), request.arguments.get(2))
        {
            if !matches!(value.kind, ValueKindFact::Cell(_) | ValueKindFact::Unknown)
                || (known_shape(name_shape).is_some()
                    && known_shape(&value.shape).is_some()
                    && known_shape(name_shape) != known_shape(&value.shape))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PREFERENCE-VALUE-SHAPE",
                    "multiple names require an equally shaped cell array of values",
                    2,
                ));
            }
        }
    }
    if matches!(rule, PreferenceInferenceRule::Read)
        && count == 3
        && request
            .arguments
            .get(2)
            .is_some_and(|value| matches!(value.residency, ResidencyFact::Device { .. }))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GETPREF-RESIDENT-DEFAULT",
            "getpref does not accept a provider-resident default",
            2,
        ));
    }
    let output = match rule {
        PreferenceInferenceRule::Read if count <= 1 => ValueFact::scalar(ValueKindFact::Struct(
            StructFact::scalar(BTreeMap::new(), false),
        )),
        PreferenceInferenceRule::Read => match names_shape.flatten() {
            Some(shape) => ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(ValueFact::unknown(DynamicReason::DynamicDispatch)),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                shape,
                StorageFact::Dense,
            ),
            None => ValueFact::unknown(DynamicReason::DynamicDispatch),
        },
        PreferenceInferenceRule::Exists => ValueFact::proven(
            ValueKindFact::Logical,
            if request
                .arguments
                .get(1)
                .is_some_and(|value| matches!(value.kind, ValueKindFact::Unknown))
            {
                ShapeFact::Unknown
            } else {
                names_shape.flatten().unwrap_or(ShapeFact::Scalar)
            },
            StorageFact::Dense,
        ),
        PreferenceInferenceRule::Set => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn possible_text_scalar(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => {
            crate::catalog::inference::possible_character_row(&value.shape)
        }
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

/// `None` is invalid; `Some(None)` is a scalar or not-yet-known form.
fn possible_names(value: &ValueFact) -> Option<Option<ShapeFact>> {
    match &value.kind {
        ValueKindFact::String => {
            if value.shape.element_count() == Some(1) {
                Some(None)
            } else {
                Some(Some(value.shape.clone()))
            }
        }
        ValueKindFact::Character if possible_text_scalar(value) => Some(None),
        ValueKindFact::Cell(cell)
            if possible_cell_name(&cell.element)
                && cell.elements.iter().all(possible_cell_name) =>
        {
            Some(Some(value.shape.clone()))
        }
        ValueKindFact::Unknown => Some(None),
        _ => None,
    }
}

fn possible_cell_name(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => possible_text_scalar(value),
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn known_shape(shape: &ShapeFact) -> Option<Vec<usize>> {
    shape.known_dims()?.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_string_and_cell_name_shapes_are_preserved() {
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        assert_eq!(possible_names(&strings), Some(Some(strings.shape.clone())));

        let row = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(5)]),
            StorageFact::Dense,
        );
        let names = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(row.clone()),
                elements: vec![row.clone(), row],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        assert_eq!(possible_names(&names), Some(Some(names.shape.clone())));
    }

    #[test]
    fn known_invalid_controls_do_not_require_runtime_storage() {
        let non_text = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        assert!(!possible_text_scalar(&non_text));
        assert_eq!(possible_names(&non_text), None);

        let matrix = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(4)]),
            StorageFact::Dense,
        );
        assert!(!possible_text_scalar(&matrix));
        assert_eq!(possible_names(&matrix), None);
    }
}
