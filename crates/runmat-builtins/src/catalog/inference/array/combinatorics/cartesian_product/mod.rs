mod column;

use std::collections::BTreeMap;

use crate::BuiltinCatalogEntry;
use runmat_types::{
    standard, CallInference, CallRequest, ObjectFact, ShapeFact, StructFact, ValueFact,
    ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-COMBINATIONS-ARITY",
            "combinations requires at least one input",
            0,
        ));
    }

    let columns = request
        .arguments
        .iter()
        .map(column::infer)
        .collect::<Vec<_>>();
    let row_count = columns.iter().try_fold(1_usize, |product, column| {
        product.checked_mul(column.sequence_length?)
    });
    let column_shape = ShapeFact::from(vec![row_count, Some(1)]);
    let variables = columns
        .into_iter()
        .enumerate()
        .map(|(index, column)| {
            (
                format!("Var{}", index + 1),
                column.with_shape(column_shape.clone()),
            )
        })
        .collect();

    let output = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::TABLE.owned()),
        properties: BTreeMap::from([(
            "Variables".into(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(variables, true))),
        )]),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    finish_fixed(entry, request, output, diagnostics)
}
