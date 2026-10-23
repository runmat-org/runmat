use super::super::facts;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
use std::collections::BTreeMap;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            crate::TABLE_ERROR_INVALID_ARGUMENT.code,
            "dictionary requires complete key/value pairs",
            request.arguments.len() - 1,
        ));
    }
    for index in (0..request.arguments.len().saturating_sub(1)).step_by(2) {
        let keys = &request.arguments[index];
        let values = &request.arguments[index + 1];
        let key_shape = facts::dictionary_shape(keys);
        let value_shape = facts::dictionary_shape(values);
        if value_shape.element_count() != Some(1) {
            if let (Some(k), Some(v)) = (key_shape.known_dims(), value_shape.known_dims()) {
                if k.len() == v.len()
                    && k.iter()
                        .zip(v)
                        .any(|(a, b)| a.zip(b).is_some_and(|(a, b)| a != b))
                {
                    diagnostics.push(argument_error(
                        crate::TABLE_ERROR_INVALID_ARGUMENT.code,
                        "keys and non-scalar values must have equal shape",
                        index + 1,
                    ));
                }
            }
        }
    }
    let length = request.arguments.is_empty().then_some(0);
    let key = request
        .arguments
        .first()
        .map(facts::scalar_element)
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let value = request
        .arguments
        .get(1)
        .map(facts::scalar_element)
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    finish_fixed(
        entry,
        request,
        facts::object(
            standard::DICTIONARY,
            BTreeMap::from([
                ("Keys".into(), facts::cells(key, length)),
                ("Values".into(), facts::cells(value, length)),
            ]),
            true,
        ),
        diagnostics,
    )
}
