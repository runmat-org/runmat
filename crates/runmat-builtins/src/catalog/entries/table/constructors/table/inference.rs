//! Named schema facts are retained without confusing object shape with rows.
use super::super::{facts, TableConstructorOption};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
use std::collections::BTreeMap;
fn sizes(request: &CallRequest, index: usize) -> Option<[usize; 2]> {
    let LiteralValue::Vector(values) = request.literals.literal_args.get(index)? else {
        return None;
    };
    if values.len() != 2 {
        return None;
    }
    let mut out = [0; 2];
    for (to, value) in out.iter_mut().zip(values) {
        *to = if let Some(integer) = value.integer_value_i128() {
            usize::try_from(integer).ok()?
        } else {
            let n = LiteralContext::numeric_from_literal(value)?;
            if !n.is_finite() || n < 0.0 || n.fract() != 0.0 || n >= usize::MAX as f64 {
                return None;
            }
            n as usize
        };
    }
    Some(out)
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let (mut variable_names, mut size, mut types, mut row_names) = (None, None, None, None);
    let mut columns = Vec::new();
    let mut index = 0;
    let mut known = true;
    while index < request.arguments.len() {
        if index + 1 < request.arguments.len() {
            if let Some(key) = facts::text(request, index).and_then(TableConstructorOption::parse) {
                match key {
                    TableConstructorOption::VariableNames => variable_names = Some(index + 1),
                    TableConstructorOption::RowNames => row_names = Some(index + 1),
                    TableConstructorOption::Size => size = Some(index + 1),
                    TableConstructorOption::VariableTypes => types = Some(index + 1),
                }
                index += 2;
                continue;
            }
            if facts::text(request, index).is_none()
                && matches!(
                    request.arguments[index].kind,
                    ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
                )
            {
                known = false;
            }
        }
        columns.push(facts::retained(&request.arguments[index]));
        index += 1;
    }
    let mut rows = columns.first().and_then(facts::row_count);
    if let Some(expected) = rows {
        for (position, column) in columns.iter().enumerate().skip(1) {
            if facts::row_count(column).is_some_and(|n| n != expected) {
                diagnostics.push(argument_error(
                    crate::TABLE_ERROR_INVALID_VARIABLE.code,
                    "table variables must have equal row counts",
                    position,
                ));
            }
        }
    }
    if size.is_some() || types.is_some() {
        if !columns.is_empty() || size.is_none() || types.is_none() {
            diagnostics.push(argument_error(
                crate::TABLE_ERROR_INVALID_ARGUMENT.code,
                "preallocation requires Size and VariableTypes without data variables",
                size.or(types).unwrap_or(0),
            ));
        }
        let geometry = size.and_then(|i| sizes(request, i));
        let class_names = types.and_then(|i| facts::names(request, i));
        columns.clear();
        known &= geometry.is_some() && class_names.is_some();
        if let Some([n, width]) = geometry {
            rows = Some(n);
            if let Some(names) = class_names {
                if names.len() != width {
                    diagnostics.push(argument_error(
                        crate::TABLE_ERROR_INVALID_ARGUMENT.code,
                        "VariableTypes count must match Size width",
                        types.unwrap_or(0),
                    ));
                }
                for name in names {
                    if let Some(class) = NumericClass::from_class_name(name.trim()) {
                        columns.push(ValueFact::proven(
                            ValueKindFact::Numeric(NumericFact {
                                class,
                                domain: NumericDomain::Real,
                            }),
                            ShapeFact::from(vec![Some(n), Some(1)]),
                            StorageFact::Dense,
                        ));
                    } else {
                        diagnostics.push(argument_error(
                            crate::TABLE_ERROR_INVALID_ARGUMENT.code,
                            "this preallocation type is not implemented",
                            types.unwrap_or(0),
                        ));
                        known = false;
                    }
                }
            }
        }
    }
    let names = variable_names
        .and_then(|i| facts::names(request, i))
        .or_else(|| {
            variable_names
                .is_none()
                .then(|| (1..=columns.len()).map(|i| format!("Var{i}")).collect())
        });
    if let Some(names) = &names {
        if known && names.len() != columns.len() {
            diagnostics.push(argument_error(
                crate::TABLE_ERROR_INVALID_VARIABLE.code,
                "VariableNames count must match the variable count",
                variable_names.unwrap_or(0),
            ));
        }
        let mut unique = std::collections::BTreeSet::new();
        if names
            .iter()
            .any(|name| name.is_empty() || !unique.insert(name))
        {
            diagnostics.push(argument_error(
                crate::TABLE_ERROR_INVALID_VARIABLE.code,
                "variable names must be nonempty and unique",
                variable_names.unwrap_or(0),
            ));
        }
    }
    if let Some(index) = row_names {
        if let (Some(rows), Some(names)) = (rows, facts::names(request, index)) {
            if rows != names.len() {
                diagnostics.push(argument_error(
                    crate::TABLE_ERROR_INVALID_VARIABLE.code,
                    "RowNames count must match table height",
                    index,
                ));
            }
        }
    }
    known &= names.is_some();
    let fields = if known {
        names.unwrap_or_default().into_iter().zip(columns).collect()
    } else {
        BTreeMap::new()
    };
    let variables = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, known)),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(
        entry,
        request,
        facts::object(
            standard::TABLE,
            BTreeMap::from([("__table_variables".into(), variables)]),
            false,
        ),
        diagnostics,
    )
}
