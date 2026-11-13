use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error("RM-DIAG-ARITY", "diag requires an input", 0));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }
    let input = &request.arguments[0];
    if !matches!(
        input.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
    ) {
        diagnostics.push(argument_error(
            "RM-DIAG-INPUT",
            "diag requires numeric, logical, or character input",
            0,
        ));
    }
    if input.storage == StorageFact::Sparse {
        // The present runtime does not implement sparse diag as a structural CSC operation.
        diagnostics.push(argument_error(
            "RM-DIAG-SPARSE",
            "sparse diag input is unsupported",
            0,
        ));
    }
    if matches!(input.residency, ResidencyFact::Remote { .. }) {
        diagnostics.push(argument_error(
            "RM-DIAG-REMOTE",
            "diag does not accept distributed input",
            0,
        ));
    }
    let mut offset = Some(0isize);
    let mut offset_seen = false;
    let mut size = None;
    let mut vector_option = false;
    let mut output_kind = input.kind.clone();
    let mut like = false;
    let mut like_prototype = None;
    let mut unknown_control = false;
    for (index, control) in request.arguments.iter().enumerate().skip(1) {
        if like_prototype == Some(index) {
            continue;
        }
        if !unknown_control
            && matches!(
                control.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-DIAG-CONTROL-RESIDENCY",
                "diag controls must be host values except a like prototype",
                index,
            ));
        }
        match request.literals.literal_args.get(index) {
            Some(literal) if keyword(literal).is_some() => {
                match keyword(literal).as_deref().unwrap() {
                    "vector" => vector_option = true,
                    "like" => {
                        like = true;
                        like_prototype = Some(index + 1);
                    }
                    "logical" => output_kind = ValueKindFact::Logical,
                    "double" => {
                        output_kind = ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Double,
                            domain: input.numeric().map_or(NumericDomain::Real, |n| n.domain),
                        })
                    }
                    name => {
                        if let Some(class) = NumericClass::from_class_name(name) {
                            if class.integer_class().is_some() {
                                output_kind = ValueKindFact::Numeric(NumericFact {
                                    class,
                                    domain: input
                                        .numeric()
                                        .map_or(NumericDomain::Real, |n| n.domain),
                                });
                            } else {
                                diagnostics.push(argument_error(
                                    "RM-DIAG-OPTION",
                                    "unknown diag class option",
                                    index,
                                ));
                            }
                        } else {
                            diagnostics.push(argument_error(
                                "RM-DIAG-OPTION",
                                "unknown diag option",
                                index,
                            ));
                        }
                    }
                }
            }
            Some(literal) if !offset_seen && singleton_numeric_element(literal).is_some() => {
                offset_seen = true;
                let element = singleton_numeric_element(literal).expect("singleton checked");
                if matches!(element, LiteralValue::Unknown) {
                    unknown_control = true;
                    offset = None;
                } else {
                    offset = literal_offset(element);
                    if offset.is_none() {
                        diagnostics.push(argument_error(
                            "RM-DIAG-OFFSET",
                            "invalid diag offset",
                            index,
                        ));
                    }
                }
            }
            Some(LiteralValue::Vector(values)) if size.is_none() => {
                size = literal_size(values);
                if size.is_none() {
                    diagnostics.push(argument_error(
                        "RM-DIAG-SIZE",
                        "invalid diag size vector",
                        index,
                    ));
                }
            }
            Some(LiteralValue::Matrix(rows)) if size.is_none() && rows.len() == 1 => {
                size = literal_size(&rows[0]);
                if size.is_none() {
                    diagnostics.push(argument_error(
                        "RM-DIAG-SIZE",
                        "invalid diag size vector",
                        index,
                    ));
                }
            }
            Some(LiteralValue::Unknown) | None => {
                unknown_control = true;
            }
            Some(literal) if !offset_seen => {
                offset_seen = true;
                offset = literal_offset(literal);
                if offset.is_none() {
                    diagnostics.push(argument_error(
                        "RM-DIAG-OFFSET",
                        "invalid diag offset",
                        index,
                    ));
                }
            }
            _ => diagnostics.push(argument_error(
                "RM-DIAG-ARGUMENT",
                "diag does not accept another control argument",
                index,
            )),
        }
    }
    if like_prototype.is_some_and(|index| index >= request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-DIAG-LIKE",
            "diag like requires a prototype",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = if like || unknown_control {
        ValueKindFact::Unknown
    } else {
        output_kind
    };
    let dims = input.shape.known_dims();
    if let Some(dims) = &dims {
        if dims.len() > 2
            && dims[2..]
                .iter()
                .any(|dimension| dimension.is_some_and(|dimension| dimension != 1))
        {
            diagnostics.push(argument_error(
                "RM-DIAG-RANK",
                "diag accepts at most two non-singleton dimensions",
                0,
            ));
        }
    }
    let vector_length = dims.as_ref().and_then(|dims| {
        if dims.len() == 1 {
            return dims[0];
        }
        let rows = dims.first().copied().flatten();
        let cols = dims.get(1).copied().flatten();
        match (rows, cols) {
            (Some(1), Some(cols)) => Some(cols),
            (Some(rows), Some(1)) => Some(rows),
            _ => None,
        }
    });
    if vector_option {
        if let Some(length) = vector_length {
            output.shape = ShapeFact::from(vec![Some(length), Some(1)]);
        } else if input.shape.element_count().is_some() {
            diagnostics.push(argument_error(
                "RM-DIAG-VECTOR",
                "diag vector mode requires vector input",
                0,
            ));
        }
    } else if let Some((rows, cols)) = size {
        if vector_length.is_some() {
            output.shape = ShapeFact::from(vec![Some(rows), Some(cols)]);
        } else if input.shape.element_count().is_some() {
            diagnostics.push(argument_error(
                "RM-DIAG-SIZE",
                "diag size override requires vector input",
                0,
            ));
        }
    } else if let (Some(length), Some(offset)) = (vector_length, offset) {
        if let Some(side) = length.checked_add(offset.unsigned_abs()) {
            output.shape = ShapeFact::from(vec![Some(side), Some(side)]);
        } else {
            diagnostics.push(argument_error(
                "RM-DIAG-SIZE",
                "diag output size overflows",
                0,
            ));
        }
    } else if let Some(dims) = dims {
        if dims.len() >= 2 {
            let rows = dims[0];
            let cols = dims[1];
            let length = match (rows, cols, offset) {
                (Some(rows), Some(cols), Some(offset)) if offset >= 0 => {
                    Some(rows.min(cols.saturating_sub(offset as usize)))
                }
                (Some(rows), Some(cols), Some(offset)) => {
                    Some(rows.saturating_sub(offset.unsigned_abs()).min(cols))
                }
                _ => None,
            };
            output.shape = ShapeFact::from(vec![length, Some(1)]);
        }
    }
    output.storage = if like || unknown_control || input.residency != ResidencyFact::Host {
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    };
    output.layout = if output.storage == StorageFact::Dense {
        LayoutFact::ColumnMajor
    } else {
        LayoutFact::Unknown
    };
    output.residency = if like || unknown_control || input.residency != ResidencyFact::Host {
        ResidencyFact::Unknown
    } else {
        ResidencyFact::Host
    };
    if unknown_control {
        output.shape = ShapeFact::Unknown;
    }
    if !diagnostics.is_empty() {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
        output.shape = ShapeFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn keyword(value: &LiteralValue) -> Option<String> {
    match value {
        LiteralValue::Character(value)
        | LiteralValue::String(value)
        | LiteralValue::Keyword(value) => Some(value.to_ascii_lowercase()),
        _ => None,
    }
}

fn singleton_numeric_element(value: &LiteralValue) -> Option<&LiteralValue> {
    match value {
        LiteralValue::Vector(values) if values.len() == 1 => values.first(),
        LiteralValue::Matrix(rows) if rows.len() == 1 && rows[0].len() == 1 => rows[0].first(),
        _ => None,
    }
}

fn literal_offset(value: &LiteralValue) -> Option<isize> {
    if let Some(integer) = value.integer_value_i128() {
        return isize::try_from(integer).ok();
    }
    let value = LiteralContext::numeric_from_literal(value)?;
    if !value.is_finite()
        || value.fract() != 0.0
        || value < isize::MIN as f64
        || value >= isize::MAX as f64
    {
        return None;
    }
    Some(value as isize)
}

fn literal_size(values: &[LiteralValue]) -> Option<(usize, usize)> {
    let mut dimensions = values.iter().map(|value| {
        if let Some(integer) = value.integer_value_i128() {
            return usize::try_from(integer).ok();
        }
        let value = LiteralContext::numeric_from_literal(value)?;
        if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value >= usize::MAX as f64 {
            return None;
        }
        Some(value as usize)
    });
    let rows = dimensions.next()??;
    let columns = dimensions.next().unwrap_or(Some(rows))?;
    dimensions.next().is_none().then_some((rows, columns))
}

#[cfg(test)]
mod tests;
