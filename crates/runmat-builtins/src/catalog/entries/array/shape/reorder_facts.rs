//! Conservative static facts for value-preserving array reorderings.
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum Reorder {
    Flip,
    FlipLeftRight,
    FlipUpDown,
    Rotate90,
    CircularShift,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: Reorder,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let valid_arity = match operation {
        Reorder::FlipLeftRight | Reorder::FlipUpDown => request.arguments.len() == 1,
        Reorder::Flip | Reorder::Rotate90 => (1..=2).contains(&request.arguments.len()),
        Reorder::CircularShift => (2..=3).contains(&request.arguments.len()),
    };
    if !valid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-REORDER-ARITY",
            "invalid array reordering argument count",
            0,
        ));
    }
    let Some(source) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let cell_allowed = matches!(operation, Reorder::FlipUpDown);
    let supported = source.storage != StorageFact::Sparse
        && (matches!(
            source.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::String
                | ValueKindFact::Struct(_)
        ) || cell_allowed && matches!(source.kind, ValueKindFact::Cell(_)));
    if !supported {
        diagnostics.push(argument_error(
            "RM-CATALOG-REORDER-INPUT",
            "unsupported array reordering input",
            0,
        ));
    }
    for (index, control) in request.arguments.iter().enumerate().skip(1) {
        let class_supported = match operation {
            Reorder::Flip | Reorder::Rotate90 => matches!(
                control.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
                    | ValueKindFact::Logical
                    | ValueKindFact::Character
                    | ValueKindFact::String
            ),
            Reorder::CircularShift => matches!(
                control.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
                    | ValueKindFact::Logical
            ),
            Reorder::FlipLeftRight | Reorder::FlipUpDown => false,
        };
        let resident_control = matches!(
            control.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        );
        let scalar_required =
            matches!(operation, Reorder::Rotate90) && control.shape.is_definitely_nonscalar();
        let vector_required = match operation {
            Reorder::Rotate90 => false,
            Reorder::CircularShift => control.shape.element_count() != Some(0),
            _ => true,
        };
        let matrix_control = vector_required
            && control.shape.known_dims().is_some_and(|dims| {
                dims.into_iter()
                    .filter(|extent| extent.is_some_and(|extent| extent > 1))
                    .count()
                    > 1
            });
        let invalid_literal = request
            .literals
            .literal_args
            .get(index)
            .is_some_and(|literal| {
                invalid_control_literal(
                    operation,
                    index,
                    literal,
                    control.storage == StorageFact::Dense,
                )
            });
        if !class_supported
            || resident_control
            || scalar_required
            || matrix_control
            || invalid_literal
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-REORDER-CONTROL",
                "unsupported array reordering control class, shape or residency",
                index,
            ));
        }
    }
    let mut output = source.clone();
    // The runtime constructs a numeric tensor for a scalar Bool input.
    if source.storage == StorageFact::Scalar && source.kind == ValueKindFact::Logical {
        output.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
    }
    if matches!(operation, Reorder::Rotate90) {
        let turns = if request.arguments.len() == 1 {
            Some(1)
        } else {
            request
                .literals
                .literal_args
                .get(1)
                .and_then(rotation_turns)
        };
        if let Some(mut dims) = source.shape.known_dims() {
            if turns == Some(0) {
                // The runtime returns its original Tensor before rank padding.
                output.shape = source.shape.clone();
            } else if turns.is_none() && dims.len() < 2 {
                // An unknown count may be zero (identity) or a nonzero turn.
                output.shape = ShapeFact::Unknown;
            } else {
                dims.resize(dims.len().max(2), Some(1));
                match turns {
                    Some(turns) if turns % 2 == 1 => dims.swap(0, 1),
                    None => {
                        dims[0] = None;
                        dims[1] = None;
                    }
                    _ => {}
                }
                output.shape = ShapeFact::from(dims);
            }
        }
    }
    // A reorder may allocate or restore an automatic device value to host.
    // No alias, contiguous layout, or owning provider is statically promised.
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    output.contiguity = ContiguityFact::Unknown;
    output.layout = LayoutFact::Unknown;
    if matches!(
        source.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Unknown
    ) {
        output.residency = ResidencyFact::Unknown;
        output.storage = StorageFact::Unknown;
    }
    if let ValueKindFact::Struct(structure) = &mut output.kind {
        structure.elements.clear();
        structure.elements_complete = false;
    }
    if let ValueKindFact::Cell(cell) = &mut output.kind {
        cell.elements.clear();
        cell.elements_complete = false;
    }
    if !diagnostics.is_empty() {
        output.shape = ShapeFact::Unknown;
        output.certainty = CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation);
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn rotation_turns(literal: &LiteralValue) -> Option<usize> {
    if let LiteralValue::Vector(values) = literal {
        return (values.len() == 1)
            .then(|| rotation_turns(&values[0]))
            .flatten();
    }
    if let LiteralValue::Matrix(rows) = literal {
        return (rows.len() == 1 && rows[0].len() == 1)
            .then(|| rotation_turns(&rows[0][0]))
            .flatten();
    }
    let direction = match literal {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text.trim().to_ascii_lowercase()),
        _ => None,
    };
    match direction.as_deref() {
        Some("clockwise" | "cw") => return Some(3),
        Some("counterclockwise" | "anticlockwise" | "ccw" | "acw") => return Some(1),
        _ => {}
    }
    if let LiteralValue::Bool(value) = literal {
        return Some(usize::from(*value));
    }
    if let Some(exact) = literal.integer_value_i128() {
        return Some(exact.rem_euclid(4) as usize);
    }
    let number = LiteralContext::numeric_from_literal(literal)?;
    if !number.is_finite()
        || (number.round() - number).abs() > f64::EPSILON
        || number < i64::MIN as f64
        || number >= i64::MAX as f64
    {
        return None;
    }
    Some((number as i64).rem_euclid(4) as usize)
}

fn invalid_control_literal(
    operation: Reorder,
    index: usize,
    literal: &LiteralValue,
    vector_item: bool,
) -> bool {
    if matches!(literal, LiteralValue::Unknown) {
        return false;
    }
    if matches!(operation, Reorder::Rotate90) {
        return rotation_turns(literal).is_none();
    }
    if let LiteralValue::Vector(values) = literal {
        return (matches!(operation, Reorder::Flip) && values.is_empty())
            || values
                .iter()
                .any(|value| invalid_control_literal(operation, index, value, true));
    }
    if matches!(literal, LiteralValue::Matrix(_)) {
        return true;
    }
    if matches!(operation, Reorder::Flip) {
        if let LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) = literal
        {
            return !matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "horizontal"
                    | "left-right"
                    | "leftright"
                    | "lr"
                    | "right-left"
                    | "righthoriz"
                    | "vertical"
                    | "up-down"
                    | "updown"
                    | "ud"
                    | "down-up"
                    | "both"
            );
        }
    }
    if let LiteralValue::Bool(value) = literal {
        return (matches!(operation, Reorder::Flip) && (!vector_item || !*value))
            || (index == 2 && !*value);
    }
    if let Some(exact) = literal.integer_value_i128() {
        return if matches!(operation, Reorder::Flip) || index == 2 {
            exact < 1 || usize::try_from(exact).is_err()
        } else {
            isize::try_from(exact).is_err()
        };
    }
    let Some(number) = LiteralContext::numeric_from_literal(literal) else {
        return true;
    };
    if matches!(operation, Reorder::Flip) || index == 2 {
        if matches!(operation, Reorder::Flip) {
            if !vector_item {
                return runmat_types::dimension::decode_scalar_dimension(number, false).is_err();
            }
            return !number.is_finite()
                || number.fract() != 0.0
                || number < 1.0
                || number >= usize::MAX as f64;
        }
        return runmat_types::dimension::decode_shape_dimension(number)
            .map_or(true, |dimension| dimension < 1);
    }
    !number.is_finite()
        || (number.round() - number).abs() > f64::EPSILON
        || number < isize::MIN as f64
        || number >= isize::MAX as f64 + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::array::shape::{
        CIRCSHIFT_CATALOG_ENTRY, FLIPLR_CATALOG_ENTRY, FLIPUD_CATALOG_ENTRY, FLIP_CATALOG_ENTRY,
        ROT90_CATALOG_ENTRY,
    };

    fn numeric(shape: &[usize]) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
            StorageFact::Dense,
        )
    }

    fn call(
        entry: &BuiltinCatalogEntry,
        data: ValueFact,
        controls: Vec<LiteralValue>,
    ) -> CallInference {
        let mut arguments = vec![data];
        arguments.extend(
            controls
                .iter()
                .map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)),
        );
        let mut literals = vec![LiteralValue::Unknown];
        literals.extend(controls);
        crate::infer_catalog_call(
            entry,
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
        )
    }

    #[test]
    fn reorder_facts_preserve_native_class_and_only_proven_geometry() {
        let source = numeric(&[2, 3, 4]);
        for (entry, controls) in [
            (&FLIP_CATALOG_ENTRY, vec![LiteralValue::Number(2.0)]),
            (&FLIPLR_CATALOG_ENTRY, vec![]),
            (&FLIPUD_CATALOG_ENTRY, vec![]),
            (&CIRCSHIFT_CATALOG_ENTRY, vec![LiteralValue::Number(-1.0)]),
        ] {
            let result = call(entry, source.clone(), controls);
            assert!(
                result.diagnostics.is_empty(),
                "{}: {:?}",
                entry.identity.name,
                result.diagnostics
            );
            assert_eq!(result.outputs[0].numeric(), source.numeric());
            assert_eq!(result.outputs[0].shape, source.shape);
        }
        let rotated = call(&ROT90_CATALOG_ENTRY, source.clone(), vec![]);
        assert_eq!(
            rotated.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(2), Some(4)])
        );
        let unknown_turn = call(
            &ROT90_CATALOG_ENTRY,
            source.clone(),
            vec![LiteralValue::Unknown],
        );
        assert_eq!(
            unknown_turn.outputs[0].shape,
            ShapeFact::from(vec![None, None, Some(4)])
        );
        for alias in ["cw", "ccw", "acw", "anticlockwise"] {
            let result = call(
                &ROT90_CATALOG_ENTRY,
                source.clone(),
                vec![LiteralValue::String(alias.into())],
            );
            assert!(result.diagnostics.is_empty(), "{alias}");
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(3), Some(2), Some(4)])
            );
        }
        let boundary = call(
            &ROT90_CATALOG_ENTRY,
            source.clone(),
            vec![LiteralValue::Number(i64::MAX as f64)],
        );
        assert!(!boundary.diagnostics.is_empty());
        assert_eq!(boundary.outputs[0].shape, ShapeFact::Unknown);
        let rank_one = numeric(&[4]);
        let zero_turn = call(
            &ROT90_CATALOG_ENTRY,
            rank_one.clone(),
            vec![LiteralValue::Number(0.0)],
        );
        assert_eq!(zero_turn.outputs[0].shape, rank_one.shape);
        let unknown_rank_one = call(&ROT90_CATALOG_ENTRY, rank_one, vec![LiteralValue::Unknown]);
        assert_eq!(unknown_rank_one.outputs[0].shape, ShapeFact::Unknown);
        let bad_shift = call(
            &CIRCSHIFT_CATALOG_ENTRY,
            source.clone(),
            vec![LiteralValue::Number(f64::INFINITY)],
        );
        assert!(!bad_shift.diagnostics.is_empty());
        assert_eq!(bad_shift.outputs[0].shape, ShapeFact::Unknown);
        for entry in [&FLIP_CATALOG_ENTRY, &CIRCSHIFT_CATALOG_ENTRY] {
            let request = CallRequest {
                arguments: vec![source.clone(), numeric(&[2, 2])],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            };
            let result = crate::infer_catalog_call(entry, &request);
            assert!(!result.diagnostics.is_empty(), "{}", entry.identity.name);
            assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
        }
        let mut device = source;
        device.residency = ResidencyFact::Device {
            provider: Some("fixture".into()),
        };
        let result = call(&FLIP_CATALOG_ENTRY, device, vec![]);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
        assert_eq!(result.outputs[0].layout, LayoutFact::Unknown);
    }
}
