use crate::catalog::inference::{argument_error, default_double_scalar, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy)]
enum Shift {
    RemoveLeading,
    Explicit(isize),
    Dynamic,
    Invalid,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHIFTDIM-ARITY",
            "shiftdim requires one or two inputs",
            0,
        ));
    }
    if request.arguments.len() == 2
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHIFTDIM-OUTPUTS",
            "the explicit-shift form has only one output",
            1,
        ));
    }
    let mut invalid_control_reported = false;
    let shift = if request.arguments.len() == 1 {
        Shift::RemoveLeading
    } else if let Some(control) = request.arguments.get(1) {
        if !control_supported(control) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SHIFTDIM-SHIFT",
                "shift must be a host real integer scalar",
                1,
            ));
            invalid_control_reported = true;
            Shift::Invalid
        } else {
            literal_shift(request.literals.literal_args.get(1))
        }
    } else {
        Shift::Invalid
    };
    if matches!(shift, Shift::Invalid) && request.arguments.len() == 2 && !invalid_control_reported
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHIFTDIM-SHIFT",
            "shift is outside the finite signed integer range",
            1,
        ));
    }
    let Some(source) = request.arguments.first() else {
        return finish_fixed_outputs(
            entry,
            request,
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            diagnostics,
        );
    };
    if !source_supported(source) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHIFTDIM-INPUT",
            "unsupported shiftdim input representation",
            0,
        ));
    }
    let (shape, _) = shifted_shape(&source.shape, shift);
    if source.storage == StorageFact::Sparse && proven_shape_change(&source.shape, &shape) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHIFTDIM-SPARSE",
            "sparse arrays support only unchanged two-dimensional shape",
            0,
        ));
    }
    let mut output = source.clone();
    output.shape = shape;
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    if matches!(shift, Shift::Explicit(value) if value > 0) || matches!(shift, Shift::Dynamic) {
        output.contiguity = ContiguityFact::Unknown;
        output.layout = LayoutFact::Unknown;
    }
    if matches!(shift, Shift::Explicit(value) if value > 0) || matches!(shift, Shift::Dynamic) {
        if let ValueKindFact::Cell(cell) = &mut output.kind {
            cell.elements.clear();
            cell.elements_complete = false;
        }
        if let ValueKindFact::Struct(structure) = &mut output.kind {
            structure.elements.clear();
            structure.elements_complete = false;
        }
    }
    if source.storage == StorageFact::Scalar && !matches!(&source.kind, ValueKindFact::Cell(_)) {
        output.shape = source.shape.clone();
        output.storage = source.storage;
    }
    if !diagnostics.is_empty() {
        output.shape = ShapeFact::Unknown;
    }
    let mut outputs = vec![output];
    if request.arguments.len() == 1 {
        outputs.push(default_double_scalar());
    }
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}

fn source_supported(source: &ValueFact) -> bool {
    matches!(
        &source.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Cell(_)
            | ValueKindFact::Struct(_)
    )
}

fn control_supported(control: &ValueFact) -> bool {
    if control.shape.is_definitely_nonscalar()
        || matches!(
            control.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        )
    {
        return false;
    }
    matches!(
        &control.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}

fn literal_shift(value: Option<&LiteralValue>) -> Shift {
    match value {
        None | Some(LiteralValue::Unknown) => Shift::Dynamic,
        Some(LiteralValue::Bool(value)) => Shift::Explicit(if *value { 1 } else { 0 }),
        Some(LiteralValue::Integer { .. }) => value
            .and_then(LiteralValue::integer_value_i128)
            .and_then(|value| isize::try_from(value).ok())
            .map_or(Shift::Invalid, Shift::Explicit),
        Some(LiteralValue::Number(value)) => float_shift(*value),
        Some(LiteralValue::Real { text, .. }) => {
            text.parse::<f64>().map_or(Shift::Invalid, float_shift)
        }
        _ => Shift::Invalid,
    }
}

fn float_shift(value: f64) -> Shift {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= isize::MIN as f64
        && value < (isize::MAX as u128 + 1) as f64
    {
        Shift::Explicit(value as isize)
    } else {
        Shift::Invalid
    }
}

fn proven_shape_change(before: &ShapeFact, after: &ShapeFact) -> bool {
    if matches!(before, ShapeFact::Ranked { .. }) || matches!(after, ShapeFact::Ranked { .. }) {
        return false;
    }
    let (Some(before), Some(after)) = (before.known_dims(), after.known_dims()) else {
        return false;
    };
    if before.iter().chain(&after).any(Option::is_none) {
        return false;
    }
    !ShapeFact::from(before).is_proven_equivalent(&ShapeFact::from(after))
}

fn shifted_shape(shape: &ShapeFact, shift: Shift) -> (ShapeFact, Option<usize>) {
    if let ShapeFact::Ranked { rank } = shape {
        return match shift {
            Shift::Explicit(n) if n < 0 => {
                let added = n
                    .checked_abs()
                    .and_then(|value| usize::try_from(value).ok());
                match added.and_then(|added| rank.checked_add(added)) {
                    Some(rank) => (ShapeFact::Ranked { rank }, None),
                    None => (ShapeFact::Unknown, None),
                }
            }
            _ => (ShapeFact::Unknown, None),
        };
    }
    let Some(mut dims) = shape.known_dims() else {
        return (ShapeFact::Unknown, None);
    };
    if dims.is_empty() {
        dims = vec![Some(1), Some(1)];
    }
    if dims.len() == 1 {
        dims.insert(0, Some(1));
    }
    while dims.len() > 2 && dims.last() == Some(&Some(1)) {
        dims.pop();
    }
    match shift {
        Shift::RemoveLeading => {
            let Some(first) = dims.iter().position(|dimension| *dimension != Some(1)) else {
                return (ShapeFact::from(dims), Some(0));
            };
            if dims[first].is_none() {
                return (ShapeFact::Unknown, None);
            }
            dims.drain(..first);
            if dims.len() == 1 {
                dims.push(Some(1));
            }
            (ShapeFact::from(dims), Some(first))
        }
        Shift::Explicit(n) if n < 0 => {
            let Some(count) = n
                .checked_abs()
                .and_then(|value| usize::try_from(value).ok())
            else {
                return (ShapeFact::Unknown, None);
            };
            let Some(rank) = dims.len().checked_add(count) else {
                return (ShapeFact::Unknown, None);
            };
            // Static inference cannot allocate metadata proportional to an
            // untrusted literal. Keep the rank when expansion exceeds the
            // already-materialized input geometry.
            if count > dims.len() {
                return (ShapeFact::Ranked { rank }, None);
            }
            let mut output = Vec::new();
            if output.try_reserve_exact(rank).is_err() {
                return (ShapeFact::Unknown, None);
            }
            output.resize(count, Some(1));
            output.extend(dims);
            (ShapeFact::from(output), None)
        }
        Shift::Explicit(n) => {
            let amount = (n as usize) % dims.len();
            dims.rotate_left(amount);
            while dims.len() > 2 && dims.last() == Some(&Some(1)) {
                dims.pop();
            }
            (ShapeFact::from(dims), None)
        }
        Shift::Dynamic | Shift::Invalid => (ShapeFact::Unknown, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::array::shape::shiftdim::SHIFTDIM_CATALOG_ENTRY;

    fn source(shape: Vec<Option<usize>>) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    }

    fn request(
        arguments: Vec<ValueFact>,
        literals: Vec<LiteralValue>,
        count: RequestedOutputCount,
    ) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(count),
        }
    }

    #[test]
    fn default_and_explicit_forms_keep_shape_and_output_roles() {
        let original = source(vec![Some(1), Some(1), Some(2), Some(3)]);
        let inferred = infer(
            &request(
                vec![original.clone()],
                vec![LiteralValue::Unknown],
                RequestedOutputCount::Exactly(2),
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs.len(), 2);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(inferred.outputs[1].shape, ShapeFact::Scalar);

        let control = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let explicit = infer(
            &request(
                vec![original.clone(), control.clone()],
                vec![LiteralValue::Unknown, LiteralValue::Number(-2.0)],
                RequestedOutputCount::One,
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert_eq!(
            explicit.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(1), Some(1), Some(1), Some(2), Some(3)])
        );
        let rotated = infer(
            &request(
                vec![original, control],
                vec![LiteralValue::Unknown, LiteralValue::Number(1.0)],
                RequestedOutputCount::One,
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert_eq!(
            rotated.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(2), Some(3)])
        );
    }

    #[test]
    fn unknown_and_invalid_controls_do_not_invent_geometry() {
        let original = source(vec![Some(1), None, Some(3)]);
        let inferred = infer(
            &request(
                vec![original.clone()],
                vec![LiteralValue::Unknown],
                RequestedOutputCount::One,
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);

        let control = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        let invalid = infer(
            &request(
                vec![original, control],
                vec![LiteralValue::Unknown, LiteralValue::Number(1.0)],
                RequestedOutputCount::Exactly(2),
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert!(!invalid.diagnostics.is_empty());
        assert_eq!(invalid.outputs[0].shape, ShapeFact::Unknown);
    }

    #[test]
    fn sparse_cycles_and_large_literals_keep_sound_shape_facts() {
        let mut sparse = source(vec![Some(2), Some(3)]);
        sparse.storage = StorageFact::Sparse;
        let control = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let full_cycle = infer(
            &request(
                vec![sparse.clone(), control.clone()],
                vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
                RequestedOutputCount::One,
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert!(full_cycle.diagnostics.is_empty());
        assert_eq!(full_cycle.outputs[0].shape, sparse.shape);

        let changed = infer(
            &request(
                vec![sparse, control.clone()],
                vec![LiteralValue::Unknown, LiteralValue::Number(1.0)],
                RequestedOutputCount::One,
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert_eq!(changed.diagnostics.len(), 1);

        let large = infer(
            &request(
                vec![source(vec![Some(2), Some(3)]), control],
                vec![LiteralValue::Unknown, LiteralValue::Number(-1_000_000.0)],
                RequestedOutputCount::One,
            ),
            &SHIFTDIM_CATALOG_ENTRY,
        );
        assert_eq!(
            large.outputs[0].shape,
            ShapeFact::Ranked { rank: 1_000_002 }
        );
    }
}
