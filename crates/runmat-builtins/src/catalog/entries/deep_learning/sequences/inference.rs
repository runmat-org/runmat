use super::pad_inference_options;
use crate::{
    catalog::inference::{argument_error, finish_fixed, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn numeric_sequence_possible(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        _ => false,
    }
}

pub(super) fn combvec(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-DL-COMBVEC-ARITY",
            "combvec requires at least one numeric input",
            0,
        ));
    }
    let mut rows = Some(0usize);
    let mut columns = Some(1usize);
    for (index, value) in request.arguments.iter().enumerate() {
        if !numeric_sequence_possible(value) || value.shape.rank().is_some_and(|rank| rank > 2) {
            diagnostics.push(argument_error(
                "RM-DL-COMBVEC-INPUT",
                "each input must be a real numeric scalar or two-dimensional matrix",
                index,
            ));
        }
        let dims = value.shape.known_dims();
        let (row_count, col_count) = match dims.as_deref() {
            Some([]) => (Some(0), Some(0)),
            Some([columns]) => (Some(1), *columns),
            Some([rows, columns]) => (*rows, *columns),
            _ => (None, None),
        };
        if col_count == Some(0) || row_count == Some(0) {
            diagnostics.push(argument_error(
                "RM-DL-COMBVEC-EMPTY",
                "each input must have at least one row and column",
                index,
            ));
        }
        rows = rows.and_then(|total| row_count.and_then(|count| total.checked_add(count)));
        columns = columns.and_then(|total| col_count.and_then(|count| total.checked_mul(count)));
    }
    if columns.is_some_and(|count| count > 1_000_000)
        || rows
            .zip(columns)
            .is_some_and(|(r, c)| r.checked_mul(c).is_none_or(|count| count > 10_000_000))
    {
        diagnostics.push(argument_error(
            "RM-DL-COMBVEC-CAPACITY",
            "the combination matrix exceeds the supported output capacity",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        double(ShapeFact::from(vec![rows, columns])),
        diagnostics,
    )
}

fn cell_sequence_possible(cell: &CellFact) -> bool {
    numeric_sequence_possible(&cell.element) && cell.elements.iter().all(numeric_sequence_possible)
}

#[derive(Clone, Copy)]
enum SequenceClass {
    Same(NumericClass),
    Mixed,
    Unknown,
}

fn sequence_class(cell: &CellFact) -> SequenceClass {
    let common = cell.element.numeric().map(|numeric| numeric.class);
    let mut class = if cell.elements_complete { None } else { common };
    for element in &cell.elements {
        let Some(element_class) = element.numeric().map(|numeric| numeric.class) else {
            return SequenceClass::Unknown;
        };
        if class.is_some_and(|existing| existing != element_class) {
            return SequenceClass::Mixed;
        }
        class = Some(element_class);
    }
    if cell.elements_complete {
        SequenceClass::Same(class.unwrap_or(NumericClass::Double))
    } else {
        class.map_or(SequenceClass::Unknown, SequenceClass::Same)
    }
}

pub(super) fn padsequences(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 || !(request.arguments.len() - 2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-DL-PADSEQUENCES-ARITY",
            "padsequences requires a cell array, a padding dimension, and paired options",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let sequences = request.arguments.first();
    if let Some(value) = sequences {
        match &value.kind {
            ValueKindFact::Unknown => {}
            ValueKindFact::Cell(cell) if cell_sequence_possible(cell) => {}
            _ => diagnostics.push(argument_error(
                "RM-DL-PADSEQUENCES-SEQUENCES",
                "sequences must be a cell array of real numeric arrays",
                0,
            )),
        }
    }
    if let Some(value) = request.arguments.get(1) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || value.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-DL-PADSEQUENCES-DIMENSION",
                "paddingDim must be a positive integer scalar",
                1,
            ));
        }
        pad_inference_options::validate_dimension(request, &mut diagnostics);
    }
    let uniform = pad_inference_options::validate_options(request, &mut diagnostics);
    let class = sequences.map_or(SequenceClass::Unknown, |value| match &value.kind {
        ValueKindFact::Cell(cell) => sequence_class(cell),
        _ => SequenceClass::Unknown,
    });
    let numeric = |class| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Unknown,
            StorageFact::Dense,
        )
    };
    let output = match uniform {
        Some(true) => match class {
            SequenceClass::Same(class) => numeric(class),
            SequenceClass::Mixed => numeric(NumericClass::Double),
            SequenceClass::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        Some(false) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(match class {
                    SequenceClass::Same(class) => numeric(class),
                    SequenceClass::Mixed | SequenceClass::Unknown => {
                        ValueFact::unknown(DynamicReason::RuntimeValue)
                    }
                }),
                elements: Vec::new(),
                elements_complete: false,
            }),
            sequences.map_or(ShapeFact::Unknown, |value| value.shape.clone()),
            StorageFact::Dense,
        ),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mask = match uniform {
        Some(true) => ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::Unknown,
            StorageFact::Dense,
        ),
        Some(false) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::proven(
                    ValueKindFact::Logical,
                    ShapeFact::Unknown,
                    StorageFact::Dense,
                )),
                elements: Vec::new(),
                elements_complete: false,
            }),
            sequences.map_or(ShapeFact::Unknown, |value| value.shape.clone()),
            StorageFact::Dense,
        ),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let outputs = if diagnostics.is_empty() {
        vec![output, mask]
    } else {
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ]
    };
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
