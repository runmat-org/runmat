use runmat_types::{
    CellFact, DynamicReason, NumericDomain, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use super::super::super::super::support;

pub(super) struct ColumnFact {
    kind: ValueKindFact,
    pub sequence_length: Option<usize>,
    dynamic: bool,
}

impl ColumnFact {
    pub(super) fn with_shape(self, shape: ShapeFact) -> ValueFact {
        let mut output = if self.dynamic {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        } else {
            ValueFact::proven(self.kind, shape.clone(), StorageFact::Dense)
        };
        output.shape = shape;
        support::facts::materialize(&mut output);
        output.residency = ResidencyFact::Host;
        output
    }
}

pub(super) fn infer(input: &ValueFact) -> ColumnFact {
    match &input.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Real
                && matches!(input.storage, StorageFact::Scalar | StorageFact::Dense) =>
        {
            known(
                ValueKindFact::Numeric(*numeric),
                input.shape.element_count(),
            )
        }
        ValueKindFact::Logical
            if matches!(input.storage, StorageFact::Scalar | StorageFact::Dense) =>
        {
            known(ValueKindFact::Logical, input.shape.element_count())
        }
        ValueKindFact::String => known(ValueKindFact::String, input.shape.element_count()),
        ValueKindFact::Character if is_row(&input.shape) == Some(true) => {
            known(ValueKindFact::String, input.shape.element_count())
        }
        ValueKindFact::Character if is_row(&input.shape) == Some(false) => cell(input, Some(1)),
        ValueKindFact::Cell(cell) => known(
            ValueKindFact::Cell(CellFact {
                element: cell.element.clone(),
                elements: Vec::new(),
                elements_complete: false,
            }),
            input.shape.element_count(),
        ),
        ValueKindFact::Unknown => dynamic(),
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                || matches!(input.storage, StorageFact::Sparse | StorageFact::Opaque) =>
        {
            cell(input, Some(1))
        }
        ValueKindFact::Logical
            if matches!(input.storage, StorageFact::Sparse | StorageFact::Opaque) =>
        {
            cell(input, Some(1))
        }
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Character => dynamic(),
        _ => cell(input, Some(1)),
    }
}

fn known(kind: ValueKindFact, sequence_length: Option<usize>) -> ColumnFact {
    ColumnFact {
        kind,
        sequence_length,
        dynamic: false,
    }
}

fn dynamic() -> ColumnFact {
    ColumnFact {
        kind: ValueKindFact::Unknown,
        sequence_length: None,
        dynamic: true,
    }
}

fn cell(input: &ValueFact, sequence_length: Option<usize>) -> ColumnFact {
    known(
        ValueKindFact::Cell(CellFact {
            element: Box::new(input.clone()),
            elements: Vec::new(),
            elements_complete: false,
        }),
        sequence_length,
    )
}

fn is_row(shape: &ShapeFact) -> Option<bool> {
    match shape {
        ShapeFact::Scalar => Some(true),
        ShapeFact::Shaped { dims } => dims.first().and_then(|dimension| match dimension {
            runmat_types::DimensionFact::Known(rows) => Some(*rows == 1),
            runmat_types::DimensionFact::Symbolic(_) | runmat_types::DimensionFact::Unknown => None,
        }),
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => None,
    }
}
