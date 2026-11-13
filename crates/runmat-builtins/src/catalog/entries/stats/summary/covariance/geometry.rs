use crate::catalog::inference::argument_error;
use runmat_types::*;

#[derive(Clone)]
pub(super) struct ObservationGeometry {
    pub rows: DimensionFact,
    pub columns: DimensionFact,
    pub variables: DimensionFact,
}

pub(super) fn observations(
    input: &ValueFact,
    index: usize,
    allow_complex: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ObservationGeometry {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical
        | ValueKindFact::Unknown => {}
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) if allow_complex => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-COVARIANCE-INPUT",
            "observations must be numeric arrays in a supported domain",
            index,
        )),
    }
    let (rows, columns) = match &input.shape {
        ShapeFact::Scalar => (DimensionFact::Known(1), DimensionFact::Known(1)),
        ShapeFact::Shaped { dims } if dims.len() <= 2 => (
            dims.first().cloned().unwrap_or(DimensionFact::Known(1)),
            dims.get(1).cloned().unwrap_or(DimensionFact::Known(1)),
        ),
        ShapeFact::Shaped { dims } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COVARIANCE-RANK",
                "observations must be two-dimensional",
                index,
            ));
            (
                dims.first().cloned().unwrap_or(DimensionFact::Unknown),
                DimensionFact::Unknown,
            )
        }
        ShapeFact::Ranked { rank } if *rank > 2 => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COVARIANCE-RANK",
                "observations must be two-dimensional",
                index,
            ));
            (DimensionFact::Unknown, DimensionFact::Unknown)
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => {
            (DimensionFact::Unknown, DimensionFact::Unknown)
        }
    };
    let vector = match (&rows, &columns) {
        (DimensionFact::Known(1), _) | (_, DimensionFact::Known(1)) => Some(true),
        (DimensionFact::Known(_), DimensionFact::Known(_)) => Some(false),
        _ => None,
    };
    let variables = match vector {
        Some(true) => DimensionFact::Known(1),
        Some(false) => columns.clone(),
        None => DimensionFact::Unknown,
    };
    ObservationGeometry {
        rows,
        columns,
        variables,
    }
}

pub(super) fn square_shape(width: DimensionFact) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![width.clone(), width],
    }
}

pub(super) fn output_class(observations: &[&ValueFact]) -> Option<NumericClass> {
    let mut all_single = true;
    for observation in observations {
        match &observation.kind {
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                ..
            }) => {}
            ValueKindFact::Numeric(_) | ValueKindFact::Logical => all_single = false,
            ValueKindFact::Unknown => return None,
            _ => return None,
        }
    }
    Some(if all_single {
        NumericClass::Single
    } else {
        NumericClass::Double
    })
}

pub(super) fn numeric_output(
    input: &ValueFact,
    shape: ShapeFact,
    domain: NumericDomain,
    class: Option<NumericClass>,
) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(
        class.map_or(ValueKindFact::Unknown, |class| {
            ValueKindFact::Numeric(NumericFact { class, domain })
        }),
        shape,
        storage,
    );
    if class.is_none() || output.shape.element_count().is_none() {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    if !matches!(&input.residency, ResidencyFact::Host) {
        // Both dedicated provider kernels and host materialization exist.
        // Placement is dynamic, but output class follows the observation data.
        output.residency = ResidencyFact::Unknown;
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}
