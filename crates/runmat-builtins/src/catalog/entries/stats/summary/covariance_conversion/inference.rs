use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum ConversionKind {
    Corrcov,
    Cov2corr,
}

fn matrix_width(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) -> DimensionFact {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical
        | ValueKindFact::Unknown => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-COVARIANCE-CONVERSION-INPUT",
            "covariance input must be a real numeric matrix",
            0,
        )),
    }
    match &input.shape {
        ShapeFact::Scalar => DimensionFact::Known(1),
        ShapeFact::Shaped { dims } if dims.len() <= 2 => {
            let rows = dims.first().cloned().unwrap_or(DimensionFact::Known(1));
            let columns = dims.get(1).cloned().unwrap_or(DimensionFact::Known(1));
            if let (DimensionFact::Known(rows), DimensionFact::Known(columns)) = (&rows, &columns) {
                if rows != columns {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-COVARIANCE-CONVERSION-SQUARE",
                        "covariance matrix must be square",
                        0,
                    ));
                }
            }
            if matches!(&rows, DimensionFact::Unknown)
                && matches!(&columns, DimensionFact::Known(_))
            {
                columns
            } else {
                rows
            }
        }
        ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COVARIANCE-CONVERSION-RANK",
                "covariance matrix must be two-dimensional",
                0,
            ));
            DimensionFact::Unknown
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => DimensionFact::Unknown,
    }
}

fn output_class(input: &ValueFact) -> Option<NumericClass> {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }) => Some(NumericClass::Single),
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => Some(NumericClass::Double),
        _ => None,
    }
}

fn output(input: &ValueFact, shape: ShapeFact, class: Option<NumericClass>) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut fact = match class {
        Some(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            shape,
            storage,
        ),
        None => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.shape = shape;
            fact.storage = storage;
            fact
        }
    };
    if !matches!(&input.residency, ResidencyFact::Host) {
        // A provider hook can keep results resident, but a supported
        // host-materialization path can also be selected at execution.
        fact.residency = ResidencyFact::Unknown;
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    fact
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: ConversionKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-COVARIANCE-CONVERSION-ARITY",
            "covariance conversion expects exactly one input",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let (width, class, input) = if let Some(input) = request.arguments.first() {
        (
            matrix_width(input, &mut diagnostics),
            output_class(input),
            input.clone(),
        )
    } else {
        (
            DimensionFact::Unknown,
            None,
            ValueFact::unknown(DynamicReason::RuntimeValue),
        )
    };
    let correlation = ShapeFact::Shaped {
        dims: vec![width.clone(), width.clone()],
    };
    let standard_deviation = match kind {
        ConversionKind::Corrcov => ShapeFact::Shaped {
            dims: vec![width, DimensionFact::Known(1)],
        },
        ConversionKind::Cov2corr => ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), width],
        },
    };
    let correlation = output(&input, correlation, class);
    let standard_deviation = output(&input, standard_deviation, class);
    let outputs = match kind {
        ConversionKind::Corrcov => vec![correlation, standard_deviation],
        ConversionKind::Cov2corr => vec![standard_deviation, correlation],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
