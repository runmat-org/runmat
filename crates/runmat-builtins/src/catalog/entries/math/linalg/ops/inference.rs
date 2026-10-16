use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

mod dimension;
use dimension::literal_positive_dimension;

#[derive(Clone, Copy)]
pub(super) enum Operation {
    Transpose,
    Pagetranspose,
    Dot,
    Trace,
    Cross,
    Pagemtimes,
}

impl Operation {
    const fn accepts(self, inputs: usize) -> bool {
        match self {
            Self::Transpose | Self::Pagetranspose | Self::Trace => inputs == 1,
            Self::Dot | Self::Cross => inputs == 2 || inputs == 3,
            Self::Pagemtimes => inputs == 2 || inputs == 4,
        }
    }
}

fn output_numeric(
    left: &ValueFact,
    right: Option<&ValueFact>,
    single_if_either: bool,
) -> ValueKindFact {
    let logical = NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    };
    let left_numeric = left
        .numeric()
        .or_else(|| matches!(&left.kind, ValueKindFact::Logical).then_some(logical));
    let right_numeric = right.and_then(|value| {
        value
            .numeric()
            .or_else(|| matches!(&value.kind, ValueKindFact::Logical).then_some(logical))
    });
    if left_numeric.is_none() || right.is_some_and(|_| right_numeric.is_none()) {
        return ValueKindFact::Unknown;
    }
    let single = if single_if_either {
        left_numeric.is_some_and(|n| n.class == NumericClass::Single)
            || right_numeric.is_some_and(|n| n.class == NumericClass::Single)
    } else {
        left_numeric.is_some_and(|n| n.class == NumericClass::Single)
            && right_numeric.is_none_or(|n| n.class == NumericClass::Single)
    };
    let complex = left_numeric.is_some_and(|n| n.domain == NumericDomain::Complex)
        || right_numeric.is_some_and(|n| n.domain == NumericDomain::Complex);
    ValueKindFact::Numeric(NumericFact {
        class: if single {
            NumericClass::Single
        } else {
            NumericClass::Double
        },
        domain: if complex {
            NumericDomain::Complex
        } else {
            NumericDomain::Real
        },
    })
}

fn numeric_result(kind: ValueKindFact, shape: ShapeFact, resident: bool) -> ValueFact {
    let single_or_complex = matches!(&kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Single || n.domain == NumericDomain::Complex);
    let mut result = ValueFact::proven(
        kind,
        shape.clone(),
        if shape.element_count() == Some(1) && !single_or_complex {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    );
    result.residency = if resident {
        ResidencyFact::Unknown
    } else {
        ResidencyFact::Host
    };
    result
}

fn check_numeric(
    input: &ValueFact,
    index: usize,
    allow_logical: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let supported = match &input.kind {
        ValueKindFact::Unknown | ValueKindFact::Numeric(_) => true,
        ValueKindFact::Logical => allow_logical,
        _ => false,
    };
    if !supported {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINALG-OPS-CLASS",
            "matrix operation requires numeric operands",
            index,
        ));
    }
}

fn transpose(
    input: &ValueFact,
    pagewise: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    if matches!(
        &input.kind,
        ValueKindFact::Unknown | ValueKindFact::Object(_)
    ) {
        return ValueFact::unknown(DynamicReason::DynamicDispatch);
    }
    if !pagewise
        && input
            .shape
            .known_dims()
            .is_some_and(|dims| dims.iter().skip(2).any(|d| d.is_some_and(|n| n != 1)))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRANSPOSE-RANK",
            "transpose requires a two-dimensional array",
            0,
        ));
    }
    let mut out = input.clone();
    out.shape = transpose_shape(&input.shape);
    // The same shape transform swaps axes one and two; pagewise form retains trailing axes.
    if out.shape.element_count() != Some(1) {
        match &mut out.kind {
            ValueKindFact::Cell(cell) => {
                cell.elements.clear();
                cell.elements_complete = false;
            }
            ValueKindFact::Struct(structure) => {
                structure.elements.clear();
                structure.elements_complete = false;
            }
            _ => {}
        }
    }
    if matches!(
        &input.kind,
        ValueKindFact::Cell(_) | ValueKindFact::Struct(_) | ValueKindFact::String
    ) {
        // Container elements move; source position and allocation facts cannot be reused.
        out.alias = AliasFact::Unknown;
        out.view = ViewFact::Unknown;
        out.layout = LayoutFact::Unknown;
    }
    if input.residency != ResidencyFact::Host {
        out.residency = ResidencyFact::Unknown;
        out.storage = StorageFact::Unknown;
        out.layout = LayoutFact::Unknown;
        out.alias = AliasFact::Unknown;
        out.view = ViewFact::Unknown;
    }
    out
}

fn trace(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    let accepted = matches!(
        &input.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
    );
    if !accepted {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRACE-CLASS",
            "trace requires a numeric, logical, or character matrix",
            0,
        ));
    }
    if let Some(dims) = input.shape.known_dims() {
        if dims.iter().skip(2).any(|d| d.is_some_and(|n| n != 1))
            || dims
                .first()
                .copied()
                .flatten()
                .zip(dims.get(1).copied().flatten())
                .is_some_and(|(r, c)| r != c)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRACE-SQUARE",
                "trace requires a square matrix",
                0,
            ));
        }
    }
    let kind = match &input.kind {
        ValueKindFact::Numeric(n) if n.class == NumericClass::Single => ValueKindFact::Numeric(*n),
        ValueKindFact::Numeric(n) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: n.domain,
        }),
        ValueKindFact::Unknown => ValueKindFact::Unknown,
        _ => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
    };
    numeric_result(
        kind,
        ShapeFact::Scalar,
        input.residency != ResidencyFact::Host,
    )
}

fn dot_shape(
    request: &CallRequest,
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let (Some(a), Some(b)) = (left.shape.known_dims(), right.shape.known_dims()) else {
        return ShapeFact::Unknown;
    };
    let av = a;
    let bv = b;
    let a_vector = av.iter().filter(|d| d.is_some_and(|n| n > 1)).count() <= 1;
    let b_vector = bv.iter().filter(|d| d.is_some_and(|n| n > 1)).count() <= 1;
    if request.arguments.len() == 2 && a_vector && b_vector {
        if left
            .shape
            .element_count()
            .zip(right.shape.element_count())
            .is_some_and(|(x, y)| x != y)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DOT-SIZE",
                "dot vector lengths must agree",
                1,
            ));
        }
        return ShapeFact::Scalar;
    }
    if av.len() != bv.len()
        || av
            .iter()
            .zip(&bv)
            .any(|(x, y)| x.zip(*y).is_some_and(|(m, n)| m != n))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DOT-SIZE",
            "dot arrays must have matching shape",
            1,
        ));
        return ShapeFact::Unknown;
    }
    let selected = if request.arguments.len() == 3 {
        request
            .literals
            .literal_args
            .get(2)
            .and_then(literal_positive_dimension)
    } else {
        av.iter()
            .position(|d| d.is_none_or(|n| n != 1))
            .and_then(|i| av[i].map(|_| i + 1))
    };
    let Some(dim) = selected else {
        return ShapeFact::Unknown;
    };
    let mut out = av;
    if dim <= out.len() {
        out[dim - 1] = Some(1);
    }
    ShapeFact::from(out)
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: Operation,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !operation.accepts(request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINALG-OPS-ARITY",
            "unsupported matrix operation argument count",
            request.arguments.len(),
        ));
    }
    let unknown = || ValueFact::unknown(DynamicReason::RuntimeValue);
    let first = request.arguments.first();
    let second = match operation {
        Operation::Pagemtimes if request.arguments.len() == 4 => request.arguments.get(2),
        _ => request.arguments.get(1),
    };
    let output = match (operation, first) {
        (Operation::Transpose | Operation::Pagetranspose, Some(input)) => transpose(
            input,
            matches!(operation, Operation::Pagetranspose),
            &mut diagnostics,
        ),
        (Operation::Trace, Some(input)) => trace(input, &mut diagnostics),
        (Operation::Dot | Operation::Cross | Operation::Pagemtimes, Some(left)) => {
            if let Some(right) = second {
                let allow_logical = matches!(operation, Operation::Dot | Operation::Cross);
                check_numeric(left, 0, allow_logical, &mut diagnostics);
                check_numeric(
                    right,
                    if matches!(operation, Operation::Pagemtimes) && request.arguments.len() == 4 {
                        2
                    } else {
                        1
                    },
                    allow_logical,
                    &mut diagnostics,
                );
                let shape = match operation {
                    Operation::Dot => dot_shape(request, left, right, &mut diagnostics),
                    Operation::Cross => {
                        let known = left.shape.known_dims().zip(right.shape.known_dims());
                        if known.as_ref().is_some_and(|(a, b)| a != b) {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-CROSS-SIZE",
                                "cross operand shapes must match",
                                1,
                            ));
                        }
                        if let Some((dims, _)) = known {
                            let selected = if request.arguments.len() == 3 {
                                request
                                    .literals
                                    .literal_args
                                    .get(2)
                                    .and_then(literal_positive_dimension)
                            } else {
                                dims.iter()
                                    .position(|dimension| *dimension == Some(3))
                                    .map(|index| index + 1)
                            };
                            if let Some(dimension) = selected {
                                if dims
                                    .get(dimension - 1)
                                    .is_none_or(|extent| extent.is_some_and(|n| n != 3))
                                {
                                    diagnostics.push(argument_error(
                                        "RM-CATALOG-CROSS-DIMENSION",
                                        "cross requires a selected dimension of length three",
                                        2,
                                    ));
                                }
                            } else if request.arguments.len() == 2
                                && dims.iter().all(Option::is_some)
                            {
                                diagnostics.push(argument_error(
                                    "RM-CATALOG-CROSS-DIMENSION",
                                    "cross requires a dimension of length three",
                                    0,
                                ));
                            }
                        }
                        left.shape.clone()
                    }
                    Operation::Pagemtimes => super::pagemtimes::page_product_shape(
                        request,
                        left,
                        right,
                        &mut diagnostics,
                    ),
                    _ => unreachable!(),
                };
                let kind = output_numeric(left, Some(right), matches!(operation, Operation::Cross));
                numeric_result(
                    kind,
                    shape,
                    left.residency != ResidencyFact::Host || right.residency != ResidencyFact::Host,
                )
            } else {
                unknown()
            }
        }
        _ => unknown(),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests;
