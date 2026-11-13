use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-PAGEFUN-ARITY",
            "expected a matrix-product selector and two operands",
            0,
        ));
    }
    let selector = request.literals.literal_string_at(0).or_else(|| {
        match request.arguments.first().map(|value| &value.kind) {
            Some(ValueKindFact::Callable(callable)) => callable
                .identity
                .as_ref()
                .and_then(CallableIdentity::display_name),
            _ => None,
        }
    });
    if selector.is_some_and(|s| {
        !s.trim()
            .trim_start_matches('@')
            .eq_ignore_ascii_case("mtimes")
    }) {
        diagnostics.push(argument_error(
            "RM-PAGEFUN-SELECTOR",
            "only the matrix-product operator is supported",
            0,
        ));
    }
    if let Some(selector) = request.arguments.first() {
        if !matches!(
            selector.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Callable(_)
                | ValueKindFact::String
                | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-PAGEFUN-SELECTOR",
                "expected an operator handle or text selector",
                0,
            ));
        }
    }
    if let (Some(left), Some(right)) = (request.arguments.get(1), request.arguments.get(2)) {
        for (index, input) in [(1, left), (2, right)] {
            if !matches!(
                input.kind,
                ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) || input.storage == StorageFact::Sparse
                || input.numeric().is_some_and(|n| {
                    n.class.integer_class().is_some() && n.domain == NumericDomain::Complex
                })
            {
                diagnostics.push(argument_error(
                    "RM-PAGEFUN-INPUT",
                    "expected dense floating, real integer or logical operands",
                    index,
                ));
            }
        }
        output.kind = product_numeric(left, right)
            .map(ValueKindFact::Numeric)
            .unwrap_or(ValueKindFact::Unknown);
        output.shape = product_shape(&left.shape, &right.shape, &mut diagnostics);
        output.layout = LayoutFact::ColumnMajor;
        output.residency =
            if left.residency == ResidencyFact::Host && right.residency == ResidencyFact::Host {
                ResidencyFact::Host
            } else {
                ResidencyFact::Unknown
            };
        output.storage = if output.residency == ResidencyFact::Host {
            StorageFact::Dense
        } else {
            StorageFact::Unknown
        };
        if let (Some(l), Some(r)) = (left.numeric(), right.numeric()) {
            if l.class.integer_class().is_some() || r.class.integer_class().is_some() {
                if matrix_nonscalar(&left.shape) && matrix_nonscalar(&right.shape) {
                    diagnostics.push(argument_error(
                        "RM-PAGEFUN-INTEGER",
                        "integer multiplication requires a scalar page",
                        1,
                    ));
                }
                if l.class != r.class
                    && l.class != NumericClass::Double
                    && r.class != NumericClass::Double
                {
                    diagnostics.push(argument_error(
                        "RM-PAGEFUN-INTEGER",
                        "integer operands must have matching classes or pair with a scalar double",
                        1,
                    ));
                }
            }
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn product_numeric(left: &ValueFact, right: &ValueFact) -> Option<NumericFact> {
    let numeric = |value: &ValueFact| match value.kind {
        ValueKindFact::Logical => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => value.numeric(),
    };
    let (left, right) = (numeric(left)?, numeric(right)?);
    let class = if left.class.integer_class().is_some() {
        left.class
    } else if right.class.integer_class().is_some() {
        right.class
    } else if left.class == NumericClass::Single || right.class == NumericClass::Single {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let domain = if left.domain == NumericDomain::Complex || right.domain == NumericDomain::Complex
    {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    };
    Some(NumericFact { class, domain })
}

fn matrix_dims(shape: &ShapeFact) -> Option<Vec<Option<usize>>> {
    let mut dims = shape.known_dims()?.to_vec();
    match dims.len() {
        0 => dims = vec![Some(1), Some(1)],
        1 => dims.insert(0, Some(1)),
        _ => {}
    }
    Some(dims)
}
fn matrix_nonscalar(shape: &ShapeFact) -> bool {
    matrix_dims(shape).is_some_and(|dims| dims[..2].iter().any(|d| d.is_some_and(|n| n != 1)))
}
fn product_shape(
    left: &ShapeFact,
    right: &ShapeFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let (Some(left), Some(right)) = (matrix_dims(left), matrix_dims(right)) else {
        return ShapeFact::Unknown;
    };
    let scalar = |dims: &[Option<usize>]| dims[0] == Some(1) && dims[1] == Some(1);
    let (rows, cols) = if scalar(&left) {
        (right[0], right[1])
    } else if scalar(&right) {
        (left[0], left[1])
    } else if left[..2].iter().any(|d| d.is_some_and(|n| n != 1))
        && right[..2].iter().any(|d| d.is_some_and(|n| n != 1))
    {
        if left[1].is_some() && right[0].is_some() && left[1] != right[0] {
            diagnostics.push(argument_error(
                "RM-PAGEFUN-SHAPE",
                "inner matrix dimensions must agree",
                2,
            ));
        }
        (left[0], right[1])
    } else {
        return ShapeFact::Unknown;
    };
    let mut output = vec![rows, cols];
    for axis in 2..left.len().max(right.len()) {
        let l = left.get(axis).copied().unwrap_or(Some(1));
        let r = right.get(axis).copied().unwrap_or(Some(1));
        output.push(match (l, r) {
            (Some(1), other) | (other, Some(1)) => other,
            (Some(l), Some(r)) if l != r => {
                diagnostics.push(argument_error(
                    "RM-PAGEFUN-PAGES",
                    "page dimensions are incompatible",
                    2,
                ));
                None
            }
            (Some(n), _) | (_, Some(n)) => Some(n),
            _ => None,
        });
    }
    ShapeFact::from(output)
}
