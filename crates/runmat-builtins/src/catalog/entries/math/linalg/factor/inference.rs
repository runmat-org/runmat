//! Facts shared by the six ordinary matrix factorization calls.
//!
//! Algorithm- and output-count-specific shapes stay in each identity leaf.
use crate::catalog::entries::math::linalg::factor::FactorOption;
use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, DynamicReason, InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn literal_factor_option(request: &CallRequest, index: usize) -> Option<FactorOption> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => FactorOption::parse(text),
        _ => None,
    }
}

pub(super) fn matrix_input(
    request: &CallRequest,
    maximum_arguments: usize,
    square: bool,
    invalid_argument_code: &'static str,
    invalid_input_code: &'static str,
) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            invalid_argument_code,
            "a matrix input is required",
            0,
        ));
        return diagnostics;
    }
    if request.arguments.len() > maximum_arguments {
        diagnostics.push(argument_error(
            invalid_argument_code,
            "too many factorization arguments",
            maximum_arguments,
        ));
    }
    let matrix = &request.arguments[0];
    if !matches!(
        &matrix.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            invalid_input_code,
            "expected a numeric or logical matrix",
            0,
        ));
    }
    if matrix.shape.rank().is_some_and(|rank| rank > 2) {
        diagnostics.push(argument_error(
            invalid_input_code,
            "factorization requires a two-dimensional matrix",
            0,
        ));
    }
    if square {
        if let Some((Some(rows), Some(columns))) = dimensions(matrix) {
            if rows != columns {
                diagnostics.push(argument_error(
                    invalid_input_code,
                    "factorization requires a square matrix",
                    0,
                ));
            }
        }
    }
    diagnostics
}

pub(super) fn schur_operand(
    value: &ValueFact,
    argument: usize,
    invalid_input_code: &'static str,
) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    match &value.kind {
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Single | NumericClass::Double) => {}
        ValueKindFact::Unknown => {}
        _ => diagnostics.push(argument_error(
            invalid_input_code,
            "Schur operands must be single or double numeric matrices",
            argument,
        )),
    }
    if value.shape.rank().is_some_and(|rank| rank > 2)
        || dimensions(value)
            .is_some_and(|(rows, cols)| rows.zip(cols).is_some_and(|(rows, cols)| rows != cols))
    {
        diagnostics.push(argument_error(
            invalid_input_code,
            "Schur operands must be square two-dimensional matrices",
            argument,
        ));
    }
    if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            invalid_input_code,
            "Schur operands require dense storage",
            argument,
        ));
    }
    if matches!(
        &value.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) {
        diagnostics.push(argument_error(
            invalid_input_code,
            "Schur operands require host residency",
            argument,
        ));
    }
    diagnostics
}

pub(super) fn dimensions(value: &ValueFact) -> Option<(Option<usize>, Option<usize>)> {
    let dimensions = value.shape.known_dims()?;
    match dimensions.as_slice() {
        [rows, columns] => Some((*rows, *columns)),
        _ => None,
    }
}

pub(super) fn matrix_shape(rows: Option<usize>, columns: Option<usize>) -> ShapeFact {
    ShapeFact::from(vec![rows, columns])
}

pub(super) fn extent(value: &ValueFact) -> Option<usize> {
    dimensions(value).and_then(|(rows, _)| rows)
}

pub(super) fn minimum_extent(value: &ValueFact) -> Option<usize> {
    dimensions(value).and_then(|(rows, columns)| Some(rows?.min(columns?)))
}

/// Preserve only class/domain facts common to every admitted runtime route.
/// Real nonsymmetric eigenproblems can acquire complex spectra, while complex
/// input can produce real eigenvalues. `spectrum` therefore leaves the value
/// kind dynamic until the numerical result is known.
pub(super) fn factor_output(
    input: &ValueFact,
    shape: ShapeFact,
    real_only: bool,
    spectrum: bool,
    host_only: bool,
) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = shape;
    result.storage = StorageFact::Dense;
    result.residency = if host_only || input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    let numeric = match &input.kind {
        ValueKindFact::Numeric(value) => Some(*value),
        ValueKindFact::Logical => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => None,
    };
    if let Some(numeric) = numeric {
        let class = match numeric.class {
            NumericClass::Single => NumericClass::Single,
            _ => NumericClass::Double,
        };
        if real_only || (!spectrum && numeric.domain == NumericDomain::Real) {
            result.kind = ValueKindFact::Numeric(NumericFact {
                class,
                domain: if real_only {
                    NumericDomain::Real
                } else {
                    numeric.domain
                },
            });
        }
    }
    result
}

/// Project outputs whose host contract is binary64, such as SVD factors and
/// QR permutations. Provider results can use their native precision, so keep
/// their class dynamic unless the input is known to be host-resident.
pub(super) fn binary64_output(
    input: &ValueFact,
    shape: ShapeFact,
    real_only: bool,
    host_only: bool,
) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = shape;
    result.storage = StorageFact::Dense;
    result.residency = if host_only || input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    if host_only || input.residency == ResidencyFact::Host {
        let domain = match &input.kind {
            ValueKindFact::Numeric(numeric) => Some(if real_only {
                NumericDomain::Real
            } else {
                numeric.domain
            }),
            ValueKindFact::Logical => Some(NumericDomain::Real),
            _ => None,
        };
        if let Some(domain) = domain {
            result.kind = ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain,
            });
        }
    }
    result
}

pub(super) fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
