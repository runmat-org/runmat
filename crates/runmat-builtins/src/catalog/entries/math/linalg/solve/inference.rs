//! Conservative static facts for the ordinary matrix-solve and norm family.
//! Actual numerical options, provider precision, and singularity remain the
//! Runtime algorithm's authority; facts are emitted only when route-invariant.

use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn input(request: &CallRequest) -> ValueFact {
    request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
}

fn dimensions(input: &ValueFact) -> Option<(Option<usize>, Option<usize>)> {
    if input.is_scalar() {
        return Some((Some(1), Some(1)));
    }
    let dims = input.shape.known_dims()?;
    if dims.len() < 2 || dims.iter().skip(2).any(|dim| *dim != Some(1)) {
        return None;
    }
    Some((dims[0], dims[1]))
}

fn shape(rows: Option<usize>, columns: Option<usize>) -> ShapeFact {
    ShapeFact::from(vec![rows, columns])
}

fn output(input: &ValueFact, shape: ShapeFact, real: bool, double: bool) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.storage = if shape.element_count() == Some(1) {
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    };
    result.shape = shape;
    result.residency = if input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    // Provider hooks and owning-provider upload may choose their physical
    // floating precision. Only a proved host route can carry a class fact.
    if input.residency == ResidencyFact::Host {
        let numeric = match &input.kind {
            ValueKindFact::Numeric(numeric) => Some(*numeric),
            ValueKindFact::Logical => Some(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            _ => None,
        };
        if let Some(numeric) = numeric {
            result.kind = ValueKindFact::Numeric(NumericFact {
                class: if !double && numeric.class == NumericClass::Single {
                    NumericClass::Single
                } else {
                    NumericClass::Double
                },
                domain: if real {
                    NumericDomain::Real
                } else {
                    numeric.domain
                },
            });
        }
    }
    result
}

fn scalar(input: &ValueFact, double: bool, real: bool) -> ValueFact {
    let mut result = output(input, ShapeFact::Scalar, real, double);
    // A host Single scalar is physically a 1x1 Tensor in several members;
    // do not infer scalar storage from scalar geometry.
    result.storage = StorageFact::Unknown;
    result
}

fn diagnostics(
    request: &CallRequest,
    max_args: usize,
    code: &'static str,
) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(code, "matrix input is required", 0));
    } else if request.arguments.len() > max_args {
        diagnostics.push(argument_error(code, "too many arguments", max_args));
    }
    diagnostics
}

macro_rules! scalar_inference {
    ($name:ident, $max:expr, $code:expr, $double:expr, $real:expr) => {
        pub(super) fn $name(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            let matrix = input(request);
            finish_fixed_outputs(
                entry,
                request,
                vec![scalar(&matrix, $double, $real)],
                diagnostics(request, $max, $code),
            )
        }
    };
}

scalar_inference!(
    infer_cond,
    2,
    super::COND_ERROR_INVALID_ARGUMENT.code,
    false,
    true
);
scalar_inference!(
    infer_det,
    1,
    super::DET_ERROR_INVALID_INPUT.code,
    false,
    false
);
scalar_inference!(
    infer_norm,
    2,
    super::NORM_ERROR_INVALID_ARGUMENT.code,
    false,
    true
);
scalar_inference!(
    infer_rank,
    2,
    super::RANK_ERROR_INVALID_ARGUMENT.code,
    true,
    true
);
scalar_inference!(
    infer_rcond,
    1,
    super::RCOND_ERROR_INVALID_INPUT.code,
    false,
    true
);

pub(super) fn infer_inv(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let matrix = input(request);
    finish_fixed_outputs(
        entry,
        request,
        vec![output(&matrix, matrix.shape.clone(), false, false)],
        diagnostics(request, 1, super::INV_ERROR_INVALID_INPUT.code),
    )
}

pub(super) fn infer_linsolve(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let lhs = input(request);
    let rhs = request.arguments.get(1);
    let rows = dimensions(&lhs).map(|(_, cols)| cols).unwrap_or(None);
    let columns = rhs
        .and_then(dimensions)
        .map(|(_, cols)| cols)
        .unwrap_or(None);
    let mut solution = output(&lhs, shape(rows, columns), false, false);
    if lhs.residency == ResidencyFact::Host
        && rhs.is_some_and(|value| value.residency == ResidencyFact::Host)
    {
        solution.kind = match (lhs.numeric(), rhs.and_then(ValueFact::numeric)) {
            (Some(left), Some(right)) => ValueKindFact::Numeric(NumericFact {
                class: if left.class == NumericClass::Single && right.class == NumericClass::Single
                {
                    NumericClass::Single
                } else {
                    NumericClass::Double
                },
                domain: if left.domain == NumericDomain::Complex
                    || right.domain == NumericDomain::Complex
                {
                    NumericDomain::Complex
                } else {
                    NumericDomain::Real
                },
            }),
            _ => ValueKindFact::Unknown,
        };
    } else {
        solution.kind = ValueKindFact::Unknown;
        solution.residency = ResidencyFact::Unknown;
    }
    let mut condition = scalar(&lhs, true, true);
    if rhs.is_none_or(|value| value.residency != ResidencyFact::Host) {
        condition.kind = ValueKindFact::Unknown;
        condition.residency = ResidencyFact::Unknown;
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![solution, condition],
        diagnostics(request, 3, super::LINSOLVE_ERROR_INVALID_ARGUMENT.code),
    )
}

pub(super) fn infer_null(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let matrix = input(request);
    let rows = dimensions(&matrix)
        .map(|(_, columns)| columns)
        .unwrap_or(None);
    let mut basis = output(&matrix, shape(rows, None), false, false);
    if request.arguments.len() > 1 {
        // Rational basis selection and a dynamic tolerance have different
        // publication paths; neither proves the original floating class.
        basis.kind = ValueKindFact::Unknown;
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![basis],
        diagnostics(request, 2, super::NULL_ERROR_INVALID_ARGUMENT.code),
    )
}

pub(super) fn infer_pinv(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let matrix = input(request);
    let output_shape = dimensions(&matrix)
        .map(|(rows, columns)| shape(columns, rows))
        .unwrap_or(ShapeFact::Ranked { rank: 2 });
    finish_fixed_outputs(
        entry,
        request,
        vec![output(&matrix, output_shape, false, false)],
        diagnostics(request, 2, super::PINV_ERROR_INVALID_ARGUMENT.code),
    )
}

pub(super) fn infer_rref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let matrix = input(request);
    let pivots = output(&matrix, shape(Some(1), None), true, true);
    finish_fixed_outputs(
        entry,
        request,
        vec![output(&matrix, matrix.shape.clone(), false, true), pivots],
        diagnostics(request, 2, super::RREF_ERROR_INVALID_ARGUMENT.code),
    )
}

pub(super) fn infer_vecnorm(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let matrix = input(request);
    let output_shape = vecnorm_shape(request, &matrix);
    finish_fixed_outputs(
        entry,
        request,
        vec![output(&matrix, output_shape, true, false)],
        diagnostics(request, 3, super::VECNORM_ERROR_INVALID_ARGUMENT.code),
    )
}

fn vecnorm_shape(request: &CallRequest, input: &ValueFact) -> ShapeFact {
    let Some(mut dimensions) = input.shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    let axis = if request.arguments.len() >= 3 {
        let Some(index) = request
            .literals
            .literal_args
            .get(2)
            .and_then(positive_dimension)
        else {
            return ShapeFact::Ranked {
                rank: dimensions.len(),
            };
        };
        index - 1
    } else {
        let Some(index) = dimensions.iter().position(|dim| *dim != Some(1)) else {
            return input.shape.clone();
        };
        if dimensions[index].is_none() {
            return ShapeFact::Ranked {
                rank: dimensions.len(),
            };
        }
        index
    };
    if axis < dimensions.len() {
        dimensions[axis] = Some(1);
    }
    ShapeFact::from(dimensions)
}

fn positive_dimension(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Integer { .. } => usize::try_from(literal.integer_value_i128()?)
            .ok()
            .filter(|n| *n > 0),
        LiteralValue::Number(value)
            if value.is_finite() && *value > 0.0 && value.fract() == 0.0 =>
        {
            let integer = *value as u128;
            usize::try_from(integer)
                .ok()
                .filter(|n| *n > 0 && *n as f64 == *value)
        }
        _ => None,
    }
}
