//! Domain-owned static mechanics shared by downsample and upsample; resample
//! reuses only the exact structural-integer literal decoder.

use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SampleDirection {
    Down,
    Up,
}

pub(super) fn infer_sample(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    direction: SampleDirection,
) -> CallInference {
    let name = match direction {
        SampleDirection::Down => "downsample",
        SampleDirection::Up => "upsample",
    };
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SAMPLE-RATE-ARITY",
            format!("{name} expects X, N, and optional PHASE"),
            request.arguments.len().saturating_sub(1),
        ));
    }
    let input = request.arguments.first();
    if let Some(value) = input {
        if matches!(&value.kind, ValueKindFact::Numeric(fact)
            if fact.domain == NumericDomain::Complex && fact.class.integer_class().is_some())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SAMPLE-RATE-COMPLEX-INTEGER",
                format!("{name} cannot operate on complex integer samples without an exact integer-complex kernel"),
                0,
            ));
        }
        if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                &value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SAMPLE-RATE-INPUT",
                format!("{name} requires a dense numeric or logical input"),
                0,
            ));
        }
    }
    let factor = control(request, 1, true, "N", &mut diagnostics);
    let phase = if request.arguments.len() > 2 {
        control(request, 2, false, "PHASE", &mut diagnostics)
    } else {
        Some(0)
    };
    if let (Some(n), Some(p)) = (factor, phase) {
        if p >= n {
            diagnostics.push(argument_error(
                "RM-CATALOG-SAMPLE-RATE-PHASE",
                "PHASE must satisfy 0 <= PHASE < N",
                2,
            ));
        }
    }
    let output = match input {
        Some(input) if diagnostics.is_empty() => output(input, factor, phase, direction),
        Some(_) => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn control(
    request: &CallRequest,
    index: usize,
    positive: bool,
    label: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let value = request.arguments.get(index)?;
    if value.shape.is_definitely_nonscalar()
        || !matches!(
            &value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        )
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SAMPLE-RATE-CONTROL",
            format!("{label} must be a real integer scalar"),
            index,
        ));
        return None;
    }
    let literal = request.literals.literal_args.get(index)?;
    let parsed = structural_integer_literal(literal, positive)?;
    let Ok(parsed) = parsed else {
        diagnostics.push(argument_error(
            "RM-CATALOG-SAMPLE-RATE-CONTROL",
            format!("{label} is outside the supported integer range"),
            index,
        ));
        return None;
    };
    Some(parsed)
}

/// Preserve a typed integer's source-known bits at structural-control boundaries.
/// Non-numeric or unknown literals cannot prove an extent; numeric but invalid
/// literals are diagnosed by the caller. Float controls retain runtime's
/// binary64/platform-size policy.
pub(super) fn structural_integer_literal(
    literal: &LiteralValue,
    positive: bool,
) -> Option<Result<usize, ()>> {
    match literal {
        LiteralValue::Integer { .. } => Some(
            literal
                .integer_value_i128()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| !positive || *value > 0)
                .ok_or(()),
        ),
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            let value = LiteralContext::numeric_from_literal(literal)?;
            Some(
                (value.is_finite()
                    && value.fract() == 0.0
                    && value >= if positive { 1.0 } else { 0.0 }
                    && (value < usize::MAX as f64
                        || (usize::BITS < 64 && value == usize::MAX as f64)))
                    .then_some(value as usize)
                    .ok_or(()),
            )
        }
        _ => None,
    }
}

fn output(
    input: &ValueFact,
    factor: Option<usize>,
    phase: Option<usize>,
    direction: SampleDirection,
) -> ValueFact {
    let kind = match &input.kind {
        ValueKindFact::Numeric(fact) => ValueKindFact::Numeric(*fact),
        ValueKindFact::Logical if input.residency == ResidencyFact::Host => {
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        }
        ValueKindFact::Logical => return ValueFact::unknown(DynamicReason::RuntimeValue),
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => return ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let shape = transform_shape(&input.shape, factor, phase, direction);
    let host_scalar = matches!(&kind, ValueKindFact::Numeric(fact) if fact.class == NumericClass::Double || (fact.domain == NumericDomain::Real && fact.class.integer_class().is_some()));
    let storage = match shape.element_count() {
        Some(1) if input.residency == ResidencyFact::Host && host_scalar => StorageFact::Scalar,
        Some(1) if input.residency != ResidencyFact::Host => StorageFact::Unknown,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut result = ValueFact::proven(kind, shape, storage);
    if matches!(&result.shape, ShapeFact::Unknown)
        || matches!(&result.shape, ShapeFact::Shaped { dims } if dims.iter().any(|dim| matches!(dim, DimensionFact::Unknown)))
    {
        result.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    result.residency = if input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    result
}

fn transform_shape(
    shape: &ShapeFact,
    factor: Option<usize>,
    phase: Option<usize>,
    direction: SampleDirection,
) -> ShapeFact {
    let mut dims = match shape {
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Shaped { dims } if !dims.is_empty() => dims.clone(),
        ShapeFact::Shaped { .. } => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        _ => return ShapeFact::Unknown,
    };
    // A leading unknown extent may be singleton at runtime, in which case a
    // later non-singleton dimension becomes the operating axis. Preserve only
    // dimensions that are invariant across every possible selected axis.
    let mut candidates = Vec::new();
    let mut found_definite_axis = false;
    for (axis, dim) in dims.iter().enumerate() {
        match dim {
            DimensionFact::Known(1) => {}
            DimensionFact::Known(_) => {
                candidates.push(axis);
                found_definite_axis = true;
                break;
            }
            DimensionFact::Symbolic(_) | DimensionFact::Unknown => candidates.push(axis),
        }
    }
    if !found_definite_axis {
        candidates.push(0); // all extents may be singleton
    }
    candidates.sort_unstable();
    candidates.dedup();
    let unique_axis = candidates.len() == 1;
    for axis in candidates {
        let transformed = match (dims[axis].clone(), factor, phase) {
            (DimensionFact::Known(n), Some(f), _) if direction == SampleDirection::Up => n
                .checked_mul(f)
                .map(DimensionFact::Known)
                .unwrap_or(DimensionFact::Unknown),
            (DimensionFact::Known(n), Some(f), Some(p)) => {
                DimensionFact::Known(if p >= n { 0 } else { ((n - 1 - p) / f) + 1 })
            }
            _ => DimensionFact::Unknown,
        };
        if unique_axis {
            dims[axis] = transformed;
        } else if transformed != dims[axis] {
            dims[axis] = DimensionFact::Unknown;
        }
    }
    ShapeFact::Shaped { dims }
}
