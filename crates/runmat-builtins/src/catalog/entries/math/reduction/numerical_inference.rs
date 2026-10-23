//! Facts shared by sampled-data transforms and the host count boundary.
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

mod difference;
mod gradient;
mod integration;
mod nonzero;

pub(super) use difference::infer as difference;
pub(super) use gradient::infer as gradient;
pub(super) use integration::{infer as integration, Mode as IntegrationMode};
pub(super) use nonzero::infer as nonzero;

fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn invalid(entry: &BuiltinCatalogEntry, detail: &str) -> InferenceDiagnostic {
    InferenceDiagnostic::error(entry.descriptor.errors[0].code, detail)
}
fn invalid_input(
    kind: super::numerical_operation::NumericalReductionKind,
    detail: &str,
) -> InferenceDiagnostic {
    InferenceDiagnostic::error(kind.input_error().code, detail)
}

fn numeric_output(input: &ValueFact, shape: ShapeFact) -> ValueFact {
    let kind = match input.kind {
        ValueKindFact::Numeric(NumericFact { class, domain }) => {
            ValueKindFact::Numeric(NumericFact {
                class: if class == NumericClass::Single {
                    class
                } else {
                    NumericClass::Double
                },
                domain,
            })
        }
        ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => return ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let storage = if shape.element_count() == Some(1)
        && matches!(
            kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                ..
            })
        ) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let mut output = ValueFact::proven(kind, shape, storage);
    if input.residency != ResidencyFact::Host {
        output.residency = ResidencyFact::Unknown;
        output.storage = StorageFact::Unknown;
    }
    output
}

fn dense_samples(input: &ValueFact) -> bool {
    input.storage != StorageFact::Sparse
        && match input.kind {
            ValueKindFact::Numeric(n) => {
                n.domain != NumericDomain::Complex || n.class.integer_class().is_none()
            }
            ValueKindFact::Logical => true,
            _ => false,
        }
}

fn literal(request: &CallRequest, position: usize) -> Option<&LiteralValue> {
    request.literals.literal_args.get(position)
}

fn positive_axis(value: &LiteralValue) -> Option<usize> {
    if let Some(value) = value.integer_value_i128() {
        return usize::try_from(value).ok().filter(|value| *value > 0);
    }
    let value = crate::ResolveContext::numeric_from_literal(value)?;
    runmat_types::dimension::decode_scalar_dimension(value, false).ok()
}

fn default_axis(shape: &ShapeFact, include_zero: bool) -> Option<usize> {
    let dimensions = shape.known_dims()?;
    for (axis, extent) in dimensions.into_iter().enumerate() {
        let extent = extent?;
        if (include_zero && extent != 1) || (!include_zero && extent > 1) {
            return Some(axis + 1);
        }
    }
    Some(1)
}

fn reduced_shape(shape: &ShapeFact, axis: Option<usize>) -> ShapeFact {
    let Some(axis) = axis else {
        return ShapeFact::Unknown;
    };
    let Some(mut dimensions) = shape.known_dims() else {
        return shape
            .rank()
            .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank });
    };
    if let Some(extent) = dimensions.get_mut(axis - 1) {
        *extent = Some(1);
    }
    ShapeFact::from(dimensions)
}
