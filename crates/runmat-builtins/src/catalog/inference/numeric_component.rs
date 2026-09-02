use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, NumericComponentRule};
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    NumericClass, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer_numeric_component_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rule: NumericComponentRule,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-COMPONENT-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-COMPONENT-ARITY",
            format!("{} accepts exactly one input", entry.identity.name),
            1,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-COMPONENT-SPARSE",
            format!(
                "{} does not currently accept sparse input",
                entry.identity.name
            ),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let mut materializes = false;
    let mut changes_class = false;
    match (&mut output.kind, rule) {
        (ValueKindFact::Numeric(numeric), NumericComponentRule::Conjugate) => {
            materializes = matches!(numeric.domain, NumericDomain::Complex);
        }
        (ValueKindFact::Numeric(numeric), NumericComponentRule::RealPart) => {
            materializes = matches!(numeric.domain, NumericDomain::Complex);
            numeric.domain = NumericDomain::Real;
        }
        (ValueKindFact::Numeric(numeric), NumericComponentRule::ImaginaryPart) => {
            materializes = true;
            numeric.domain = NumericDomain::Real;
        }
        (ValueKindFact::Logical, NumericComponentRule::Conjugate) => {}
        (
            ValueKindFact::Logical | ValueKindFact::Character,
            NumericComponentRule::RealPart | NumericComponentRule::ImaginaryPart,
        )
        | (ValueKindFact::Character, NumericComponentRule::Conjugate) => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materializes = true;
            changes_class = true;
        }
        (ValueKindFact::Unknown, _) => {
            let shape = output.shape.clone();
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = shape;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-COMPONENT-INPUT",
                format!(
                    "{} requires numeric, logical, or character input",
                    entry.identity.name
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    if materializes {
        output.storage = if output.is_scalar() {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        };
        output.layout = LayoutFact::ColumnMajor;
        output.contiguity = ContiguityFact::Contiguous;
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }

    finish_fixed(entry, request, output, diagnostics)
}
