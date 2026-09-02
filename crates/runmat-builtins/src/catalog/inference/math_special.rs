use super::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    NumericClass, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

pub(super) fn infer_gamma(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMMA-ARITY",
            "gamma requires exactly one input",
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let ValueKindFact::Numeric(numeric) = input.kind else {
        let reason = if matches!(input.kind, ValueKindFact::Unknown) {
            DynamicReason::RuntimeValue
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-GAMMA-INPUT",
                "gamma requires real single or double input",
                0,
            ));
            DynamicReason::UnsupportedRepresentation
        };
        return finish_fixed(entry, request, ValueFact::unknown(reason), diagnostics);
    };
    if numeric.domain != NumericDomain::Real
        || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        || matches!(input.storage, StorageFact::Sparse)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GAMMA-INPUT",
            "gamma requires dense real single or double input",
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
    output.kind = numeric_kind(numeric.class, NumericDomain::Real);
    output.storage = if input.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.residency = match input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { .. } => input.residency.clone(),
        _ => ResidencyFact::Unknown,
    };
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::super::infer_catalog_call;
    use crate::{builtin_catalog_entry_by_name, BuiltinInferenceRule, MathInferenceRule};
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    fn request(input: ValueFact) -> CallRequest {
        CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn gamma_preserves_float_class_and_shape_and_rejects_other_domains() {
        let entry = builtin_catalog_entry_by_name("gamma").expect("gamma catalog entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::Gamma)
        );
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let inferred = infer_catalog_call(entry, &request(input));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );

        for input in [
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int32,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
            ValueFact::scalar(ValueKindFact::Logical),
        ] {
            assert!(!infer_catalog_call(entry, &request(input))
                .diagnostics
                .is_empty());
        }
    }
}
