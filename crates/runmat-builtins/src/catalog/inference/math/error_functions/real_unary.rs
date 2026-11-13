use crate::{BuiltinCatalogEntry, ErrorFunctionInferenceRule};
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    NumericClass, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

use super::super::super::{argument_error, finish_fixed, numeric_kind};

struct ErrorFunctionSpec {
    name: &'static str,
    arity_code: &'static str,
    input_code: &'static str,
}

const fn spec(rule: ErrorFunctionInferenceRule) -> ErrorFunctionSpec {
    match rule {
        ErrorFunctionInferenceRule::Erf => ErrorFunctionSpec {
            name: "erf",
            arity_code: "RM-CATALOG-ERF-ARITY",
            input_code: "RM-CATALOG-ERF-INPUT",
        },
        ErrorFunctionInferenceRule::InverseComplementary => ErrorFunctionSpec {
            name: "erfcinv",
            arity_code: "RM-CATALOG-ERFCINV-ARITY",
            input_code: "RM-CATALOG-ERFCINV-INPUT",
        },
    }
}

pub(super) fn infer(
    rule: ErrorFunctionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let spec = spec(rule);
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            spec.arity_code,
            format!("{} requires exactly one input", spec.name),
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
        if matches!(input.kind, ValueKindFact::Unknown) {
            let mut output = input.clone();
            output.alias = AliasFact::Unique;
            output.view = ViewFact::Materialized;
            output.mutation = MutationFact::ValueSemantics;
            return finish_fixed(entry, request, output, diagnostics);
        }
        diagnostics.push(argument_error(
            spec.input_code,
            format!("{} requires dense real single or double input", spec.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    };

    if numeric.domain != NumericDomain::Real
        || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        || matches!(input.storage, StorageFact::Sparse)
    {
        diagnostics.push(argument_error(
            spec.input_code,
            format!("{} requires dense real single or double input", spec.name),
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
    use super::super::super::super::infer_catalog_call;
    use crate::{
        builtin_catalog_entry_by_name, BuiltinInferenceRule, ErrorFunctionInferenceRule,
        MathInferenceRule,
    };
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
    fn error_functions_preserve_float_facts_and_reject_other_domains() {
        for (name, rule) in [
            ("erf", ErrorFunctionInferenceRule::Erf),
            ("erfcinv", ErrorFunctionInferenceRule::InverseComplementary),
        ] {
            let entry = builtin_catalog_entry_by_name(name).expect("catalog entry");
            assert_eq!(
                entry.contract.inference_rule,
                BuiltinInferenceRule::Math(MathInferenceRule::ErrorFunction(rule))
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
            assert!(inferred.diagnostics.is_empty(), "{name}");
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

            for unsupported in [
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
                assert!(
                    !infer_catalog_call(entry, &request(unsupported))
                        .diagnostics
                        .is_empty(),
                    "{name}"
                );
            }
        }
    }
}
