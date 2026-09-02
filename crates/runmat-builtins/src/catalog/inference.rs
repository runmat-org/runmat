use super::{BuiltinCatalogEntry, BuiltinContractMaturity, BuiltinInferenceRule};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ValueFact,
    ValueKindFact,
};

mod acceleration_semantics;
mod aggregate_semantics;
mod array_semantics;
mod distributed_semantics;
mod introspection_semantics;
mod math_binary;
mod math_components;
mod math_degree_trigonometric;
mod math_exponential;
mod math_fact_transforms;
mod math_hyperbolic;
mod math_inverse;
mod math_logarithms;
mod math_roots;
mod math_rounding;
mod math_special;
mod math_trigonometric;
mod metadata_predicate;
mod numeric_abs;
mod numeric_classification;
mod numeric_component;
mod numeric_conversion;
mod numeric_limit;
mod parallel_semantics;
mod routing;
mod scalar_logical_reduction;
mod shape_predicate;
mod shape_scalar_query;
mod stats_random;
mod unary_logical_scalar;

pub use distributed_semantics::infer_partition_local_call;

pub fn infer_catalog_call(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let distributed = request.arguments.iter().find_map(|argument| {
        let ValueKindFact::Distributed(distributed) = &argument.kind else {
            return None;
        };
        Some(distributed.clone())
    });
    if let Some(distributed) = distributed {
        match entry.placement.distributed {
            crate::BuiltinDistributedPolicy::MapUnary => {
                return distributed_semantics::infer_distributed_map(entry, request, distributed);
            }
            crate::BuiltinDistributedPolicy::ScalarLikePrototype => {
                return distributed_semantics::infer_distributed_scalar_like(
                    entry,
                    request,
                    distributed,
                );
            }
            crate::BuiltinDistributedPolicy::MaterializeArguments => {
                return infer_partition_local_call(entry, request);
            }
            crate::BuiltinDistributedPolicy::Unsupported
            | crate::BuiltinDistributedPolicy::InspectHandles => {}
        }
    }
    infer_catalog_call_local(entry, request)
}

fn infer_catalog_call_local(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Array(rule) => routing::array::infer(rule, request, entry),
        BuiltinInferenceRule::Math(rule) => routing::math::infer(rule, request, entry),
        BuiltinInferenceRule::Stats(rule) => routing::stats::infer(rule, request, entry),
        BuiltinInferenceRule::Acceleration(rule) => {
            routing::acceleration::infer(rule, request, entry)
        }
        BuiltinInferenceRule::Aggregate(rule) => routing::aggregate::infer(rule, request, entry),
        BuiltinInferenceRule::Introspection(rule) => {
            routing::introspection::infer(rule, request, entry)
        }
        BuiltinInferenceRule::Logical(rule) => routing::logical::infer(rule, request, entry),
        BuiltinInferenceRule::Parallel(rule) => routing::parallel::infer(rule, request, entry),
    }
}

fn literal_text(literal: &LiteralValue) -> Option<String> {
    match literal {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => Some(value.clone()),
        _ => None,
    }
}

fn numeric_kind(class: NumericClass, domain: NumericDomain) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact { class, domain })
}

fn preserved_binary_residency(left: &ResidencyFact, right: &ResidencyFact) -> ResidencyFact {
    match (left, right) {
        (ResidencyFact::Host, ResidencyFact::Host) => ResidencyFact::Host,
        (
            ResidencyFact::Device {
                provider: left_owner,
            },
            ResidencyFact::Device {
                provider: right_owner,
            },
        ) if left_owner == right_owner => left.clone(),
        (ResidencyFact::Device { .. }, ResidencyFact::Host) => left.clone(),
        (ResidencyFact::Host, ResidencyFact::Device { .. }) => right.clone(),
        _ => ResidencyFact::Unknown,
    }
}

fn default_double_scalar() -> ValueFact {
    ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real))
}

fn unavailable_rule(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::UnsupportedRepresentation);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    if matches!(entry.contract.maturity, BuiltinContractMaturity::Complete) {
        inference.diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-INFERENCE-RULE",
            format!(
                "complete builtin contract `{:?}` has no registered inference rule",
                entry.contract.inference_rule
            ),
        ));
    }
    inference
}

fn finish_fixed(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    output: ValueFact,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn argument_error(
    code: impl Into<String>,
    message: impl Into<String>,
    argument: usize,
) -> InferenceDiagnostic {
    let mut diagnostic = InferenceDiagnostic::error(code, message);
    diagnostic.argument = Some(argument);
    diagnostic
}
