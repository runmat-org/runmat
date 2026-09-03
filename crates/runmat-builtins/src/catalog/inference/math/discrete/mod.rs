use crate::{BuiltinCatalogEntry, DiscreteInferenceRule};
use runmat_types::{CallInference, CallRequest};

mod binary;
mod factor;
mod factorial;
mod isprime;
mod primes;

#[cfg(test)]
mod primes_tests;

pub(in crate::catalog::inference) fn infer(
    rule: DiscreteInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        DiscreteInferenceRule::Binary(operation) => binary::infer(operation, request, entry),
        DiscreteInferenceRule::Factor => factor::infer(request, entry),
        DiscreteInferenceRule::Factorial => factorial::infer(request, entry),
        DiscreteInferenceRule::IsPrime => isprime::infer(request, entry),
        DiscreteInferenceRule::Primes => primes::infer(request, entry),
    }
}
