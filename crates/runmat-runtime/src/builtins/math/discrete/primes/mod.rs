//! MATLAB-compatible `primes` execution.

use runmat_builtins::PRIMES_ERROR_LIMIT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use self::arguments::{parse_request, MAX_PRIMES_LIMIT};
use self::error::primes_error;
use crate::BuiltinResult;

mod arguments;
mod error;
mod evaluation;
mod sieve;

#[runtime_builtin(
    name = "primes",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::discrete::primes"
)]
async fn primes_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(primes_error(
            &runmat_builtins::PRIMES_ERROR_INVALID_INPUT,
            "expected exactly one input argument",
        ));
    }
    let request = parse_request(value).await?;
    if request.limit > MAX_PRIMES_LIMIT {
        return Err(primes_error(
            &PRIMES_ERROR_LIMIT,
            format!("n must be <= {MAX_PRIMES_LIMIT}"),
        ));
    }
    evaluation::evaluate(request)
}

#[cfg(test)]
mod tests;
