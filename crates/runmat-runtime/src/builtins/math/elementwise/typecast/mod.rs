//! Exact byte reinterpretation for numeric and logical vectors.

mod error;
mod host;
mod invocation;
mod provider;
mod target;

use invocation::Invocation;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "typecast";

#[runtime_builtin(
    name = "typecast",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::typecast"
)]
async fn typecast_builtin_registered(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let invocation = Invocation::parse(args)?;
    if let Some(handle) = invocation.source_handle() {
        return provider::execute(handle, invocation.target).await;
    }
    host::reinterpret(invocation.source, invocation.target)
}

#[cfg(test)]
mod tests;
#[cfg(test)]
use tests::typecast_builtin;
