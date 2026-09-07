//! Exact byte reinterpretation for numeric and logical vectors.

mod error;
mod host;
mod provider;
mod target;

use runmat_builtins::TYPECAST_ERROR_INVALID_ARGUMENT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "typecast";

#[runtime_builtin(
    name = "typecast",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::typecast"
)]
async fn typecast_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let invocation = Invocation::parse(args)?;
    if let Some(handle) = invocation.source_handle() {
        return provider::execute(handle, invocation.target).await;
    }
    host::reinterpret(invocation.source, invocation.target)
}

struct Invocation {
    source: Value,
    target: target::OutputTarget,
}

impl Invocation {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        if !(2..=3).contains(&args.len()) {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "expected two or three inputs",
            ));
        }
        let mut args = args.into_iter();
        let source = args.next().expect("validated source");
        let selector = args.next().expect("validated selector");
        let target = match args.next() {
            Some(prototype) => {
                target::OutputTarget::from_prototype(&source, &selector, &prototype)?
            }
            None => target::OutputTarget::from_selector(&selector)?,
        };
        Ok(Self { source, target })
    }

    fn source_handle(&self) -> Option<runmat_accelerate_api::GpuTensorHandle> {
        match &self.source {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
