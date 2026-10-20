use super::{error, target};
use crate::BuiltinResult;
use runmat_builtins::TYPECAST_ERROR_INVALID_ARGUMENT;
use runmat_value::Value;

pub(super) struct Invocation {
    pub(super) source: Value,
    pub(super) target: target::OutputTarget,
}

impl Invocation {
    pub(super) fn parse(mut args: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if !(2..=3).contains(&args.len()) {
            return Err(error::build(
                &TYPECAST_ERROR_INVALID_ARGUMENT,
                "expected two or three inputs",
            ));
        }
        let source = args.pop_front().expect("validated source");
        let selector = args.pop_front().expect("validated selector");
        let target = match args.pop_front() {
            Some(prototype) => {
                target::OutputTarget::from_prototype(&source, &selector, &prototype)?
            }
            None => target::OutputTarget::from_selector(&selector)?,
        };
        Ok(Self { source, target })
    }

    pub(super) fn source_handle(&self) -> Option<runmat_accelerate_api::GpuTensorHandle> {
        match &self.source {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }
    }
}
