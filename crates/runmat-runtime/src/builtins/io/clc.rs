//! Clear-console control event.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::{build_runtime_error, console, BuiltinResult};

const BUILTIN_NAME: &str = "clc";

#[runtime_builtin(
    name = "clc",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::clc"
)]
async fn clc_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(argument_count_error());
    }
    console::record_clear_screen();
    Ok(empty_return_value())
}

fn argument_count_error() -> crate::RuntimeError {
    let descriptor: &'static BuiltinErrorDescriptor = &runmat_builtins::CLC_ERROR_ARG_COUNT;
    let mut builder = build_runtime_error(descriptor.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleStream;

    #[test]
    fn records_clear_event_and_returns_empty_sink_value() {
        console::reset_thread_buffer();
        let result = futures::executor::block_on(clc_builtin(Vec::new())).expect("clc");
        let Value::Tensor(empty) = result else {
            panic!("expected empty tensor")
        };
        assert_eq!(empty.shape, vec![0, 0]);
        let entries = console::take_thread_buffer();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].stream, ConsoleStream::ClearScreen);
        assert!(entries[0].text.is_empty());
    }

    #[test]
    fn rejects_input_without_emitting_clear_event() {
        console::reset_thread_buffer();
        let error = futures::executor::block_on(clc_builtin(vec![Value::Num(1.0)]))
            .expect_err("clc input must fail");
        assert_eq!(error.message(), "clc: expected no input arguments");
        assert_eq!(error.identifier(), Some("RunMat:clc:ArgumentCount"));
        assert!(console::take_thread_buffer().is_empty());
    }
}
