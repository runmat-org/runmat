//! Keep argument placement alive through the runtime's complete builtin call.
use runmat_runtime::{context::RuntimeContext, RuntimeError};
use runmat_value::Value;
#[cfg(test)]
mod tests;

#[cfg(feature = "native-accel")]
pub use runmat_accelerate::native_auto::PreparedBuiltinCall;

#[cfg(not(feature = "native-accel"))]
pub struct PreparedBuiltinCall(Vec<Value>);

#[cfg(not(feature = "native-accel"))]
impl PreparedBuiltinCall {
    pub fn args(&self) -> &[Value] {
        &self.0
    }
}

pub async fn prepare_builtin_args(
    name: &str,
    args: &[Value],
    runtime: &RuntimeContext,
) -> Result<PreparedBuiltinCall, RuntimeError> {
    #[cfg(feature = "native-accel")]
    {
        runmat_accelerate::native_auto::prepare_builtin_call(name, args, runtime)
            .await
            .map_err(super::map_prepare_builtin_args_error)
    }
    #[cfg(not(feature = "native-accel"))]
    {
        let _ = (name, runtime);
        Ok(PreparedBuiltinCall(args.to_vec()))
    }
}

pub fn call_prepared_builtin(
    name: &str,
    prepared: PreparedBuiltinCall,
    outputs: usize,
) -> impl std::future::Future<Output = Result<Value, RuntimeError>> + '_ {
    #[cfg(feature = "native-accel")]
    {
        prepared.execute(move |args| async move {
            runmat_runtime::call_builtin_async_with_outputs(name, &args, outputs).await
        })
    }
    #[cfg(not(feature = "native-accel"))]
    {
        let args = prepared.0;
        async move { runmat_runtime::call_builtin_async_with_outputs(name, &args, outputs).await }
    }
}
