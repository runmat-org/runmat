pub use runmat_builtins::MAKE_ANON_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub async fn make_anon_builtin(params: String, body: String) -> crate::BuiltinResult<Value> {
    crate::make_anon_builtin(params, body).await
}

#[runtime_builtin(
    name = "make_anon",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::make_anon"
)]
pub async fn make_anon_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &MAKE_ANON_DESCRIPTOR,
        &runmat_builtins::MAKE_ANON_ERROR_OUTPUTS,
        "make_anon",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "make_anon",
            &runmat_builtins::MAKE_ANON_ERROR_ARITY,
        ));
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "make_anon: allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "make_anon: construction backing has no certified finite-resource bound",
        )
        .with_builtin("make_anon")
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let future = async {
        let first: String = (&args[0]).try_into()?;
        let second: String = (&args[1]).try_into()?;
        let result = make_anon_builtin(first, second).await;
        drop(args);
        result
    };
    match runtime.as_ref() {
        Some(context) => context.scope(future).await,
        None => future.await,
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::*;

    #[test]
    fn retained_ingress_preserves_exact_integer_text() {
        let values = [
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            Value::Bool(true),
        ];
        let args = crate::call::BuiltinArguments::from_borrowed(&values, None).unwrap();
        let output = futures::executor::block_on(make_anon_builtin_registered(None, args)).unwrap();
        let Value::String(runmat_value::StringElement::Text(text)) = output else {
            panic!("textual representation")
        };
        assert_eq!(text.as_str(), "@anon(18446744073709551615) true");
    }

    #[test]
    fn selected_cancellation_precedes_text_projection() {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        );
        let values = [Value::Num(1.0), Value::Num(2.0)];
        let args = crate::call::BuiltinArguments::from_borrowed(&values, Some(&runtime)).unwrap();
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error = futures::executor::block_on(make_anon_builtin_registered(Some(runtime), args))
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    }
}
