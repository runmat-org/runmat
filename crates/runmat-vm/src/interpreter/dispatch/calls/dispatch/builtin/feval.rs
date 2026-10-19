use super::*;
use runmat_runtime::call::arguments::ArgumentSpec;

enum FevalArguments<'a> {
    Plain(usize),
    Expanded(&'a [ArgumentSpec]),
}

pub(super) async fn dispatch(
    ctx: CallDispatchContext<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let CallDispatchContext {
        instr,
        stack,
        vars,
        sequence_register,
        context,
        source_id,
        call_arg_spans,
        function_registry,
        ..
    } = ctx;
    // Resolve output-slot errors before consuming the argument stack or
    // suspending to gather an expanded argument.
    let (arguments, out_count, publication) = match instr {
        Instr::CallFevalMulti(argc, out_count) => (
            FevalArguments::Plain(*argc),
            *out_count,
            OutputPublication::Fixed(*out_count),
        ),
        Instr::CallFevalMultiUsingOutputSlot(argc, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            (
                FevalArguments::Plain(*argc),
                out_count,
                OutputPublication::Sequence,
            )
        }
        Instr::CallFevalExpandMultiOutput(specs, out_count) => (
            FevalArguments::Expanded(specs.as_slice()),
            *out_count,
            OutputPublication::Fixed(*out_count),
        ),
        Instr::CallFevalExpandMultiOutputUsingOutputSlot(specs, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars, *out_count_slot)?;
            (
                FevalArguments::Expanded(specs.as_slice()),
                out_count,
                OutputPublication::Sequence,
            )
        }
        _ => unreachable!("feval route received another instruction"),
    };
    let args = match arguments {
        FevalArguments::Plain(argc) => runmat_runtime::execution::RootedValues::new(
            crate::call::builtins::collect_call_args(stack, argc)?,
        )
        .map_err(|error| error.into_runtime_error())?,
        FevalArguments::Expanded(specs) => {
            build_feval_expand_multi_args(stack, specs, sequence_register, &context.runtime).await?
        }
    };
    let func_val = crate::interpreter::stack::pop_value(stack)?;
    let _function_input_callsite_guard =
        runmat_runtime::callsite::push_function_input_callsite_spans(
            source_id,
            call_arg_spans.as_ref().cloned(),
        )?;
    let crate::call::feval::FevalDispatch::Completed(sequence) = args
        .with_unbounded_values_async(|args| {
            crate::call::feval::execute_feval(
                &context.runtime,
                func_val,
                args,
                out_count,
                function_registry,
            )
        })
        .await??;
    match publication {
        OutputPublication::Fixed(count) => {
            calls::publish_rooted_fixed_outputs(stack, sequence, count)?
        }
        OutputPublication::Sequence => {
            sequence_register.stage_rooted_call_output_sequence(stack, sequence)?;
        }
    }
    Ok(Some(DispatchHandled::Generic(
        DispatchDecision::FallThrough,
    )))
}
