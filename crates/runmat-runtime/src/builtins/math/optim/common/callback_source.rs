//! Borrowed source authority for an existing numerical invocation loop.
//! Dynamic is the supported ordinary call path; Selected retains the canonical
//! immutable descriptor chosen by its actual source-owning producer.
use crate::{
    call::{descriptor::RepeatableCallable, BuiltinArguments},
    context::RuntimeContext,
    execution::RootedValueSequence,
    BuiltinResult,
};
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(crate) enum Callable<'a> {
    Dynamic(&'a Value),
    Selected(&'a RepeatableCallable),
}
enum Prepared<'a, A> {
    Dynamic(&'a Value, A),
    Selected(&'a RepeatableCallable, BuiltinArguments, A),
}
impl<'source> Callable<'source> {
    pub(crate) fn invoke<'call, A: AsRef<[Value]> + 'call>(
        self,
        runtime: Option<&'call RuntimeContext>,
        arguments: A,
        outputs: usize,
    ) -> impl std::future::Future<Output = BuiltinResult<RootedValueSequence>>
           + 'call
           + use<'call, 'source, A>
    where
        'source: 'call,
    {
        crate::execution::after_result(
            async move {
                match self {
                    Self::Dynamic(value) => Ok(Prepared::Dynamic(value, arguments)),
                    Self::Selected(selected) => {
                        let retained =
                            BuiltinArguments::from_borrowed(arguments.as_ref(), runtime)?;
                        Ok::<_, crate::RuntimeError>(Prepared::Selected(
                            selected, retained, arguments,
                        ))
                    }
                }
            },
            move |prepared| match prepared {
                Prepared::Dynamic(value, arguments) => futures::future::Either::Left(
                    super::call_function_sequence_in_context(runtime, value, arguments, outputs),
                ),
                Prepared::Selected(selected, retained, arguments) => {
                    futures::future::Either::Right(async move {
                        // Preserve the original source carrier's lifetime as well
                        // as the canonical retained argument/root owner.
                        let _arguments = arguments;
                        selected.invoke_retained(runtime, retained, outputs).await
                    })
                }
            },
        )
    }
}
