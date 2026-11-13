use runmat_value::Value;

pub(super) fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    target: Value,
    arguments: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _input = super::super::construction::input(runtime, &target)?;
    if arguments.len() > 1 {
        return Err(super::error::too_many_inputs());
    }
    let target = super::target::Target::parse(target)?;
    let original = target.source_order();
    let order = super::order::resolve(&original, arguments.first())?;
    target.validate_schema(order.as_slice())?;
    let ordered = target.reorder(runtime, order.as_slice())?;
    ordered.with_owned(|ordered| {
        let output =
            super::output::Evaluation::new(ordered, &original, order.as_slice())?.finish()?;
        super::super::construction::root(runtime, output)
    })
}
