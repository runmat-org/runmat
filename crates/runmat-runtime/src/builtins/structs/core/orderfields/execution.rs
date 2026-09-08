use runmat_value::Value;

pub(super) fn execute(target: Value, arguments: &[Value]) -> crate::BuiltinResult<Value> {
    if arguments.len() > 1 {
        return Err(super::error::too_many_inputs());
    }
    let target = super::target::Target::parse(target)?;
    let original = target.source_order();
    let order = super::order::resolve(&original, arguments.first())?;
    target.validate_schema(order.as_slice())?;
    let ordered = target.reorder(order.as_slice())?;
    super::output::Evaluation::new(ordered, &original, order.as_slice())?.finish()
}
