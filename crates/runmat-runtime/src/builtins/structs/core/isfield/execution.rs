use runmat_value::Value;

pub(super) fn execute(target: &Value, names: Value) -> crate::BuiltinResult<Value> {
    let query = super::names::parse(names)?;
    let fields = super::target::common_fields(target);
    super::output::evaluate(query, fields.as_ref())
}
