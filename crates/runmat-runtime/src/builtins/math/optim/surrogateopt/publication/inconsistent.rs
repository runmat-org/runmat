//! Empty-domain termination needs no objective, RNG, Clock, UI or pool.
use super::*;
pub(in super::super) async fn publish(
    variables: usize,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let requested = crate::output_count::current_output_count_in(c.runtime());
    let count = requested.unwrap_or(1);
    if count > 5 {
        return Err(super::super::invalid("at most five outputs are available"));
    }
    let mut outputs = values::capacity(count)?;
    if count > 0 {
        outputs.push(values::empty(c)?);
    }
    if count > 1 {
        outputs.push(values::empty(c)?);
    }
    if count > 2 {
        outputs.push(Value::Num(-2.0));
    }
    if count > 3 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("funccount", Value::Num(0.0));
        fields.insert("elapsedtime", Value::Num(0.0));
        fields.insert(
            "message",
            Value::string(crate::call::argument_text::copy_str(
                "The lower and upper bounds are inconsistent.",
                c.runtime(),
            )?),
        );
        for name in ["constrviolation", "ineq", "rngstate"] {
            fields.insert(name, values::empty(c)?);
        }
        outputs.push(Value::Struct(fields.finish()));
    }
    if count > 4 {
        outputs.push(values::trials(&super::super::history::History::new(variables), c).await?);
    }
    root(outputs, requested, c)
}
