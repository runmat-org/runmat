use runmat_value::{StructValue, Value};

pub(super) fn execute(target: Value, fields: Vec<Value>) -> crate::BuiltinResult<Value> {
    let Some((first, rest)) = fields.split_first() else {
        return Err(super::error::not_enough_inputs());
    };
    let names = super::names::parse(first, rest)?;
    let target = super::target::Target::parse(target)?;
    if names.is_empty() {
        return target.into_value();
    }
    match target {
        super::target::Target::Scalar(mut structure) => {
            remove(&mut structure, names.as_slice())?;
            Ok(Value::Struct(structure))
        }
        super::target::Target::Array(mut array) => {
            remove_array(&mut array, names.as_slice())?;
            Ok(Value::StructArray(array))
        }
    }
}

fn remove_array(
    array: &mut runmat_value::StructArray,
    names: &[String],
) -> crate::BuiltinResult<()> {
    for name in names {
        if array.remove_field(name).is_none() {
            return Err(super::error::missing_field(name));
        }
    }
    Ok(())
}

fn remove(structure: &mut StructValue, names: &[String]) -> crate::BuiltinResult<()> {
    for name in names {
        if structure.remove(name).is_none() {
            return Err(super::error::missing_field(name));
        }
    }
    Ok(())
}
