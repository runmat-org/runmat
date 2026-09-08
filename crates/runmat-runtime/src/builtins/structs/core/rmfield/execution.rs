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
            array.into_cell().map(Value::Cell)
        }
    }
}

fn remove_array(
    array: &mut super::target::StructArray,
    names: &[String],
) -> crate::BuiltinResult<()> {
    validate_array(array, names)?;
    for structure in &mut array.elements {
        remove_known(structure, names);
    }
    Ok(())
}

fn validate_array(
    array: &super::target::StructArray,
    names: &[String],
) -> crate::BuiltinResult<()> {
    for name in names {
        for structure in &array.elements {
            if !structure.fields.contains_key(name) {
                return Err(super::error::missing_field(name));
            }
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

fn remove_known(structure: &mut StructValue, names: &[String]) {
    for name in names {
        let removed = structure.remove(name);
        debug_assert!(removed.is_some());
    }
}
