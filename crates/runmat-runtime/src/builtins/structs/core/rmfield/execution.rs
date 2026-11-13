use runmat_value::Value;

pub(super) fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    target: Value,
    fields: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let _input = super::super::construction::input(runtime, &target)?;
    let Some((first, rest)) = fields.view().split_first() else {
        return Err(super::error::not_enough_inputs());
    };
    let names = super::names::parse(first, rest)?;
    let names = names.as_slice();
    let target = super::target::Target::parse(target)?;
    if names.is_empty() {
        return super::super::construction::root(runtime, target.into_value()?);
    }
    match target {
        super::target::Target::Scalar(structure) => {
            for name in names {
                if !structure.fields.contains_key(name) {
                    return Err(super::error::missing_field(name));
                }
            }
            super::super::construction::scalar(
                runtime,
                structure
                    .fields
                    .iter()
                    .filter(|(name, _)| !names.iter().any(|removed| removed.as_str() == *name)),
            )
        }
        super::target::Target::Array(array) => {
            for name in names {
                if array.column_owner(name).is_none() {
                    return Err(super::error::missing_field(name));
                }
            }
            super::super::construction::array(
                runtime,
                array
                    .column_owners()
                    .filter(|(name, _)| !names.iter().any(|removed| removed.as_str() == *name)),
                array.shape_owner(),
            )
        }
    }
}
