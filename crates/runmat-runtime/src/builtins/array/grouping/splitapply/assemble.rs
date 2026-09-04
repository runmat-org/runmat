use runmat_value::Value;

use crate::BuiltinResult;

use super::error;

pub(super) async fn outputs(groups: Vec<Vec<Value>>) -> BuiltinResult<Vec<Value>> {
    let mut outputs = Vec::with_capacity(groups.len());
    for values in groups {
        let output = crate::call_builtin_async("vertcat", &values)
            .await
            .map_err(|source| {
                error::output(
                    "splitapply: group results must have compatible classes and dimensions",
                    Some(source),
                )
            })?;
        outputs.push(output);
    }
    Ok(outputs)
}
