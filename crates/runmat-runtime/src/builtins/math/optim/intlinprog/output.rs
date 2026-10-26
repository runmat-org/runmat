//! Raw value presentation delegates to the same typed MILP publication owner.
use super::*;
use engine::{Stop, Terminal};
use publication::{nonnegative, number, BinaryPoint};
mod prepare;
pub(super) use prepare::prepare;
pub(in crate::builtins::math::optim) struct Published {
    pub x: Option<Vec<f64>>,
    pub fval: Option<f64>,
    pub flag: i32,
    pub information: Option<runmat_value::StructValue>,
}
pub(super) async fn finalize(
    outcome: engine::Outcome,
    workspace: &model::Workspace,
    controls: &controls::Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    let count = requested.unwrap_or(1);
    let Published {
        x,
        fval,
        flag,
        information,
    } = prepare(
        outcome,
        workspace,
        controls,
        count >= 4,
        &mut compiled::Supplied,
        cooperation,
    )
    .await?;
    if count == 0 {
        return Ok(Value::output_list(Vec::new()));
    }
    let x = x
        .map(|values| {
            let n = values.len();
            Tensor::new(values, vec![n, 1])
                .map(Value::Tensor)
                .map_err(invalid)
        })
        .transpose()?
        .unwrap_or_else(empty);
    if requested.is_none() {
        return Ok(x);
    }
    let mut outputs = vec![x];
    if count >= 2 {
        outputs.push(fval.map(Value::Num).unwrap_or_else(empty));
    }
    if count >= 3 {
        outputs.push(Value::Num(flag as f64));
    }
    if let Some(info) = information {
        outputs.push(Value::Struct(info));
    }
    Ok(crate::output_count::output_list_with_padding(
        count, outputs,
    ))
}
