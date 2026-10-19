//! Physical scratch for kernels that consume a preallocated numerical output.
//!
//! Legacy and compound host bodies retain the conservative scratch bound in
//! their canonical resource callback; this module describes only the shared
//! whole-owner execution path.
use crate::builtins::common::spec::FusionHostResourceQuery;
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::NumericClass;

fn real_floating(class: NumericClass) -> bool {
    matches!(class, NumericClass::Single | NumericClass::Double)
}

/// Exact temporary storage used when the selected host body consumes a
/// preallocated numerical output.
pub fn scratch_bytes(
    operation: BuiltinPreparationOperation,
    query: &FusionHostResourceQuery<'_>,
) -> Option<u64> {
    match operation {
        BuiltinPreparationOperation::Unary | BuiltinPreparationOperation::RealUnary(_) => {
            let [input] = query.inputs else { return None };
            if input.class != query.output.class || !real_floating(input.class) {
                return None;
            }
        }
        BuiltinPreparationOperation::FloatingCast(target) => {
            let [input] = query.inputs else { return None };
            if target != query.output.class
                || !real_floating(input.class)
                || !real_floating(query.output.class)
            {
                return None;
            }
        }
        BuiltinPreparationOperation::ElementwiseBinary | BuiltinPreparationOperation::RealPower => {
            let [left, right] = query.inputs else {
                return None;
            };
            if !real_floating(left.class)
                || !real_floating(right.class)
                || !real_floating(query.output.class)
            {
                return None;
            }
        }
        _ => return None,
    }
    scratch_for_shape_inputs(
        operation,
        query.inputs.len(),
        query.inputs.first().map(|input| input.shape),
        query.inputs.get(1).map(|input| input.shape),
        query.output.shape,
    )
}

fn scratch_for_shape_inputs(
    operation: BuiltinPreparationOperation,
    input_count: usize,
    first: Option<&[usize]>,
    second: Option<&[usize]>,
    output: &[usize],
) -> Option<u64> {
    match operation {
        BuiltinPreparationOperation::Unary
        | BuiltinPreparationOperation::FloatingCast(_)
        | BuiltinPreparationOperation::RealUnary(_)
            if input_count == 1 && first? == output =>
        {
            Some(0)
        }
        BuiltinPreparationOperation::ElementwiseBinary | BuiltinPreparationOperation::RealPower
            if input_count == 2 =>
        {
            crate::builtins::common::broadcast::PreparedBroadcastPlan::quoted_for_shapes(
                first?, second?, output,
            )
            .ok()
        }
        _ => None,
    }
}

pub(crate) fn scratch_for_shapes(
    operation: BuiltinPreparationOperation,
    inputs: &[Vec<usize>],
    output: &[usize],
) -> Option<u64> {
    scratch_for_shape_inputs(
        operation,
        inputs.len(),
        inputs.first().map(Vec::as_slice),
        inputs.get(1).map(Vec::as_slice),
        output,
    )
}
