use super::*;
use crate::fusion::FusionOp;
use crate::graph::PrimitiveOp;
use runmat_accelerate_api::{MatmulEpilogue, ScaleOp};
use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, FusionStructuralOperation,
};

pub(super) struct AdmittedMatmul {
    pub left: ValueId,
    pub right: ValueId,
    pub shape: [usize; 2],
    alpha: f64,
    beta: f64,
    row: Option<(ValueId, ScaleOp)>,
    column: Option<(ValueId, ScaleOp)>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Product,
    Scale,
    Bias,
    Row,
    Column,
}

fn matrix_root(operation: &FusionOp) -> Option<(ValueId, ValueId, ValueId)> {
    let FusionOp::Builtin {
        name,
        inputs,
        output: Some(output),
    } = operation
    else {
        return None;
    };
    let entry = runmat_builtins::builtin_catalog_entry_by_name(name)?;
    let [binding] = entry.bindings else {
        return None;
    };
    let declaration = canonical_fusion_declaration(entry.binding_identity(binding))?;
    if declaration.structural != Some(FusionStructuralOperation::MatrixMultiply) {
        return None;
    }
    let [left, right] = inputs.as_slice() else {
        return None;
    };
    Some((*left, *right, *output))
}

impl AdmittedMatmul {
    pub fn value<'a>(
        &self,
        request: &'a FusionExecutionRequest<'_>,
        id: ValueId,
    ) -> Option<&'a Value> {
        request
            .plan
            .inputs
            .iter()
            .position(|input| *input == id)
            .and_then(|index| request.inputs.get(index))
            .or_else(|| request.plan.const_values.get(&id))
    }

    pub fn new(
        request: &FusionExecutionRequest<'_>,
        provider: &'static dyn AccelProvider,
    ) -> Option<Self> {
        if request.plan.group.kind != FusionKind::MatmulEpilogue
            || request.inputs.len() != request.plan.inputs.len()
        {
            return None;
        }
        let (root, operations) = request.plan.operations.split_first()?;
        let (left, right, mut current) = matrix_root(root)?;
        let mut result = Self {
            left,
            right,
            shape: [0, 0],
            alpha: 1.0,
            beta: 0.0,
            row: None,
            column: None,
        };
        let a = representation::shape(result.value(request, left)?, provider)?;
        let b = representation::shape(result.value(request, right)?, provider)?;
        if a[1] != b[0] || a[0].checked_mul(a[1])? == 1 || b[0].checked_mul(b[1])? == 1 {
            return None;
        }
        result.shape = [a[0], b[1]];
        a[0].checked_mul(b[1])?;
        // The provider executes fixed phases. Never reorder or algebraically
        // fold source operations across them, including floating-point rounds.
        let mut phase = Phase::Product;
        for operation in operations {
            let FusionOp::Primitive {
                op,
                inputs,
                output: Some(output),
            } = operation
            else {
                return None;
            };
            let [first, second] = inputs.as_slice() else {
                return None;
            };
            let (other, current_left) = if *first == current && *second != current {
                (*second, true)
            } else if *second == current && *first != current {
                (*first, false)
            } else {
                return None;
            };
            let value = result.value(request, other)?;
            if let Some(scalar) = representation::scalar(value, provider.precision()) {
                match op {
                    PrimitiveOp::Mul | PrimitiveOp::ElemMul if phase == Phase::Product => {
                        result.alpha = scalar;
                        phase = Phase::Scale;
                    }
                    PrimitiveOp::Add if phase <= Phase::Scale => {
                        result.beta = scalar;
                        phase = Phase::Bias;
                    }
                    PrimitiveOp::Sub if phase <= Phase::Scale && current_left => {
                        result.beta = -scalar;
                        phase = Phase::Bias;
                    }
                    PrimitiveOp::Sub if phase == Phase::Product && !current_left => {
                        result.alpha = -1.0;
                        result.beta = scalar;
                        phase = Phase::Bias;
                    }
                    _ => return None,
                }
            } else {
                let shape = representation::shape(value, provider)?;
                let operation = match op {
                    PrimitiveOp::ElemMul => ScaleOp::Multiply,
                    PrimitiveOp::ElemDiv if current_left => ScaleOp::Divide,
                    _ => return None,
                };
                if shape == [result.shape[0], 1] && phase <= Phase::Bias {
                    result.row = Some((other, operation));
                    phase = Phase::Row;
                } else if shape == [1, result.shape[1]] && phase <= Phase::Row {
                    result.column = Some((other, operation));
                    phase = Phase::Column;
                } else {
                    return None;
                }
            }
            current = *output;
        }
        (request.plan.output == Some(current)).then_some(result)
    }

    pub fn data_inputs(&self) -> impl Iterator<Item = ValueId> {
        [
            Some(self.left),
            Some(self.right),
            self.row.map(|(id, _)| id),
            self.column.map(|(id, _)| id),
        ]
        .into_iter()
        .flatten()
    }

    pub fn descriptor(&self, handles: &HashMap<ValueId, GpuTensorHandle>) -> MatmulEpilogue {
        let mut descriptor = MatmulEpilogue::noop();
        descriptor.alpha = self.alpha;
        descriptor.beta = self.beta;
        if let Some((id, operation)) = self.row {
            descriptor.row_scale = Some(handles[&id].clone());
            descriptor.row_op = operation;
        }
        if let Some((id, operation)) = self.column {
            descriptor.col_scale = Some(handles[&id].clone());
            descriptor.col_op = operation;
        }
        descriptor
    }
}
