use super::*;
use crate::fusion::FusionOp;
use crate::graph::PrimitiveOp;
use runmat_accelerate_api::{MatmulAffine, MatmulEpilogue, MatmulOffset, ScaleOp};
use runmat_runtime::builtins::common::spec::{FusionEpilogueOperation, FusionStructuralOperation};

pub(super) struct AdmittedMatmul {
    pub left: ValueId,
    pub right: ValueId,
    pub shape: [usize; 2],
    scale: Option<f64>,
    offset: Option<MatmulOffset>,
    pub(super) row: Option<(ValueId, ScaleOp)>,
    pub(super) column: Option<(ValueId, ScaleOp)>,
    clamp_min: Option<f64>,
    clamp_max: Option<f64>,
    exponent: Option<f64>,
    pub diagonal: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Product,
    Scale,
    Bias,
    Row,
    Column,
    LowerClamp,
    UpperClamp,
    Power,
    Diagonal,
}

fn matrix_root(
    operation: &FusionOp,
    bindings: &AdmittedBindings,
) -> Option<(ValueId, ValueId, ValueId)> {
    let FusionOp::Builtin {
        inputs,
        output: Some(output),
        ..
    } = operation
    else {
        return None;
    };
    if bindings.structural(0) != Some(FusionStructuralOperation::MatrixMultiply) {
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
        bindings: &AdmittedBindings,
    ) -> Option<Self> {
        if request.plan.group.kind != FusionKind::MatmulEpilogue
            || request.inputs.len() != request.plan.inputs.len()
        {
            return None;
        }
        let (root, operations) = request.plan.operations.split_first()?;
        let (left, right, mut current) = matrix_root(root, bindings)?;
        let mut result = Self {
            left,
            right,
            shape: [0, 0],
            scale: None,
            offset: None,
            row: None,
            column: None,
            clamp_min: None,
            clamp_max: None,
            exponent: None,
            diagonal: false,
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
        for (index, operation) in operations.iter().enumerate() {
            let epilogue = match operation {
                FusionOp::Builtin {
                    inputs,
                    output: Some(output),
                    ..
                } => Some((bindings.epilogue(index + 1)?, inputs, output)),
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    inputs,
                    output: Some(output),
                } => Some((bindings.epilogue(index + 1)?, inputs, output)),
                _ => None,
            };
            if let Some((kind, inputs, output)) = epilogue {
                if inputs.len() != kind.input_arity() {
                    return None;
                }
                if kind == FusionEpilogueOperation::Diagonal {
                    if inputs != &[current] || phase >= Phase::Diagonal || result.shape.contains(&1)
                    {
                        return None;
                    }
                    result.diagonal = true;
                    phase = Phase::Diagonal;
                } else {
                    let (other, current_left) = if inputs[0] == current && inputs[1] != current {
                        (inputs[1], true)
                    } else if inputs[1] == current && inputs[0] != current {
                        (inputs[0], false)
                    } else {
                        return None;
                    };
                    let scalar = representation::scalar(
                        result.value(request, other)?,
                        provider.precision(),
                    )?;
                    match kind {
                        FusionEpilogueOperation::Maximum if phase <= Phase::Column => {
                            result.clamp_min = Some(scalar);
                            phase = Phase::LowerClamp;
                        }
                        FusionEpilogueOperation::Minimum if phase <= Phase::LowerClamp => {
                            result.clamp_max = Some(scalar);
                            phase = Phase::UpperClamp;
                        }
                        FusionEpilogueOperation::RealPower
                            if current_left && phase <= Phase::UpperClamp =>
                        {
                            // Every possible finite base remains real for an integral
                            // exponent. Otherwise the ordered clamps must establish
                            // a nonnegative base; real storage alone proves nothing.
                            let nonnegative = result.clamp_min.is_some_and(|x| x >= 0.0)
                                && result.clamp_max.is_none_or(|x| x >= 0.0);
                            if scalar.fract() != 0.0 && !nonnegative {
                                return None;
                            }
                            result.exponent = Some(scalar);
                            phase = Phase::Power;
                        }
                        _ => return None,
                    }
                }
                current = *output;
                continue;
            }
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
                        result.scale = Some(scalar);
                        phase = Phase::Scale;
                    }
                    PrimitiveOp::Add if phase <= Phase::Scale => {
                        result.offset = Some(MatmulOffset::Add(scalar));
                        phase = Phase::Bias;
                    }
                    PrimitiveOp::Sub if phase <= Phase::Scale && current_left => {
                        result.offset = Some(MatmulOffset::Subtract(scalar));
                        phase = Phase::Bias;
                    }
                    PrimitiveOp::Sub if phase == Phase::Product && !current_left => {
                        result.offset = Some(MatmulOffset::SubtractFrom(scalar));
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
        let mut descriptor = self.scalar_descriptor();
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

    pub(super) fn scalar_descriptor(&self) -> MatmulEpilogue {
        let mut descriptor = MatmulEpilogue::noop();
        descriptor.affine = MatmulAffine {
            scale: self.scale,
            offset: self.offset,
        };
        descriptor.clamp_min = self.clamp_min;
        descriptor.clamp_max = self.clamp_max;
        descriptor.pow_exponent = self.exponent;
        descriptor
    }
}
