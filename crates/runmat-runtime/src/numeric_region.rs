//! Executor-neutral fused evaluation for a deliberately small, exact numeric
//! region subset. Executors build plans; Runtime owns MATLAB value semantics.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_value::{Tensor, Value};

const MAX_NODES: usize = 256;
const MAX_OUTPUTS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericUnaryOperation {
    Plus,
    Minus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericBinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    LeftDivide,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericRegionNode {
    Input(usize),
    Constant(f64),
    Unary {
        operation: NumericUnaryOperation,
        input: usize,
    },
    Binary {
        operation: NumericBinaryOperation,
        left: usize,
        right: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericRegionProgram {
    pub nodes: Vec<NumericRegionNode>,
    pub outputs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericRegionExecution {
    Completed(Vec<Value>),
    Ineligible,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericRegionWorkload {
    pub elements: usize,
    pub output_bytes_per_value: u64,
}

pub fn workload(inputs: &[&Value]) -> Option<NumericRegionWorkload> {
    let (_, shape) = analyze_inputs(inputs)?;
    let elements = shape
        .as_ref()
        .map(|shape| shape.iter().try_fold(1_usize, |n, dim| n.checked_mul(*dim)))
        .unwrap_or(Some(1))?;
    Some(NumericRegionWorkload {
        elements,
        output_bytes_per_value: u64::try_from(elements).ok()?.checked_mul(8)?,
    })
}

impl NumericRegionProgram {
    pub fn validate(&self, input_count: usize) -> Result<(), &'static str> {
        if self.nodes.is_empty()
            || self.nodes.len() > MAX_NODES
            || self.outputs.is_empty()
            || self.outputs.len() > MAX_OUTPUTS
        {
            return Err("numeric region plan exceeds its structural bounds");
        }
        for (index, node) in self.nodes.iter().enumerate() {
            match node {
                NumericRegionNode::Input(input) if *input >= input_count => {
                    return Err("numeric region input is out of bounds")
                }
                NumericRegionNode::Unary { input, .. } if *input >= index => {
                    return Err("numeric region unary dependency is not topological")
                }
                NumericRegionNode::Binary { left, right, .. }
                    if *left >= index || *right >= index =>
                {
                    return Err("numeric region binary dependency is not topological")
                }
                _ => {}
            }
        }
        if self
            .outputs
            .iter()
            .any(|output| *output >= self.nodes.len())
        {
            return Err("numeric region output is out of bounds");
        }
        Ok(())
    }
}

/// Evaluate a supported scalar/dense-double expression DAG in one element
/// pass. Equal-shape dense arrays and scalar expansion are admitted; every
/// other representation or broadcast shape fails closed before publication.
pub fn execute(
    program: &NumericRegionProgram,
    inputs: &[&Value],
    cancellation: &Arc<AtomicBool>,
) -> Result<NumericRegionExecution, &'static str> {
    program.validate(inputs.len())?;
    let Some((input_views, output_shape)) = analyze_inputs(inputs) else {
        return Ok(NumericRegionExecution::Ineligible);
    };
    let element_count = output_shape
        .as_ref()
        .map(|shape| shape.iter().try_fold(1_usize, |n, dim| n.checked_mul(*dim)))
        .unwrap_or(Some(1))
        .ok_or("numeric region output shape overflows this host")?;
    let total_output_elements = element_count
        .checked_mul(program.outputs.len())
        .ok_or("numeric region output allocation overflows this host")?;
    let mut output_values = vec![Vec::with_capacity(element_count); program.outputs.len()];
    let mut values = vec![0.0; program.nodes.len()];
    for element in 0..element_count {
        if element % 1_024 == 0 && cancellation.load(Ordering::Relaxed) {
            return Ok(NumericRegionExecution::Cancelled);
        }
        for (index, node) in program.nodes.iter().enumerate() {
            values[index] = match *node {
                NumericRegionNode::Input(input) => input_views[input].at(element),
                NumericRegionNode::Constant(value) => value,
                NumericRegionNode::Unary { operation, input } => match operation {
                    NumericUnaryOperation::Plus => values[input],
                    NumericUnaryOperation::Minus => -values[input],
                },
                NumericRegionNode::Binary {
                    operation,
                    left,
                    right,
                } => match operation {
                    NumericBinaryOperation::Add => values[left] + values[right],
                    NumericBinaryOperation::Subtract => values[left] - values[right],
                    NumericBinaryOperation::Multiply => values[left] * values[right],
                    NumericBinaryOperation::Divide => values[left] / values[right],
                    NumericBinaryOperation::LeftDivide => values[right] / values[left],
                },
            };
        }
        for (output, node) in output_values.iter_mut().zip(&program.outputs) {
            output.push(values[*node]);
        }
    }
    debug_assert_eq!(
        output_values.iter().map(Vec::len).sum::<usize>(),
        total_output_elements
    );
    let outputs = if let Some(shape) = output_shape {
        output_values
            .into_iter()
            .map(|values| Tensor::new(values, shape.clone()).map(Value::Tensor))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "numeric region produced an invalid dense tensor")?
    } else {
        output_values
            .into_iter()
            .map(|values| Value::Num(values[0]))
            .collect()
    };
    Ok(NumericRegionExecution::Completed(outputs))
}

fn analyze_inputs<'a>(inputs: &[&'a Value]) -> Option<(Vec<NumericInput<'a>>, Option<Vec<usize>>)> {
    let mut output_shape: Option<Vec<usize>> = None;
    let mut tensor_shape: Option<Vec<usize>> = None;
    let mut input_views = Vec::with_capacity(inputs.len());
    for &input in inputs {
        match input {
            Value::Num(value) => input_views.push(NumericInput::Scalar(*value)),
            Value::Tensor(tensor) => {
                let values = tensor.as_f64_slice()?;
                if tensor.len() == 1 {
                    output_shape.get_or_insert_with(|| tensor.shape.clone());
                    input_views.push(NumericInput::Scalar(values[0]));
                } else {
                    if tensor_shape
                        .as_ref()
                        .is_some_and(|shape| shape != &tensor.shape)
                    {
                        return None;
                    }
                    tensor_shape = Some(tensor.shape.clone());
                    output_shape = Some(tensor.shape.clone());
                    input_views.push(NumericInput::Dense(values));
                }
            }
            _ => return None,
        }
    }
    Some((input_views, output_shape))
}

#[derive(Clone, Copy)]
enum NumericInput<'a> {
    Scalar(f64),
    Dense(&'a [f64]),
}

impl NumericInput<'_> {
    fn at(self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => value,
            Self::Dense(values) => values[index],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execute_values(
        program: &NumericRegionProgram,
        inputs: &[Value],
        cancellation: &Arc<AtomicBool>,
    ) -> Result<NumericRegionExecution, &'static str> {
        let inputs = inputs.iter().collect::<Vec<_>>();
        execute(program, &inputs, cancellation)
    }

    #[test]
    fn fuses_dense_double_chain_with_scalar_expansion() {
        let program = NumericRegionProgram {
            nodes: vec![
                NumericRegionNode::Input(0),
                NumericRegionNode::Input(1),
                NumericRegionNode::Binary {
                    operation: NumericBinaryOperation::Add,
                    left: 0,
                    right: 1,
                },
                NumericRegionNode::Constant(2.0),
                NumericRegionNode::Binary {
                    operation: NumericBinaryOperation::Multiply,
                    left: 2,
                    right: 3,
                },
            ],
            outputs: vec![4],
        };
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = execute_values(
            &program,
            &[Value::Tensor(tensor), Value::Num(1.0)],
            &Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let NumericRegionExecution::Completed(outputs) = result else {
            panic!("eligible numeric region did not execute")
        };
        let Value::Tensor(output) = &outputs[0] else {
            panic!("expected dense output")
        };
        assert_eq!(output.materialize_f64(), vec![4.0, 6.0, 8.0]);
    }

    #[test]
    fn rejects_incompatible_dense_shapes_before_publication() {
        let program = NumericRegionProgram {
            nodes: vec![
                NumericRegionNode::Input(0),
                NumericRegionNode::Input(1),
                NumericRegionNode::Binary {
                    operation: NumericBinaryOperation::Add,
                    left: 0,
                    right: 1,
                },
            ],
            outputs: vec![2],
        };
        let left = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert_eq!(
            execute_values(
                &program,
                &[Value::Tensor(left), Value::Tensor(right)],
                &Arc::new(AtomicBool::new(false)),
            )
            .unwrap(),
            NumericRegionExecution::Ineligible
        );
    }

    #[test]
    fn cancellation_discards_the_transactional_result() {
        let program = NumericRegionProgram {
            nodes: vec![
                NumericRegionNode::Input(0),
                NumericRegionNode::Constant(1.0),
                NumericRegionNode::Binary {
                    operation: NumericBinaryOperation::Add,
                    left: 0,
                    right: 1,
                },
            ],
            outputs: vec![2],
        };
        let input = Tensor::new(vec![0.0; 2_048], vec![1, 2_048]).unwrap();
        let cancellation = Arc::new(AtomicBool::new(true));
        assert_eq!(
            execute_values(&program, &[Value::Tensor(input)], &cancellation).unwrap(),
            NumericRegionExecution::Cancelled
        );
    }

    #[test]
    fn preserves_scalar_array_shape() {
        let program = NumericRegionProgram {
            nodes: vec![
                NumericRegionNode::Input(0),
                NumericRegionNode::Constant(1.0),
                NumericRegionNode::Binary {
                    operation: NumericBinaryOperation::Add,
                    left: 0,
                    right: 1,
                },
            ],
            outputs: vec![2],
        };
        let input = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result = execute_values(
            &program,
            &[Value::Tensor(input)],
            &Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let NumericRegionExecution::Completed(outputs) = result else {
            panic!("eligible numeric region did not execute")
        };
        let Value::Tensor(output) = &outputs[0] else {
            panic!("scalar arrays must remain arrays")
        };
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(output.materialize_f64(), vec![3.0]);
    }
}
