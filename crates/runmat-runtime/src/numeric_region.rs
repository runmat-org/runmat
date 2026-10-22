//! Executor-neutral fused evaluation for a deliberately small, exact numeric
//! region subset. Executors build plans; Runtime owns MATLAB value semantics.

use runmat_execution::cancellation::CancellationToken;

use runmat_value::{Tensor, Value};

mod execution;
pub mod service;
pub use execution::{
    execute_in_context, output_retained_bytes, NumericRegionOwnedExecution,
    NUMERIC_REGION_EVALUATOR_ARRAY_BYTES,
};

/// Authoritative validated program limits, also bounding native metadata slots.
pub const MAX_NODES: usize = 256;
pub const MAX_OUTPUTS: usize = 64;

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
    pub output_rank: Option<usize>,
}

pub fn workload(inputs: &[&Value]) -> Option<NumericRegionWorkload> {
    let shape = analyze_inputs(inputs)?;
    let elements = shape
        .as_ref()
        .map(|shape| shape.iter().try_fold(1_usize, |n, dim| n.checked_mul(*dim)))
        .unwrap_or(Some(1))?;
    Some(NumericRegionWorkload {
        elements,
        output_bytes_per_value: u64::try_from(elements).ok()?.checked_mul(8)?,
        output_rank: shape.map(<[usize]>::len),
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
/// Unbounded compatibility entrypoint; Runtime-owned execution must use
/// `execute_in_context` so finite physical claims survive publication.
pub fn execute(
    program: &NumericRegionProgram,
    inputs: &[&Value],
    cancellation: &CancellationToken,
) -> Result<NumericRegionExecution, &'static str> {
    program.validate(inputs.len())?;
    match execution::run(program, inputs, cancellation, None)
        .map_err(|_| "numeric region execution failed")?
    {
        execution::StorageExecution::Ineligible => Ok(NumericRegionExecution::Ineligible),
        execution::StorageExecution::Cancelled => Ok(NumericRegionExecution::Cancelled),
        execution::StorageExecution::Completed(values) => {
            let values = values
                .try_into_owned()
                .map_err(|_| "numeric output storage is shared")?
                .try_into_unbounded_vec()
                .map_err(|_| "numeric compatibility output carries a finite claim")?;
            Ok(NumericRegionExecution::Completed(values))
        }
    }
}

/// Borrow shape metadata from the original physical owner. Validation neither
/// collects input views nor clones a shape before admission.
fn analyze_inputs<'a>(inputs: &[&'a Value]) -> Option<Option<&'a [usize]>> {
    let mut output_shape: Option<&'a [usize]> = None;
    let mut tensor_shape: Option<&'a [usize]> = None;
    for &input in inputs {
        match input {
            Value::Num(_) => {}
            Value::Tensor(tensor) => {
                tensor.as_f64_slice()?;
                if tensor.len() == 1 {
                    output_shape.get_or_insert(tensor.shape());
                } else {
                    if tensor_shape.is_some_and(|shape| shape != tensor.shape()) {
                        return None;
                    }
                    tensor_shape = Some(tensor.shape());
                    output_shape = Some(tensor.shape());
                }
            }
            _ => return None,
        }
    }
    Some(output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execute_values(
        program: &NumericRegionProgram,
        inputs: &[Value],
        cancellation: &CancellationToken,
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
            &runmat_execution::cancellation::CancellationSource::new().token(),
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
                &runmat_execution::cancellation::CancellationSource::new().token(),
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
        let source = runmat_execution::cancellation::CancellationSource::new();
        source.cancel(runmat_execution::CancellationReason::User);
        let cancellation = source.token();
        assert_eq!(
            execute_values(&program, &[Value::Tensor(input)], &cancellation).unwrap(),
            NumericRegionExecution::Cancelled
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
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
        for shape in [Vec::new(), vec![1], vec![1, 1]] {
            let input = Value::Tensor(Tensor::new(vec![2.0], shape.clone()).unwrap());
            assert_eq!(
                workload(&[&input]),
                Some(NumericRegionWorkload {
                    elements: 1,
                    output_bytes_per_value: 8,
                    output_rank: Some(shape.len()),
                }),
            );
            let result = execute_values(
                &program,
                &[input],
                &runmat_execution::cancellation::CancellationSource::new().token(),
            )
            .unwrap();
            let NumericRegionExecution::Completed(outputs) = result else {
                panic!("eligible numeric region did not execute")
            };
            let Value::Tensor(output) = &outputs[0] else {
                panic!("scalar arrays must remain arrays")
            };
            assert_eq!(output.shape(), shape);
            assert_eq!(output.materialize_f64(), vec![3.0]);
        }
    }
}
