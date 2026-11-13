//! Actual checked-stream control geometry for physical recipe validation.
//! No synthetic FusionGroupPlan or unverified source length grants this bound.
use super::{program, scalar_name};
use crate::fusion::wgsl_source::{CountedWgslSource, MeasuredFragment};
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::scalar_program::{
    EmittedScalarImplementation, ScalarLiteral, ScalarNode, ScalarProgram,
};
use runmat_runtime::builtins::common::spec::{write_unary_math_scalar_expression, ScalarType};
use runmat_types::{NumericClass, OperatorKind, ScalarOperation};
use std::fmt::Write as _;

fn measure(write: impl FnOnce(&mut CountedWgslSource) -> Result<()>) -> Result<usize> {
    let mut out = CountedWgslSource::with_preamble_bytes(0);
    write(&mut out)?;
    Ok(out.bytes())
}

/// Exact short-string allocator used by the checked emitter itself. It cannot
/// grow if a formatting implementation changes between count and emission.
pub(crate) fn exact_format(arguments: std::fmt::Arguments<'_>) -> Result<String> {
    // This block is inside the caller's already-admitted producer aggregate;
    // the shared exact formatter must not mint another per-fragment claim.
    let bytes = crate::wgsl::host_buffers::formatted_len(arguments)?;
    let mut output = crate::wgsl::host_buffers::ClaimedString::with_capacity(bytes, None)?;
    output.append_fmt(arguments, None)?;
    Ok(output.into_parts().0)
}

#[derive(Default)]
struct Bytes(usize);
impl Bytes {
    fn add(&mut self, n: usize) -> Result<()> {
        self.0 = self
            .0
            .checked_add(n)
            .ok_or_else(|| anyhow!("checked producer resource extent overflow"))?;
        Ok(())
    }
    fn vector<T>(&mut self, n: usize) -> Result<()> {
        self.add(std::alloc::Layout::array::<T>(n)?.size())
    }
}

pub(crate) struct CheckedProducerResources {
    pub body_bytes: usize,
    pub scratch_bytes: usize,
}

impl CheckedProducerResources {
    pub(crate) fn for_program(
        checked: &ScalarProgram,
        scalar: ScalarType,
        inputs: usize,
        widening: bool,
    ) -> Result<Self> {
        let nodes = &checked.wire().nodes;
        let n = nodes.len();
        let max_id = nodes.iter().map(ScalarNode::id).max().unwrap_or(0);
        let max_index = nodes
            .iter()
            .filter_map(|node| match node {
                ScalarNode::Apply { source_index, .. } => Some(*source_index),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let slot = inputs.saturating_sub(1);
        let mut operand = 0usize;
        for length in [
            measure(|out| {
                write!(out, "input{slot}.data[i{slot}]")?;
                Ok(())
            })?,
            measure(|out| {
                write!(out, "bitcast<f32>(input{slot}.data[i{slot}])")?;
                Ok(())
            })?,
            measure(|out| {
                write!(out, "control{max_id}_bits")?;
                Ok(())
            })?,
            measure(|out| {
                write!(out, "bitcast<f32>(control{max_id}_bits)")?;
                Ok(())
            })?,
            measure(|out| {
                write!(out, "typed{max_index}_bits")?;
                Ok(())
            })?,
            measure(|out| {
                write!(out, "bitcast<f32>(typed{max_index}_bits)")?;
                Ok(())
            })?,
        ] {
            operand = operand.max(length);
        }
        let capture_words = scalar == ScalarType::F32
            && nodes.iter().any(|node| {
                checked.exact_product(node.id()).is_some()
                    || matches!(
                        node,
                        ScalarNode::Apply {
                            operation: ScalarOperation::Operator(OperatorKind::ElementwisePower),
                            ..
                        }
                    )
            });
        let mut body = 0usize;
        let mut expression_upper = 0usize;
        let mut operands_count = 0usize;
        let mut emitted_implementations = 0usize;
        for node in nodes {
            let line = match node {
                ScalarNode::Input { .. } => 0,
                ScalarNode::Literal {
                    id,
                    value: ScalarLiteral::F32Bits(bits),
                } if capture_words => measure(|out| {
                    writeln!(out, "    let control{id}_bits: u32 = 0x{bits:08x}u;")?;
                    Ok(())
                })?,
                ScalarNode::Literal { .. } => 0,
                ScalarNode::Apply {
                    id,
                    source_index,
                    class,
                    operation,
                    operands,
                } => {
                    operands_count = operands_count
                        .checked_add(operands.len())
                        .ok_or_else(|| anyhow!("checked operand count overflow"))?;
                    ensure!(
                        operands.len() <= 2,
                        "checked producer has unsupported arity"
                    );
                    if scalar == ScalarType::F32 && checked.exact_product(*id).is_some() {
                        emitted_implementations = emitted_implementations
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("checked implementation count overflow"))?;
                        measure(|out| {
                            writeln!(out, "    let typed{source_index}_bits: u32 = rm_exact_product_bits({}, 0xffffffffu);", MeasuredFragment(operand))?;
                            Ok(())
                        })?
                    } else if *operation
                        == ScalarOperation::Operator(OperatorKind::ElementwisePower)
                    {
                        emitted_implementations = emitted_implementations
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("checked implementation count overflow"))?;
                        let [_, exponent] = operands.as_slice() else {
                            return Err(anyhow!("certified POWER arity changed"));
                        };
                        let control = nodes
                            .iter()
                            .find_map(|node| match node {
                                ScalarNode::Literal {
                                    id,
                                    value: ScalarLiteral::F64Bits(bits),
                                } if id == exponent => {
                                    runmat_accelerate_api::ExactSingleBasePower::for_control(
                                        runmat_accelerate_api::PowerScalarControl::DoubleBits(
                                            *bits,
                                        ),
                                    )
                                }
                                _ => None,
                            })
                            .ok_or_else(|| anyhow!("certified POWER control is unavailable"))?;
                        let entry = crate::kernel_math::power_exact::entrypoint(control);
                        measure(|out| {
                            writeln!(
                                out,
                                "    let typed{source_index}_bits: u32 = {entry}({});",
                                MeasuredFragment(operand)
                            )?;
                            Ok(())
                        })?
                    } else {
                        let ty = match class {
                            NumericClass::Single => ScalarType::F32,
                            NumericClass::Double => ScalarType::F64,
                            _ => return Err(anyhow!("nonfloating checked producer")),
                        };
                        let name = scalar_name(ty)?;
                        let cast = measure(|out| {
                            write!(out, "{name}({})", MeasuredFragment(operand))?;
                            Ok(())
                        })?;
                        let expression = measure(|out| {
                            let args = [MeasuredFragment(cast), MeasuredFragment(cast)];
                            match operation {
                                ScalarOperation::RealArithmetic(recipe) => {
                                    program::write_real_arithmetic_expr(
                                        out,
                                        *recipe,
                                        [MeasuredFragment(operand), MeasuredFragment(operand)],
                                        scalar,
                                    )?
                                }
                                ScalarOperation::Operator(operator) => {
                                    crate::fusion::write_scalar_operator_expr(
                                        out,
                                        *operator,
                                        &args[..operands.len()],
                                    )?
                                }
                                ScalarOperation::UnaryMath(operation) => {
                                    write_unary_math_scalar_expression(
                                        *operation,
                                        ty,
                                        MeasuredFragment(cast),
                                        out,
                                    )?
                                }
                                ScalarOperation::Cast(
                                    NumericClass::Single | NumericClass::Double,
                                ) => write!(out, "{}", MeasuredFragment(cast))?,
                                _ => {
                                    return Err(anyhow!(
                                        "checked producer has no counted scalar lowering"
                                    ))
                                }
                            }
                            Ok(())
                        })?;
                        expression_upper = expression_upper.max(expression);
                        measure(|out| {
                            writeln!(
                                out,
                                "    let typed{source_index}: {name} = {};",
                                MeasuredFragment(expression)
                            )?;
                            if capture_words && ty == ScalarType::F32 {
                                writeln!(out, "    let typed{source_index}_bits: u32 = bitcast<u32>(typed{source_index});")?;
                            }
                            Ok(())
                        })?
                    }
                }
            };
            body = body
                .checked_add(line)
                .ok_or_else(|| anyhow!("checked producer body extent overflow"))?;
        }
        let mut bytes = Bytes::default();
        bytes.add(body)?;
        bytes.vector::<u32>(n)?; // sorted actual checked IDs
        for _ in 0..2 {
            bytes.vector::<(u32, Option<String>)>(n)?;
            bytes.add(
                n.checked_mul(operand)
                    .ok_or_else(|| anyhow!("checked expression extent overflow"))?,
            )?;
        }
        bytes.vector::<ScalarNode>(n)?; // lowerer's retained ScalarProgram clone
        bytes.vector::<u32>(operands_count)?;
        bytes.vector::<u32>(
            n.checked_mul(2)
                .ok_or_else(|| anyhow!("checked control count overflow"))?,
        )?;
        bytes.vector::<u32>(
            operands_count
                .checked_add(1)
                .ok_or_else(|| anyhow!("checked control exclusion overflow"))?,
        )?;
        bytes.vector::<EmittedScalarImplementation>(n)?;
        bytes.vector::<EmittedScalarImplementation>(
            emitted_implementations
                .checked_add(usize::from(widening))
                .ok_or_else(|| anyhow!("checked implementation count overflow"))?,
        )?;
        bytes.vector::<&str>(2)?;
        bytes.vector::<String>(2)?;
        let casted = operand
            .checked_add(5)
            .ok_or_else(|| anyhow!("checked cast extent overflow"))?;
        bytes.add(casted)?;
        bytes.add(casted)?; // two actual casted arguments
        bytes.add(expression_upper)?;
        bytes.add(operand)?; // current temporary's short name
        bytes.add(operand)?; // cloned base bits / raw bitcast fallback
        bytes.add(operand)?; // current word temporary / literal name
        bytes.add(operand)?; // input load while a casted input is inserted
        Ok(Self {
            body_bytes: body,
            scratch_bytes: bytes.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::typed_producer::{ExpressionSlots, TypedProducer};
    use runmat_accelerate_api::scalar_program::SCALAR_PROGRAM_REVISION;
    use runmat_execution::Digest;

    #[test]
    fn actual_checked_stream_bounds_both_precisions_and_word_power_without_source_plan() {
        for (class, scalar, power) in [
            (NumericClass::Single, ScalarType::F32, false),
            (NumericClass::Double, ScalarType::F64, false),
            (NumericClass::Single, ScalarType::F32, true),
        ] {
            let mut nodes = vec![ScalarNode::Input { id: 7, class }];
            if power {
                nodes.push(ScalarNode::Literal {
                    id: 8,
                    value: ScalarLiteral::F64Bits(2.0_f64.to_bits()),
                });
            }
            nodes.push(ScalarNode::Apply {
                id: 9,
                source_index: u32::MAX,
                class,
                operation: if power {
                    ScalarOperation::Operator(OperatorKind::ElementwisePower)
                } else {
                    ScalarOperation::UnaryMath(runmat_types::ScalarUnaryMath::Sin)
                },
                operands: if power { vec![7, 8] } else { vec![7] },
            });
            let checked = ScalarProgram::new(
                Digest::sha256(b"checked geometry fixture"),
                SCALAR_PROGRAM_REVISION,
                nodes,
                9,
            )
            .unwrap();
            let ids: Vec<_> = checked.wire().nodes.iter().map(ScalarNode::id).collect();
            let mut expressions = ExpressionSlots::from_sorted_ids(&ids);
            let mut words = ExpressionSlots::from_sorted_ids(&ids);
            if power {
                expressions
                    .insert(7, "bitcast<f32>(input0.data[i0])".into())
                    .unwrap();
                words.insert(7, "input0.data[i0]".into()).unwrap();
            } else {
                expressions.insert(7, "input0.data[i0]".into()).unwrap();
            }
            let bound = CheckedProducerResources::for_program(&checked, scalar, 1, power).unwrap();
            let producer = TypedProducer::lower_checked_indexed_with_words(
                &checked,
                scalar,
                expressions,
                words,
                Some(bound.body_bytes),
            )
            .unwrap()
            .unwrap();
            assert!(producer.body.len() <= bound.body_bytes);
            assert!(bound.scratch_bytes > bound.body_bytes);
            assert!(producer.body.contains("typed4294967295"));
            assert_eq!(producer.implementations().len(), usize::from(power));
        }
    }
}
