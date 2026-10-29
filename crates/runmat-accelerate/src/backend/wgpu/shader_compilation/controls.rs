//! Apply float controls to compiler-emitted SPIR-V without changing executable
//! instructions, identifiers, bindings, control flow or bounds-check code.

use spirv::{Capability, Decoration, ExecutionMode, GLOp, Op};
use std::collections::BTreeSet;

struct Instruction<'a> {
    op: Op,
    words: &'a [u32],
}

fn instructions(words: &[u32]) -> anyhow::Result<Vec<Instruction<'_>>> {
    anyhow::ensure!(
        words.len() >= 5 && words[0] == spirv::MAGIC_NUMBER,
        "invalid compiler SPIR-V header"
    );
    let mut result = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        anyhow::ensure!(
            count != 0 && count <= words.len() - offset,
            "invalid compiler SPIR-V instruction length"
        );
        let op = Op::from_u32(words[offset] & 0xffff)
            .ok_or_else(|| anyhow::anyhow!("unknown compiler SPIR-V instruction"))?;
        result.push(Instruction {
            op,
            words: &words[offset..offset + count],
        });
        offset += count;
    }
    Ok(result)
}

fn forbidden_math(op: u32) -> bool {
    matches!(
        GLOp::from_u32(op),
        Some(
            GLOp::Radians
                | GLOp::Degrees
                | GLOp::Sin
                | GLOp::Cos
                | GLOp::Tan
                | GLOp::Asin
                | GLOp::Acos
                | GLOp::Atan
                | GLOp::Atan2
                | GLOp::Sinh
                | GLOp::Cosh
                | GLOp::Tanh
                | GLOp::Asinh
                | GLOp::Acosh
                | GLOp::Atanh
                | GLOp::Exp
                | GLOp::Exp2
                | GLOp::Log
                | GLOp::Log2
                | GLOp::Pow
        )
    )
}

fn is_type_or_constant(op: Op) -> bool {
    matches!(
        op,
        Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypeOpaque
            | Op::TypePointer
            | Op::TypeFunction
            | Op::TypeForwardPointer
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::Constant
            | Op::ConstantComposite
            | Op::ConstantNull
            | Op::SpecConstant
            | Op::Variable
            | Op::Function
    )
}

fn append_instruction(output: &mut Vec<u32>, op: Op, operands: &[u32]) {
    output.push(((operands.len() as u32 + 1) << 16) | op as u32);
    output.extend_from_slice(operands);
}

pub(super) fn apply(words: &[u32]) -> anyhow::Result<Vec<u32>> {
    let instructions = instructions(words)?;
    let mut wide_types = BTreeSet::new();
    let mut imports = BTreeSet::new();
    let mut entries = BTreeSet::new();
    let mut existing_caps = BTreeSet::new();
    let mut existing_modes = BTreeSet::new();
    let mut existing_no_contraction = BTreeSet::new();
    for instruction in &instructions {
        let w = instruction.words;
        match instruction.op {
            Op::TypeFloat if w.len() == 3 && w[2] == 64 => {
                wide_types.insert(w[1]);
            }
            Op::TypeVector | Op::TypeMatrix if w.len() == 4 && wide_types.contains(&w[2]) => {
                wide_types.insert(w[1]);
            }
            Op::Capability if w.len() == 2 => {
                existing_caps.insert(w[1]);
            }
            Op::EntryPoint if w.len() >= 4 => {
                entries.insert(w[2]);
            }
            Op::ExecutionMode if w.len() == 4 => {
                existing_modes.insert((w[1], w[2], w[3]));
            }
            Op::Decorate if w.len() == 3 && w[2] == Decoration::NoContraction as u32 => {
                existing_no_contraction.insert(w[1]);
            }
            Op::ExtInstImport if w.len() >= 3 => {
                let name: Vec<u8> = w[2..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect();
                if name == b"GLSL.std.450" {
                    imports.insert(w[1]);
                }
            }
            _ => {}
        }
    }
    anyhow::ensure!(!entries.is_empty(), "SPIR-V has no compute entrypoint");
    let mut arithmetic = BTreeSet::new();
    for instruction in &instructions {
        let w = instruction.words;
        if w.len() < 3 || !wide_types.contains(&w[1]) {
            continue;
        }
        if instruction.op == Op::ExtInst && w.len() >= 5 && imports.contains(&w[3]) {
            anyhow::ensure!(
                !forbidden_math(w[4]),
                "illegal binary64 GLSL extended instruction remains"
            );
        }
        if matches!(
            instruction.op,
            Op::FAdd
                | Op::FSub
                | Op::FMul
                | Op::FDiv
                | Op::FRem
                | Op::FMod
                | Op::VectorTimesScalar
                | Op::MatrixTimesScalar
                | Op::VectorTimesMatrix
                | Op::MatrixTimesVector
                | Op::MatrixTimesMatrix
                | Op::Dot
        ) {
            arithmetic.insert(w[2]);
        }
    }
    let capability_at = instructions
        .iter()
        .position(|instruction| instruction.op != Op::Capability)
        .unwrap_or(instructions.len());
    let modes_at = instructions
        .iter()
        .rposition(|instruction| {
            matches!(
                instruction.op,
                Op::EntryPoint | Op::ExecutionMode | Op::ExecutionModeId
            )
        })
        .map(|index| index + 1)
        .ok_or_else(|| anyhow::anyhow!("SPIR-V entrypoint section is missing"))?;
    let annotations_at = instructions
        .iter()
        .position(|instruction| is_type_or_constant(instruction.op))
        .ok_or_else(|| anyhow::anyhow!("SPIR-V type section is missing"))?;
    anyhow::ensure!(
        capability_at <= modes_at && modes_at <= annotations_at,
        "unexpected compiler SPIR-V section order"
    );
    let mut output = words[..5].to_vec();
    for index in 0..=instructions.len() {
        if index == capability_at {
            for capability in [
                Capability::SignedZeroInfNanPreserve,
                Capability::RoundingModeRTE,
            ] {
                if !existing_caps.contains(&(capability as u32)) {
                    append_instruction(&mut output, Op::Capability, &[capability as u32]);
                }
            }
        }
        if index == modes_at {
            for &entry in &entries {
                for mode in [
                    ExecutionMode::SignedZeroInfNanPreserve,
                    ExecutionMode::RoundingModeRTE,
                ] {
                    if !existing_modes.contains(&(entry, mode as u32, 64)) {
                        append_instruction(
                            &mut output,
                            Op::ExecutionMode,
                            &[entry, mode as u32, 64],
                        );
                    }
                }
            }
        }
        if index == annotations_at {
            for &id in &arithmetic {
                if !existing_no_contraction.contains(&id) {
                    append_instruction(
                        &mut output,
                        Op::Decorate,
                        &[id, Decoration::NoContraction as u32],
                    );
                }
            }
        }
        if let Some(instruction) = instructions.get(index) {
            output.extend_from_slice(instruction.words);
        }
    }
    Ok(output)
}
