//! Apply binary64 float controls and exact unsigned-conversion legalization.
//! Bindings, existing result IDs, control flow and bounds checks are preserved.
use crate::wgsl::host_buffers::{self, ClaimedVec};
use runmat_execution::resource::ProviderHostMetadataAdmission;
use spirv::{Capability, Decoration, ExecutionMode, GLOp, Op};

pub(super) struct Instruction<'a> {
    pub op: Op,
    pub words: &'a [u32],
}

fn instructions<'a>(
    words: &'a [u32],
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> anyhow::Result<ClaimedVec<Instruction<'a>>> {
    anyhow::ensure!(
        words.len() >= 5 && words[0] == spirv::MAGIC_NUMBER,
        "invalid compiler SPIR-V header"
    );
    let mut offset = 5;
    let mut count = 0usize;
    while offset < words.len() {
        let length = (words[offset] >> 16) as usize;
        anyhow::ensure!(
            length != 0 && length <= words.len() - offset,
            "invalid compiler SPIR-V instruction length"
        );
        anyhow::ensure!(
            Op::from_u32(words[offset] & 0xffff).is_some(),
            "unknown compiler SPIR-V instruction"
        );
        count = count.checked_add(1).ok_or(host_buffers::Error::Overflow)?;
        offset += length;
    }
    let mut result = ClaimedVec::with_capacity(count, metadata)?;
    offset = 5;
    while offset < words.len() {
        let length = (words[offset] >> 16) as usize;
        result.push(
            Instruction {
                op: Op::from_u32(words[offset] & 0xffff).expect("validated compiler opcode"),
                words: &words[offset..offset + length],
            },
            metadata,
        )?;
        offset += length;
    }
    Ok(result)
}

fn insert<T: PartialEq>(
    set: &mut ClaimedVec<T>,
    value: T,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<(), host_buffers::Error> {
    if !set.as_slice().contains(&value) {
        set.push(value, metadata)?;
    }
    Ok(())
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

pub(super) fn append_instruction(
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
    output: &mut ClaimedVec<u32>,
    op: Op,
    operands: &[u32],
) -> anyhow::Result<()> {
    let count = operands
        .len()
        .checked_add(1)
        .and_then(|n| u16::try_from(n).ok())
        .ok_or(host_buffers::Error::Overflow)?;
    output.push((u32::from(count) << 16) | op as u32, metadata)?;
    output.extend_from_slice(operands, metadata)?;
    Ok(())
}

pub(super) fn apply(
    words: &[u32],
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> anyhow::Result<ClaimedVec<u32>> {
    let instructions = instructions(words, metadata)?;
    let conversions =
        super::unsigned_conversion::Plan::new(instructions.as_slice(), words[3], metadata)?;
    let mut wide_types = ClaimedVec::with_capacity(0, metadata)?;
    let mut imports = ClaimedVec::with_capacity(0, metadata)?;
    let mut entries = ClaimedVec::with_capacity(0, metadata)?;
    let mut existing_caps = ClaimedVec::with_capacity(0, metadata)?;
    let mut existing_modes = ClaimedVec::with_capacity(0, metadata)?;
    let mut existing_no_contraction = ClaimedVec::with_capacity(0, metadata)?;
    for instruction in instructions.as_slice() {
        let w = instruction.words;
        match instruction.op {
            Op::TypeFloat if w.len() == 3 && w[2] == 64 => {
                insert(&mut wide_types, w[1], metadata)?;
            }
            Op::TypeVector | Op::TypeMatrix
                if w.len() == 4 && wide_types.as_slice().contains(&w[2]) =>
            {
                insert(&mut wide_types, w[1], metadata)?;
            }
            Op::Capability if w.len() == 2 => {
                insert(&mut existing_caps, w[1], metadata)?;
            }
            Op::EntryPoint if w.len() >= 4 => {
                insert(&mut entries, w[2], metadata)?;
            }
            Op::ExecutionMode if w.len() == 4 => {
                insert(&mut existing_modes, (w[1], w[2], w[3]), metadata)?;
            }
            Op::Decorate if w.len() == 3 && w[2] == Decoration::NoContraction as u32 => {
                existing_no_contraction.push(w[1], metadata)?;
            }
            Op::ExtInstImport if w.len() >= 3 => {
                let glsl = w[2..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .eq(b"GLSL.std.450".iter().copied());
                if glsl {
                    insert(&mut imports, w[1], metadata)?;
                }
            }
            _ => {}
        }
    }
    anyhow::ensure!(!entries.is_empty(), "SPIR-V has no compute entrypoint");
    let mut arithmetic = ClaimedVec::with_capacity(0, metadata)?;
    for instruction in instructions.as_slice() {
        let w = instruction.words;
        if w.len() < 3 || !wide_types.as_slice().contains(&w[1]) {
            continue;
        }
        if instruction.op == Op::ExtInst && w.len() >= 5 && imports.as_slice().contains(&w[3]) {
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
            arithmetic.push(w[2], metadata)?;
        }
    }
    entries.as_mut_slice().sort_unstable();
    arithmetic.as_mut_slice().sort_unstable();
    arithmetic.dedup();
    existing_no_contraction.as_mut_slice().sort_unstable();
    existing_no_contraction.dedup();
    let capability_at = instructions
        .as_slice()
        .iter()
        .position(|instruction| instruction.op != Op::Capability)
        .unwrap_or(instructions.len());
    let modes_at = instructions
        .as_slice()
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
        .as_slice()
        .iter()
        .position(|instruction| is_type_or_constant(instruction.op))
        .ok_or_else(|| anyhow::anyhow!("SPIR-V type section is missing"))?;
    anyhow::ensure!(
        capability_at <= modes_at && modes_at <= annotations_at,
        "unexpected compiler SPIR-V section order"
    );
    let caps = [
        Capability::SignedZeroInfNanPreserve,
        Capability::RoundingModeRTE,
    ]
    .iter()
    .filter(|cap| !existing_caps.as_slice().contains(&(**cap as u32)))
    .count();
    let integer_capability = !conversions.conversions.is_empty()
        && !existing_caps
            .as_slice()
            .contains(&(Capability::Int64 as u32));
    let declarations_at = instructions
        .as_slice()
        .iter()
        .position(|instruction| instruction.op == Op::Function)
        .ok_or_else(|| anyhow::anyhow!("SPIR-V compute function is missing"))?;
    let modes = entries
        .as_slice()
        .iter()
        .try_fold(0usize, |count, &entry| {
            let missing = [
                ExecutionMode::SignedZeroInfNanPreserve,
                ExecutionMode::RoundingModeRTE,
            ]
            .iter()
            .filter(|mode| {
                !existing_modes
                    .as_slice()
                    .contains(&(entry, **mode as u32, 64))
            })
            .count();
            count
                .checked_add(missing)
                .ok_or(host_buffers::Error::Overflow)
        })?;
    let annotations = arithmetic
        .as_slice()
        .iter()
        .filter(|id| {
            existing_no_contraction
                .as_slice()
                .binary_search(id)
                .is_err()
        })
        .count();
    let output_words = caps
        .checked_mul(2)
        .and_then(|n| modes.checked_mul(4).and_then(|m| n.checked_add(m)))
        .and_then(|n| annotations.checked_mul(3).and_then(|a| n.checked_add(a)))
        .and_then(|n| words.len().checked_add(n))
        .and_then(|n| n.checked_add(conversions.declarations.len()))
        .and_then(|n| {
            conversions
                .conversions
                .len()
                .checked_mul(4)
                .and_then(|extra| n.checked_add(extra))
        })
        .and_then(|n| n.checked_add(if integer_capability { 2 } else { 0 }))
        .ok_or(host_buffers::Error::Overflow)?;
    let mut output = ClaimedVec::with_capacity(output_words, metadata)?;
    output.extend_from_slice(&words[..5], metadata)?;
    output.as_mut_slice()[3] = conversions.bound;
    let mut conversion = conversions.conversions.as_slice().iter().peekable();
    for index in 0..=instructions.len() {
        if index == capability_at {
            if integer_capability {
                append_instruction(
                    metadata,
                    &mut output,
                    Op::Capability,
                    &[Capability::Int64 as u32],
                )?;
            }
            for capability in [
                Capability::SignedZeroInfNanPreserve,
                Capability::RoundingModeRTE,
            ] {
                if !existing_caps.as_slice().contains(&(capability as u32)) {
                    append_instruction(
                        metadata,
                        &mut output,
                        Op::Capability,
                        &[capability as u32],
                    )?;
                }
            }
        }
        if index == modes_at {
            for &entry in entries.as_slice() {
                for mode in [
                    ExecutionMode::SignedZeroInfNanPreserve,
                    ExecutionMode::RoundingModeRTE,
                ] {
                    if !existing_modes
                        .as_slice()
                        .contains(&(entry, mode as u32, 64))
                    {
                        append_instruction(
                            metadata,
                            &mut output,
                            Op::ExecutionMode,
                            &[entry, mode as u32, 64],
                        )?;
                    }
                }
            }
        }
        if index == annotations_at {
            for &id in arithmetic.as_slice() {
                if existing_no_contraction
                    .as_slice()
                    .binary_search(&id)
                    .is_err()
                {
                    append_instruction(
                        metadata,
                        &mut output,
                        Op::Decorate,
                        &[id, Decoration::NoContraction as u32],
                    )?;
                }
            }
        }
        if index == declarations_at {
            output.extend_from_slice(conversions.declarations.as_slice(), metadata)?;
        }
        if let Some(instruction) = instructions.as_slice().get(index) {
            if conversion
                .peek()
                .is_some_and(|conversion| conversion.index == index)
            {
                conversion.next().expect("checked conversion").append(
                    instruction,
                    &mut output,
                    metadata,
                )?;
            } else {
                output.extend_from_slice(instruction.words, metadata)?;
            }
        }
    }
    anyhow::ensure!(
        output.len() == output_words,
        "SPIR-V control word count changed"
    );
    Ok(output)
}
