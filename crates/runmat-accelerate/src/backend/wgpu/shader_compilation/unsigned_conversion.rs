//! Widen 32-bit unsigned operands before controlled binary64 conversion.
//!
//! NVIDIA 580.88 on Windows miscompiles u32-to-f64 with RoundingModeRTE
//! as a signed conversion. Exact u32-to-u64 widening preserves the unsigned
//! domain and the required rounding mode, including for vector operands.
use super::controls::{append_instruction, Instruction};
use crate::wgsl::host_buffers::{self, ClaimedVec};
use runmat_execution::resource::ProviderHostMetadataAdmission;
use spirv::Op;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    U32,
    U64,
    F64,
}
#[derive(Clone, Copy)]
struct Shape {
    scalar: Scalar,
    lanes: u32,
}
pub(super) struct Conversion {
    pub index: usize,
    wide_type: u32,
    temporary: u32,
    operand: u32,
}
pub(super) struct Plan {
    pub conversions: ClaimedVec<Conversion>,
    pub declarations: ClaimedVec<u32>,
    pub bound: u32,
}
fn fresh_id(bound: &mut u32) -> Result<u32, host_buffers::Error> {
    let id = *bound;
    *bound = bound.checked_add(1).ok_or(host_buffers::Error::Overflow)?;
    Ok(id)
}
impl Plan {
    pub fn new(
        instructions: &[Instruction<'_>],
        mut bound: u32,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> anyhow::Result<Self> {
        // SPIR-V IDs share one namespace. This exact, admitted table stores
        // both numeric type shapes and the shapes of numeric value results.
        let mut shapes = ClaimedVec::with_capacity(bound as usize, metadata)?;
        for _ in 0..bound {
            shapes.push(None::<Shape>, metadata)?;
        }
        let mut wide = [None; 4];
        let mut conversions = ClaimedVec::with_capacity(0, metadata)?;
        let mut declarations = ClaimedVec::with_capacity(0, metadata)?;
        let mut in_function = false;
        for (index, instruction) in instructions.iter().enumerate() {
            let w = instruction.words;
            match instruction.op {
                Op::TypeInt if w.len() == 4 && w[3] == 0 => {
                    let scalar = match w[2] {
                        32 => Some(Scalar::U32),
                        64 => Some(Scalar::U64),
                        _ => None,
                    };
                    if let Some(scalar) = scalar {
                        shapes.as_mut_slice()[w[1] as usize] = Some(Shape { scalar, lanes: 1 });
                        if scalar == Scalar::U64 {
                            wide[0] = Some(w[1]);
                        }
                    }
                }
                Op::TypeFloat if w.len() == 3 && w[2] == 64 => {
                    shapes.as_mut_slice()[w[1] as usize] = Some(Shape {
                        scalar: Scalar::F64,
                        lanes: 1,
                    });
                }
                Op::TypeVector if w.len() == 4 => {
                    if let Some(component) = shapes.as_slice()[w[2] as usize] {
                        let shape = Shape {
                            scalar: component.scalar,
                            lanes: w[3],
                        };
                        anyhow::ensure!(
                            (2..=4).contains(&shape.lanes),
                            "invalid compiler vector width"
                        );
                        shapes.as_mut_slice()[w[1] as usize] = Some(shape);
                        if shape.scalar == Scalar::U64 {
                            wide[shape.lanes as usize - 1] = Some(w[1]);
                        }
                    }
                }
                Op::Function => in_function = true,
                Op::FunctionEnd => in_function = false,
                _ => {}
            }
            let constant = matches!(
                instruction.op,
                Op::Constant
                    | Op::ConstantNull
                    | Op::ConstantComposite
                    | Op::SpecConstant
                    | Op::SpecConstantComposite
                    | Op::SpecConstantOp
            );
            // Within functions a numeric type ID in operand zero identifies
            // a typed result. Labels, branch operands and value IDs cannot
            // alias a type ID in a validated module.
            if (in_function || constant) && w.len() >= 3 {
                if let Some(shape) = shapes.as_slice().get(w[1] as usize).copied().flatten() {
                    shapes.as_mut_slice()[w[2] as usize] = Some(shape);
                }
            }
            if instruction.op != Op::ConvertUToF || w.len() != 4 {
                continue;
            }
            let Some(target) = shapes.as_slice()[w[1] as usize] else {
                continue;
            };
            let Some(source) = shapes.as_slice()[w[3] as usize] else {
                continue;
            };
            if target.scalar != Scalar::F64 || source.scalar != Scalar::U32 {
                continue;
            }
            anyhow::ensure!(
                source.lanes == target.lanes,
                "compiler conversion shape mismatch"
            );
            let scalar = match wide[0] {
                Some(id) => id,
                None => {
                    let id = fresh_id(&mut bound)?;
                    append_instruction(metadata, &mut declarations, Op::TypeInt, &[id, 64, 0])?;
                    wide[0] = Some(id);
                    id
                }
            };
            let slot = &mut wide[source.lanes as usize - 1];
            let wide_type = match *slot {
                Some(id) => id,
                None => {
                    let id = fresh_id(&mut bound)?;
                    append_instruction(
                        metadata,
                        &mut declarations,
                        Op::TypeVector,
                        &[id, scalar, source.lanes],
                    )?;
                    *slot = Some(id);
                    id
                }
            };
            conversions.push(
                Conversion {
                    index,
                    wide_type,
                    temporary: fresh_id(&mut bound)?,
                    operand: w[3],
                },
                metadata,
            )?;
        }
        Ok(Self {
            conversions,
            declarations,
            bound,
        })
    }
}
impl Conversion {
    pub fn append(
        &self,
        instruction: &Instruction<'_>,
        output: &mut ClaimedVec<u32>,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> anyhow::Result<()> {
        append_instruction(
            metadata,
            output,
            Op::UConvert,
            &[self.wide_type, self.temporary, self.operand],
        )?;
        append_instruction(
            metadata,
            output,
            Op::ConvertUToF,
            &[instruction.words[1], instruction.words[2], self.temporary],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_scalar_and_vector_conversion_widens_without_changing_signed_conversion() {
        for lanes in 1..=4 {
            for existing_wide in [false, true] {
                let ty = |scalar: &str| {
                    if lanes == 1 {
                        scalar.to_owned()
                    } else {
                        format!("vec{lanes}<{scalar}>")
                    }
                };
                let (unsigned, wide, signed, float) = (ty("u32"), ty("u64"), ty("i32"), ty("f64"));
                let extra = if existing_wide {
                    format!("output[2] = {float}({wide}(input[0]));")
                } else {
                    String::new()
                };
                let source = format!("@group(0) @binding(0) var<storage, read> input: array<{unsigned}>; @group(0) @binding(1) var<storage, read_write> output: array<{float}>; @compute @workgroup_size(1) fn main() {{ output[0] = {float}(input[0]); output[1] = {float}({signed}(input[0])); {extra} }}");
                let words = super::super::spirv::compile(&source, None).unwrap();
                let words = words.as_slice();
                let mut offset = 5;
                let mut wide_values = Vec::new();
                let mut unsigned_casts = 0;
                let mut signed_casts = 0;
                while offset < words.len() {
                    let length = (words[offset] >> 16) as usize;
                    let w = &words[offset..offset + length];
                    match Op::from_u32(w[0] & 0xffff).unwrap() {
                        Op::UConvert => {
                            wide_values.push(w[2]);
                            assert!(w[2] < words[3]);
                        }
                        Op::ConvertUToF => {
                            assert!(wide_values.contains(&w[3]));
                            unsigned_casts += 1;
                        }
                        Op::ConvertSToF => signed_casts += 1,
                        _ => {}
                    }
                    offset += length;
                }
                assert_eq!(unsigned_casts, if existing_wide { 2 } else { 1 });
                assert_eq!(signed_casts, 1);
            }
        }
    }
}
