//! Captured scalar leaves shared by elementwise and reduction producers.
//! Render represented IEEE words, never a decimal round-trip or buffer cast.
use super::*;
use runmat_accelerate_api::scalar_program::ScalarLiteral;

/// Source slice whose captured constants may be emitted by this producer.
pub(super) struct ConstantSelection<'a> {
    pub(super) indices: Option<&'a SelectedProducerIndices>,
    pub(super) roots: Option<&'a [ValueId]>,
}

pub(super) fn seed(
    plan: &FusionGroupPlan,
    program: &ResolvedProgram,
    buffer_type: ScalarType,
    expressions: &mut ExpressionSlots,
    selection: ConstantSelection<'_>,
    mut words: Option<&mut ExpressionSlots>,
    body_capacity: Option<usize>,
) -> Result<String> {
    let ConstantSelection {
        indices: selected,
        roots,
    } = selection;
    let base = roots
        .unwrap_or_default()
        .len()
        .checked_add(expressions.len())
        .ok_or_else(|| anyhow!("fusion constant-use count overflows"))?;
    let capacity = plan
        .operations
        .iter()
        .try_fold(base, |count, operation| {
            count.checked_add(operands(operation).0.len())
        })
        .ok_or_else(|| anyhow!("fusion constant-use count overflows"))?;
    let mut used = Vec::new();
    used.try_reserve_exact(capacity)?;
    used.extend(roots.unwrap_or_default().iter().copied());
    for (index, operation) in plan.operations.iter().enumerate() {
        if selected.is_none_or(|selected| selected.contains(&index)) {
            used.extend(operands(operation).0.iter().copied());
        }
    }
    // Complete multi-output emission receives requested constant outputs as
    // placeholders from its indexing owner; they need no producer operation.
    used.extend(expressions.keys().copied());
    used.sort_unstable();
    used.dedup();
    let controls = program.embedded_controls_sorted_for_precision(
        plan,
        roots.unwrap_or_default(),
        if buffer_type == ScalarType::F32 {
            runmat_accelerate_api::ProviderPrecision::F32
        } else {
            runmat_accelerate_api::ProviderPrecision::F64
        },
    )?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(plan.const_values.len())?;
    ids.extend(
        plan.const_values
            .keys()
            .copied()
            .filter(|id| used.binary_search(id).is_ok() && !program.is_invocation_input(*id)),
    );
    ids.sort_unstable();
    let mut body = super::super::wgsl_source::BoundedWgslBody::new(body_capacity)?;
    for id in ids {
        // Only an authenticated, exclusively metadata scalar can omit its
        // Double representation. Complete callers identify omitted controls
        // by an empty expression; sliced callers supply their exact roots.
        if controls.binary_search(&id).is_ok()
            && (roots.is_some() || expressions.get(&id).is_some_and(String::is_empty))
        {
            expressions.insert(id, String::new())?;
            continue;
        }
        let literal = program::literal(&plan.const_values[&id])
            .ok_or_else(|| anyhow!("typed fusion has an unsupported captured scalar"))?;
        let class = program.represented_constant_class(id, literal.class())?;
        if let Some(words) = words
            .as_deref_mut()
            .filter(|_| class == NumericClass::Single)
        {
            let bits = match literal {
                ScalarLiteral::F32Bits(bits) => bits,
                // Only the existing explicit precision-policy owner can have
                // resolved original Double storage to represented Single.
                ScalarLiteral::F64Bits(bits) => (f64::from_bits(bits) as f32).to_bits(),
            };
            let name = format!("control{id}_bits");
            write!(&mut body, "    let {name}: u32 = 0x{bits:08x}u;\n    let control{id} = bitcast<f32>({name});\n")?;
            words.insert(id, name)?;
            expressions.insert(id, format!("control{id}"))?;
            continue;
        }
        let expression = match (literal, class) {
            (ScalarLiteral::F32Bits(bits), NumericClass::Single) => {
                format!("bitcast<f32>(0x{bits:08x}u)")
            }
            (ScalarLiteral::F64Bits(bits), NumericClass::Single) => {
                let bits = (f64::from_bits(bits) as f32).to_bits();
                format!("bitcast<f32>(0x{bits:08x}u)")
            }
            (ScalarLiteral::F64Bits(bits), NumericClass::Double) => {
                if buffer_type != ScalarType::F64 {
                    return Err(anyhow!("provider cannot represent a Double constant"));
                }
                // The existing F64 fallback requirement is F64IeeeWords.
                // Keep each source integer within u32; a wide `...u` WGSL
                // token must not be assumed to acquire a u64 contextual type.
                format!(
                    "runmat_f64_from_words(0x{:08x}u, 0x{:08x}u)",
                    bits as u32,
                    (bits >> 32) as u32
                )
            }
            _ => return Err(anyhow!("unsupported represented constant class")),
        };
        writeln!(&mut body, "    let control{id} = {expression};")?;
        expressions.insert(id, format!("control{id}"))?;
    }
    Ok(body.into_source())
}
