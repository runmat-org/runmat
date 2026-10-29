//! Typed scalar producer emission, independent of dispatch and physical indexing.
//! Root slicing does not grant permission to prune execution effects or bypass
//! selected-binding admission. Callers must establish both before execution.
use super::{
    expression::admitted_expression_checked, primitive_expr_checked, semantics::ResolvedProgram,
};
use super::{FusionGroupPlan, FusionOp};
use crate::graph::ValueId;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::scalar_program::{
    EmittedScalarImplementation, ScalarImplementation, ScalarProgram,
};
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};
use runmat_types::NumericClass;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Exact operation-index storage for one selected producer slice. The count is
/// separate because `Vec::len` describes the complete source program.
pub(crate) struct SelectedProducerIndices {
    bits: Vec<bool>,
    selected: usize,
}

impl SelectedProducerIndices {
    pub(crate) fn contains(&self, index: &usize) -> bool {
        self.bits.get(*index).copied().unwrap_or(false)
    }

    pub(crate) fn len(&self) -> usize {
        self.selected
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.selected == 0
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits
            .iter()
            .enumerate()
            .filter_map(|(index, selected)| selected.then_some(index))
    }
}

#[path = "typed_producer/checked_resources.rs"]
mod checked_resources;
#[path = "typed_producer/constants.rs"]
mod constants;
#[path = "typed_producer/expression_slots.rs"]
mod expression_slots;
pub(crate) use checked_resources::{exact_format as checked_format, CheckedProducerResources};
#[path = "typed_producer/program.rs"]
mod program;
pub(crate) use expression_slots::ExpressionSlots;
#[path = "typed_producer/geometry.rs"]
mod geometry;
pub(crate) use geometry::CountedRecipeUnavailable;
pub(super) use geometry::{
    body_upper_bytes, max_operand_bytes, preamble_upper_bytes, structural_upper_bytes,
};

pub(crate) struct TypedProducer {
    pub body: String,
    expressions: ExpressionSlots,
    helpers: Vec<(FusionMathHelper, ScalarType)>,
    checked_program: Option<ScalarProgram>,
    exact_power: bool,
    exact_product: bool,
    word_expressions: ExpressionSlots,
    implementations: Vec<EmittedScalarImplementation>,
}
impl TypedProducer {
    /// Select transport from the same producer slice the emitter will visit,
    /// never from unrelated receipts held by the larger resolved program.
    pub(crate) fn requires_binary32_word_transport(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        roots: Option<&[ValueId]>,
    ) -> Result<bool> {
        program.require_source_plan(plan)?;
        let selected = if let Some(roots) = roots {
            let capacity = plan
                .inputs
                .len()
                .checked_add(plan.const_values.len())
                .ok_or_else(|| anyhow!("typed fusion leaf count overflows"))?;
            let mut leaves = Vec::new();
            leaves.try_reserve_exact(capacity)?;
            leaves.extend(plan.inputs.iter().chain(plan.const_values.keys()).copied());
            leaves.sort_unstable();
            leaves.dedup();
            Some(producer_indices(plan, &leaves, roots)?)
        } else {
            None
        };
        Ok(plan
            .operations
            .iter()
            .enumerate()
            .any(|(index, operation)| {
                selected
                    .as_ref()
                    .is_none_or(|indices| indices.contains(&index))
                    && operands(operation).1.is_some_and(|output| {
                        program.exact_product(output).is_some()
                            || program.exact_single_base_power(output).is_some()
                    })
            }))
    }
    pub fn checked_program(&self) -> Option<&ScalarProgram> {
        self.checked_program.as_ref()
    }
    /// None preserves complete elementwise emission and its error order.
    /// Some emits only admitted pure producers of the roots, retaining original
    /// operation order and temporary names. Leaves come from the indexing owner.
    pub fn emit(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        expressions: HashMap<ValueId, String>,
        roots: Option<&[ValueId]>,
    ) -> Result<Self> {
        Self::emit_with_words(
            plan,
            program,
            buffer_type,
            expressions,
            roots,
            HashMap::new(),
        )
    }

    pub(crate) fn emit_with_words(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        expressions: HashMap<ValueId, String>,
        roots: Option<&[ValueId]>,
        word_expressions: HashMap<ValueId, String>,
    ) -> Result<Self> {
        Self::emit_indexed_with_words(
            plan,
            program,
            buffer_type,
            ExpressionSlots::from_map(plan, expressions)?,
            roots,
            ExpressionSlots::from_map(plan, word_expressions)?,
        )
    }

    pub(crate) fn emit_indexed_with_words(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        expressions: ExpressionSlots,
        roots: Option<&[ValueId]>,
        word_expressions: ExpressionSlots,
    ) -> Result<Self> {
        Self::emit_indexed_with_words_bounded(
            plan,
            program,
            buffer_type,
            expressions,
            roots,
            word_expressions,
            None,
        )
    }

    pub(crate) fn emit_indexed_with_words_bounded(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        mut expressions: ExpressionSlots,
        roots: Option<&[ValueId]>,
        mut word_expressions: ExpressionSlots,
        body_capacity: Option<usize>,
    ) -> Result<Self> {
        program.require_source_plan(plan)?;
        // Captured constants are leaves even when the physical indexing owner
        // has no buffer expression for them. Invocation inputs still win over
        // same-ID captured constants, exactly as in source assembly.
        let leaf_capacity = expressions
            .len()
            .checked_add(plan.const_values.len())
            .ok_or_else(|| anyhow!("typed fusion leaf count overflows"))?;
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(leaf_capacity)?;
        leaves.extend(expressions.keys().copied());
        for &id in plan.const_values.keys() {
            if !program.is_invocation_input(id) {
                leaves.push(id);
            }
        }
        leaves.sort_unstable();
        leaves.dedup();
        // Even complete emission must reject a constant which ambiguously
        // aliases a produced SSA value; None still emits every operation.
        let mut indices = producer_indices(plan, &leaves, roots.unwrap_or_default())?;
        if roots.is_none() {
            indices.bits.fill(true);
            indices.selected = plan.operations.len();
        }
        let selected = roots.map(|_| &indices);
        let capture_words = buffer_type == ScalarType::F32
            && Self::requires_binary32_word_transport(plan, program, roots)?;
        let constant_body = constants::seed(
            plan,
            program,
            buffer_type,
            &mut expressions,
            constants::ConstantSelection {
                indices: selected,
                roots,
            },
            capture_words.then_some(&mut word_expressions),
            body_capacity,
        )?;
        for (index, operation) in plan.operations.iter().enumerate() {
            if selected
                .as_ref()
                .is_some_and(|indices| !indices.contains(&index))
            {
                continue;
            }
            let (_, output) = operands(operation);
            if let Some(output) = output {
                let precision = match buffer_type {
                    ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                    ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                    _ => return Err(anyhow!("ordinary fusion has no floating precision")),
                };
                program.require_ordinary_arithmetic(output, precision)?;
            }
        }
        // The checked closed operation stream is the arithmetic authority for
        // its admitted subset. WGSL-only declarations keep their established
        // path until their exact numerical policy is represented there too.
        if let Some(&[root]) = roots {
            if let Some(checked) = Self::emit_checked_program(plan, program, buffer_type, root)? {
                if let Some(lowered) = Self::lower_checked_indexed_with_words_owned(
                    checked,
                    buffer_type,
                    expressions.clone(),
                    word_expressions.clone(),
                    body_capacity,
                )? {
                    return Ok(lowered);
                }
            }
        }
        // Resolve the selected stream in source order before any legacy
        // expression can decline or an exact recipe can skip the ordinary
        // branch. The temporary represented facts die at this boundary; the
        // indexed identities are reused below without a second inference.
        let resolved = program::selected_resolution_receipt(plan, program, &indices)?;
        let mut body =
            super::wgsl_source::BoundedWgslBody::from_source(constant_body, body_capacity)?;
        let mut helpers = runmat_value::exact_capacity_vec(6)?;
        helpers.try_reserve_exact(6)?;
        let mut exact_power = false;
        let mut exact_product = false;
        let mut implementations = runmat_value::exact_capacity_vec(plan.operations.len())?;
        implementations.try_reserve_exact(plan.operations.len())?;
        for (index, operation) in plan.operations.iter().enumerate() {
            if selected
                .as_ref()
                .is_some_and(|indices| !indices.contains(&index))
            {
                continue;
            }
            let (inputs, output) = match operation {
                FusionOp::Primitive { inputs, output, .. }
                | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
            };
            let output = output.ok_or_else(|| anyhow!("fusion operation has no output"))?;
            if buffer_type == ScalarType::F32 {
                if let Some(receipt) = program.exact_product(output).filter(|receipt| {
                    expressions
                        .get(&receipt.control)
                        .is_some_and(String::is_empty)
                }) {
                    let bits = word_expressions.get(&receipt.base).ok_or_else(|| {
                        anyhow!("exact product requires word-preserving Single data")
                    })?;
                    let temporary = checked_format(format_args!("typed{index}_bits"))?;
                    writeln!(
                        &mut body,
                        "    let {temporary}: u32 = rm_exact_product_bits({bits}, 0x{:08x}u);",
                        receipt.exact.binary32_bits()
                    )?;
                    expressions.insert(
                        output,
                        checked_format(format_args!("bitcast<f32>({temporary})"))?,
                    )?;
                    word_expressions.insert(output, temporary)?;
                    exact_product = true;
                    implementations.push(product_implementation(output));
                    continue;
                }
            }
            if let Some(operation) = program.exact_single_base_power(output) {
                // The certified implementation reads original binary32 words.
                // The legacy F64 fusion ABI stages Single data as doubles;
                // converting those back in WGSL is not a proven subnormal-
                // preserving bit transport. Let the original operation use
                // its exact F32-storage recipe instead of claiming that ABI.
                if buffer_type != ScalarType::F32 {
                    return Err(anyhow!("certified POWER fusion requires binary32 data ABI"));
                }
                let [base, _] = inputs.as_slice() else {
                    return Err(anyhow!("certified POWER arity changed"));
                };
                let base_bits = if let Some(bits) = word_expressions.get(base) {
                    bits.clone()
                } else if let Some(value) = expressions.get(base) {
                    checked_format(format_args!("bitcast<u32>({value})"))?
                } else {
                    return Err(anyhow!("missing certified POWER base"));
                };
                let name = crate::kernel_math::power_exact::entrypoint(operation);
                let temporary = checked_format(format_args!("typed{index}_bits"))?;
                writeln!(&mut body, "    let {temporary}: u32 = {name}({base_bits});")?;
                expressions.insert(
                    output,
                    checked_format(format_args!("bitcast<f32>({temporary})"))?,
                )?;
                word_expressions.insert(output, temporary)?;
                exact_power = true;
                implementations.push(crate::kernel_math::power_exact::implementation(
                    operation, output,
                ));
                continue;
            }
            let operation_type = match program.floating_class(output)? {
                NumericClass::Single => ScalarType::F32,
                NumericClass::Double => ScalarType::F64,
                _ => return Err(anyhow!("unsupported fusion arithmetic class")),
            };
            // An F32-only provider cannot execute an F64 source operation by
            // silently changing its class, including a Double interior node.
            if buffer_type == ScalarType::F32 && operation_type == ScalarType::F64 {
                return Err(anyhow!(
                    "fusion provider cannot represent a Double operation"
                ));
            }
            let operation_name = scalar_name(operation_type)?;
            let mut operands = Vec::new();
            operands.try_reserve_exact(inputs.len())?;
            for id in inputs {
                let expression = expressions
                    .get(id)
                    .ok_or_else(|| anyhow!("missing typed fusion operand"))?;
                operands.push((
                    *id,
                    checked_format(format_args!("{operation_name}({expression})"))?,
                ));
            }
            operands.sort_unstable_by_key(|(id, _)| *id);
            let resolved = resolved[index];
            let expression =
                if let Some(runmat_types::ScalarOperation::RealArithmetic(recipe)) = resolved {
                    let [left, right] = inputs.as_slice() else {
                        return Err(anyhow!("represented arithmetic has invalid arity"));
                    };
                    let arguments = [
                        expressions
                            .get(left)
                            .map(String::as_str)
                            .ok_or_else(|| anyhow!("missing represented arithmetic operand"))?,
                        expressions
                            .get(right)
                            .map(String::as_str)
                            .ok_or_else(|| anyhow!("missing represented arithmetic operand"))?,
                    ];
                    Some(program::real_arithmetic_expr(
                        recipe,
                        arguments,
                        buffer_type,
                    )?)
                } else if let Some(declaration) = program.declaration(output) {
                    if selected.is_some() {
                        let entry = program
                            .catalog_entry(output)
                            .ok_or_else(|| anyhow!("missing typed producer contract"))?;
                        if entry.contract.purity != runmat_builtins::BuiltinPurity::Pure
                            || entry.contract.workspace_effect.is_some()
                            || entry.contract.environment_effect.is_some()
                        {
                            return Err(anyhow!("typed producer slice requires a pure contract"));
                        }
                    }
                    for helper in declaration
                        .semantics
                        .into_iter()
                        .flat_map(|semantics| semantics.helpers)
                    {
                        if !helpers.contains(&(*helper, operation_type)) {
                            helpers.push((*helper, operation_type));
                        }
                    }
                    admitted_expression_checked(declaration, inputs, &operands, operation_type)?
                } else {
                    match operation {
                        FusionOp::Primitive { op, .. } => {
                            primitive_expr_checked(*op, inputs, &operands)?
                        }
                        FusionOp::Builtin { .. } => {
                            return Err(anyhow!("fusion builtin was not admitted"))
                        }
                    }
                }
                .ok_or_else(|| anyhow!("no typed fusion expression for the selected operation"))?;
            let temporary = checked_format(format_args!("typed{index}"))?;
            writeln!(
                &mut body,
                "    let {temporary}: {operation_name} = {expression};"
            )?;
            if capture_words && operation_type == ScalarType::F32 {
                // Capture the already-admitted native Single result. This is
                // not a stronger accuracy/FTZ promise for that native operation.
                let bits = checked_format(format_args!("{temporary}_bits"))?;
                writeln!(
                    &mut body,
                    "    let {bits}: u32 = bitcast<u32>({temporary});"
                )?;
                word_expressions.insert(output, bits)?;
            }
            expressions.insert(output, temporary)?;
        }

        if let Some(roots) = roots {
            for &root in roots {
                // An input-only producer must satisfy the same representation
                // contract as a computed root; absence of arithmetic is not
                // permission to downcast a Double leaf on an F32 provider.
                if program.floating_class(root)? == NumericClass::Double
                    && buffer_type == ScalarType::F32
                {
                    return Err(anyhow!(
                        "fusion provider cannot represent a Double operation"
                    ));
                }
            }
        }
        Ok(Self {
            body: body.into_source(),
            expressions,
            helpers,
            checked_program: None,
            exact_power,
            exact_product,
            word_expressions,
            implementations,
        })
    }
    pub(crate) fn word_expression(&self, id: ValueId) -> Option<&str> {
        self.word_expressions.get(&id).map(String::as_str)
    }
    pub(crate) fn uses_exact_product_words(&self) -> bool {
        self.exact_product
    }
    pub(crate) fn implementations(&self) -> &[EmittedScalarImplementation] {
        &self.implementations
    }
    pub fn expression(&self, id: ValueId) -> Result<&str> {
        self.expressions
            .get(&id)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("missing typed fusion output"))
    }
    pub fn preamble(&self, program: &ResolvedProgram) -> Result<String> {
        let mut measured = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
        self.write_preamble(program, &mut measured)?;
        let mut preamble = String::new();
        preamble.try_reserve_exact(measured.bytes())?;
        self.write_preamble(program, &mut preamble)?;
        if preamble.len() != measured.bytes() {
            return Err(anyhow!("fusion preamble changed length during emission"));
        }
        Ok(preamble)
    }

    pub(crate) fn write_preamble(
        &self,
        program: &ResolvedProgram,
        writer: &mut impl std::fmt::Write,
    ) -> Result<()> {
        // Shader text is itself the backend cache key. Include resolved facts
        // even when two otherwise identical programs happen to emit the same
        // arithmetic, so representation contracts do not share a stale key.
        let identity = program.cache_identity()?;
        writeln!(writer, "// {identity}")?;
        if self.exact_power || self.exact_product {
            writer.write_str(crate::kernel_math::power_exact::functions())?;
        }
        if self.exact_product {
            writer.write_str(crate::kernel_math::product_exact::functions())?;
        }
        for &(helper, precision) in &self.helpers {
            let precision = match precision {
                ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                _ => return Err(anyhow!("unsupported fusion math helper precision")),
            };
            match helper {
                FusionMathHelper::RealPower => {
                    crate::kernel_math::power::write_functions(precision, writer)?
                }
                FusionMathHelper::Round => {
                    crate::kernel_math::rounding::write_functions(precision, writer)?
                }
                FusionMathHelper::FloatingRemainder => {
                    crate::kernel_math::floating_remainder::write_functions(precision, writer)?
                }
            }
        }
        writer.write_char('\n')?;
        Ok(())
    }
    pub(crate) fn write_checked_preamble(
        &self,
        writer: &mut (impl std::fmt::Write + ?Sized),
    ) -> Result<()> {
        let checked = self
            .checked_program
            .as_ref()
            .ok_or_else(|| anyhow!("producer has no checked scalar program"))?;
        writeln!(writer, "// {}", checked.wire().source_identity)?;
        if self.exact_power || self.exact_product {
            writer.write_str(crate::kernel_math::power_exact::functions())?;
        }
        if self.exact_product {
            writer.write_str(crate::kernel_math::product_exact::functions())?;
        }
        writer.write_char('\n')?;
        Ok(())
    }
}
fn product_implementation(output: ValueId) -> EmittedScalarImplementation {
    EmittedScalarImplementation {
        output,
        implementation: ScalarImplementation::Binary32Product48RneV1,
        invocations_per_element: 1,
        algorithmic_units_per_invocation: crate::kernel_math::product_exact::work_prior()
            .algorithmic_units(),
    }
}
pub(crate) fn producer_indices(
    plan: &FusionGroupPlan,
    leaves: &[ValueId],
    roots: &[ValueId],
) -> Result<SelectedProducerIndices> {
    let mut producers = runmat_value::exact_capacity_vec(plan.operations.len())?;
    for (index, operation) in plan.operations.iter().enumerate() {
        let (_, output) = operands(operation);
        let output = output.ok_or_else(|| anyhow!("fusion operation has no output"))?;
        if leaves.binary_search(&output).is_ok() {
            return Err(anyhow!("ambiguous typed fusion producer"));
        }
        producers.push((output, index));
    }
    producers.sort_unstable_by_key(|(id, _)| *id);
    if producers.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(anyhow!("ambiguous typed fusion producer"));
    }
    let mut bits = runmat_value::exact_capacity_vec(plan.operations.len())?;
    bits.resize(plan.operations.len(), false);
    let mut selected = SelectedProducerIndices { bits, selected: 0 };
    let pending_capacity = plan
        .operations
        .iter()
        .try_fold(roots.len(), |count, operation| {
            count.checked_add(operands(operation).0.len())
        })
        .ok_or_else(|| anyhow!("typed fusion producer traversal overflows"))?;
    let mut pending = runmat_value::exact_capacity_vec(pending_capacity)?;
    pending.extend(roots.iter().map(|&id| (id, plan.operations.len())));
    while let Some((id, consumer)) = pending.pop() {
        if leaves.binary_search(&id).is_ok() {
            continue;
        }
        let index = producers
            .binary_search_by_key(&id, |(output, _)| *output)
            .ok()
            .map(|position| producers[position].1)
            .ok_or_else(|| anyhow!("missing typed fusion producer"))?;
        if index >= consumer {
            return Err(anyhow!("cyclic or unordered typed fusion producer"));
        }
        if !selected.bits[index] {
            selected.bits[index] = true;
            selected.selected += 1;
            let (inputs, _) = operands(&plan.operations[index]);
            pending.extend(inputs.iter().map(|&input| (input, index)));
        }
    }
    Ok(selected)
}
fn operands(operation: &FusionOp) -> (&[ValueId], Option<ValueId>) {
    match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs, *output)
        }
    }
}
pub(super) fn scalar_name(scalar: ScalarType) -> Result<&'static str> {
    match scalar {
        ScalarType::F32 => Ok("f32"),
        ScalarType::F64 => Ok("f64"),
        _ => Err(anyhow!("unsupported floating fusion buffer precision")),
    }
}
#[cfg(test)]
#[path = "typed_producer_tests.rs"]
mod tests;
