//! Typed scalar producer emission, independent of dispatch and physical indexing.
//! Root slicing does not grant permission to prune execution effects or bypass
//! selected-binding admission. Callers must establish both before execution.
use super::{expression::admitted_expression, primitive_expr, semantics::ResolvedProgram};
use super::{FusionGroupPlan, FusionOp};
use crate::graph::ValueId;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::scalar_program::{
    EmittedScalarImplementation, ScalarImplementation, ScalarProgram,
};
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};
use runmat_types::NumericClass;
use std::collections::{HashMap, HashSet};

#[path = "typed_producer/program.rs"]
mod program;
#[path = "typed_producer/constants.rs"]
mod constants;

pub(crate) struct TypedProducer {
    pub body: String,
    expressions: HashMap<ValueId, String>,
    helpers: Vec<(FusionMathHelper, ScalarType)>,
    checked_program: Option<ScalarProgram>,
    exact_power: bool,
    exact_product: bool,
    word_expressions: HashMap<ValueId, String>,
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
            let leaves = plan
                .inputs
                .iter()
                .chain(plan.const_values.keys())
                .map(|&id| (id, String::new()))
                .collect();
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
        mut expressions: HashMap<ValueId, String>,
        roots: Option<&[ValueId]>,
        mut word_expressions: HashMap<ValueId, String>,
    ) -> Result<Self> {
        program.require_source_plan(plan)?;
        // Captured constants are leaves even when the physical indexing owner
        // has no buffer expression for them. Invocation inputs still win over
        // same-ID captured constants, exactly as in source assembly.
        let mut leaves = expressions.clone();
        for &id in plan.const_values.keys() {
            if !program.is_invocation_input(id) {
                leaves.entry(id).or_default();
            }
        }
        // Even complete emission must reject a constant which ambiguously
        // aliases a produced SSA value; None still emits every operation.
        let indices = producer_indices(plan, &leaves, roots.unwrap_or_default())?;
        let selected = roots.map(|_| indices);
        let capture_words = buffer_type == ScalarType::F32
            && Self::requires_binary32_word_transport(plan, program, roots)?;
        let constant_body = constants::seed(
            plan,
            program,
            buffer_type,
            &mut expressions,
            selected.as_ref(),
            roots,
            &mut word_expressions,
            capture_words,
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
        // Multi-output and complete legacy emission do not construct a scalar
        // program. Advertised closed bindings still have to pass their typed
        // class and arity contract before their legacy WGSL expression is used.
        if !matches!(roots, Some([_])) {
            for (index, operation) in plan.operations.iter().enumerate() {
                if selected
                    .as_ref()
                    .is_some_and(|indices| !indices.contains(&index))
                {
                    continue;
                }
                let (_, output) = operands(operation);
                if output.is_some_and(|id| {
                    program
                        .declaration(id)
                        .and_then(|declaration| declaration.semantics)
                        .and_then(|semantics| semantics.scalar_operation)
                        .is_some()
                }) {
                    program::resolved_operation(program, operation)?;
                }
            }
        }
        // The checked closed operation stream is the arithmetic authority for
        // its admitted subset. WGSL-only declarations keep their established
        // path until their exact numerical policy is represented there too.
        if let Some(&[root]) = roots {
            if let Some(checked) = Self::emit_checked_program(plan, program, buffer_type, root)? {
                if let Some(lowered) = Self::lower_checked_with_words(
                    &checked,
                    buffer_type,
                    expressions.clone(),
                    word_expressions.clone(),
                )? {
                    return Ok(lowered);
                }
            }
        }
        let mut body = constant_body;
        let mut helpers = Vec::new();
        let mut exact_power = false;
        let mut exact_product = false;
        let mut implementations = Vec::new();
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
                    let temporary = format!("typed{index}_bits");
                    body.push_str(&format!(
                        "    let {temporary}: u32 = rm_exact_product_bits({bits}, 0x{:08x}u);\n",
                        receipt.exact.binary32_bits()
                    ));
                    expressions.insert(output, format!("bitcast<f32>({temporary})"));
                    word_expressions.insert(output, temporary);
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
                let base_bits = word_expressions
                    .get(base)
                    .cloned()
                    .or_else(|| {
                        expressions
                            .get(base)
                            .map(|value| format!("bitcast<u32>({value})"))
                    })
                    .ok_or_else(|| anyhow!("missing certified POWER base"))?;
                let name = crate::kernel_math::power_exact::entrypoint(operation);
                let temporary = format!("typed{index}_bits");
                body.push_str(&format!(
                    "    let {temporary}: u32 = {name}({base_bits});\n"
                ));
                expressions.insert(output, format!("bitcast<f32>({temporary})"));
                word_expressions.insert(output, temporary);
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
            let operands = inputs
                .iter()
                .map(|id| {
                    let expression = expressions
                        .get(id)
                        .ok_or_else(|| anyhow!("missing typed fusion operand"))?;
                    Ok((*id, format!("{operation_name}({expression})")))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let resolved = program::resolved_operation(program, operation)?;
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
                    admitted_expression(declaration, inputs, &operands, operation_type)
                } else {
                    match operation {
                        FusionOp::Primitive { op, .. } => primitive_expr(*op, inputs, &operands),
                        FusionOp::Builtin { .. } => {
                            return Err(anyhow!("fusion builtin was not admitted"))
                        }
                    }
                }
                .ok_or_else(|| anyhow!("no typed fusion expression for the selected operation"))?;
            let temporary = format!("typed{index}");
            body.push_str(&format!(
                "    let {temporary}: {operation_name} = {expression};\n"
            ));
            if capture_words && operation_type == ScalarType::F32 {
                // Capture the already-admitted native Single result. This is
                // not a stronger accuracy/FTZ promise for that native operation.
                let bits = format!("{temporary}_bits");
                body.push_str(&format!(
                    "    let {bits}: u32 = bitcast<u32>({temporary});\n"
                ));
                word_expressions.insert(output, bits);
            }
            expressions.insert(output, temporary);
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
            body,
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
        // Shader text is itself the backend cache key. Include resolved facts
        // even when two otherwise identical programs happen to emit the same
        // arithmetic, so representation contracts do not share a stale key.
        let identity = program.cache_identity()?;
        let mut math = String::new();
        if self.exact_power || self.exact_product {
            math.push_str(crate::kernel_math::power_exact::functions());
        }
        if self.exact_product {
            math.push_str(crate::kernel_math::product_exact::functions());
        }
        for &(helper, precision) in &self.helpers {
            let precision = match precision {
                ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                _ => return Err(anyhow!("unsupported fusion math helper precision")),
            };
            match helper {
                FusionMathHelper::RealPower => {
                    math.push_str(&crate::kernel_math::power::functions(precision))
                }
                FusionMathHelper::Round => {
                    math.push_str(&crate::kernel_math::rounding::functions(precision))
                }
                FusionMathHelper::FloatingRemainder => math.push_str(
                    &crate::kernel_math::floating_remainder::functions(precision),
                ),
            }
        }

        Ok(format!("// {identity}\n{math}\n"))
    }
    pub(crate) fn checked_preamble(&self) -> Result<String> {
        let checked = self
            .checked_program
            .as_ref()
            .ok_or_else(|| anyhow!("producer has no checked scalar program"))?;
        let mut math = String::new();
        if self.exact_power || self.exact_product {
            math.push_str(crate::kernel_math::power_exact::functions());
        }
        if self.exact_product {
            math.push_str(crate::kernel_math::product_exact::functions());
        }
        Ok(format!("// {}\n{math}\n", checked.wire().source_identity))
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
    leaves: &HashMap<ValueId, String>,
    roots: &[ValueId],
) -> Result<HashSet<usize>> {
    let mut producers = HashMap::new();
    for (index, operation) in plan.operations.iter().enumerate() {
        let (_, output) = operands(operation);
        let output = output.ok_or_else(|| anyhow!("fusion operation has no output"))?;
        if leaves.contains_key(&output) || producers.insert(output, index).is_some() {
            return Err(anyhow!("ambiguous typed fusion producer"));
        }
    }
    let mut selected = HashSet::new();
    let mut pending: Vec<_> = roots
        .iter()
        .map(|&id| (id, plan.operations.len()))
        .collect();
    while let Some((id, consumer)) = pending.pop() {
        if leaves.contains_key(&id) {
            continue;
        }
        let index = *producers
            .get(&id)
            .ok_or_else(|| anyhow!("missing typed fusion producer"))?;
        if index >= consumer {
            return Err(anyhow!("cyclic or unordered typed fusion producer"));
        }
        if selected.insert(index) {
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
