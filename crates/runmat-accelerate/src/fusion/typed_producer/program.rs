use super::*;
use crate::graph::PrimitiveOp;
use runmat_accelerate_api::scalar_program::{
    ScalarLiteral, ScalarNode, ScalarProgram, SCALAR_PROGRAM_REVISION,
};
use runmat_types::{OperatorKind, ScalarOperation};
use runmat_value::{NumericScalar, Value};

impl TypedProducer {
    /// Build a portable scalar receipt from the same selected, typed operation
    /// slice used by WGSL emission. Unsupported legacy expressions are not
    /// guessed into a second arithmetic authority.
    pub(crate) fn emit_checked_program(
        plan: &FusionGroupPlan,
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        root: ValueId,
    ) -> Result<Option<ScalarProgram>> {
        let leaves: HashMap<_, _> = plan
            .inputs
            .iter()
            .chain(plan.const_values.keys())
            .map(|&id| (id, String::new()))
            .collect();
        let selected = producer_indices(plan, &leaves, &[root])?;
        // Check every selected advertised binding before an unsupported legacy
        // operation can request fallback. A malformed closed binding must not
        // execute through its older expression template by accident.
        let resolved = plan
            .operations
            .iter()
            .enumerate()
            .filter(|(index, _)| selected.contains(index))
            .map(|(index, operation)| {
                resolved_operation(program, operation).map(|identity| (index, identity))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        if resolved.values().any(Option::is_none) {
            return Ok(None);
        }
        let produced: HashSet<_> = selected
            .iter()
            .map(|&index| match &plan.operations[index] {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                    output.ok_or_else(|| anyhow!("scalar operation has no result"))
                }
            })
            .collect::<Result<_>>()?;
        let mut leaf_ids = HashSet::new();
        if selected.is_empty() {
            leaf_ids.insert(root);
        }
        for &index in &selected {
            let inputs = match &plan.operations[index] {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            };
            leaf_ids.extend(
                inputs
                    .iter()
                    .filter(|&&id| !produced.contains(&id))
                    .copied(),
            );
        }
        let mut leaf_ids: Vec<_> = leaf_ids.into_iter().collect();
        leaf_ids.sort_unstable();
        let mut nodes = Vec::new();
        let controls = program.embedded_controls_for_precision(
            plan,
            &[root],
            if buffer_type == ScalarType::F32 {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            },
        );
        for id in leaf_ids {
            if !leaves.contains_key(&id) {
                return Err(anyhow!("scalar producer has an unbound leaf"));
            }
            let class = program.floating_class(id)?;
            if controls.contains(&id) {
                let bits = program
                    .embedded_scalar_control_bits(id)
                    .ok_or_else(|| anyhow!("certified control lost its original Double bits"))?;
                nodes.push(ScalarNode::Literal {
                    id,
                    value: ScalarLiteral::F64Bits(bits),
                });
                continue;
            }
            require_provider_class(class, buffer_type)?;
            if program.is_invocation_input(id) {
                nodes.push(ScalarNode::Input { id, class });
            } else if let Some(value) = plan.const_values.get(&id) {
                let Some(literal) = literal(value) else {
                    return Ok(None);
                };
                if literal.class() != class {
                    // An explicit provider precision policy can materialize a
                    // Double source literal as Single. This wire format keeps
                    // source bits, but does not yet encode that admission-time
                    // conversion as a literal operation; use legacy lowering.
                    return Ok(None);
                }
                nodes.push(ScalarNode::Literal { id, value: literal });
            } else {
                return Err(anyhow!("scalar producer has an unbound leaf"));
            }
        }
        for (index, operation) in plan.operations.iter().enumerate() {
            if !selected.contains(&index) {
                continue;
            }
            let (inputs, output) = match operation {
                FusionOp::Primitive { inputs, output, .. }
                | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
            };
            let id = output.ok_or_else(|| anyhow!("scalar operation has no output"))?;
            let class = program.floating_class(id)?;
            require_provider_class(class, buffer_type)?;
            let operation = resolved
                .get(&index)
                .copied()
                .flatten()
                .ok_or_else(|| anyhow!("selected scalar operation lost its binding"))?;
            nodes.push(ScalarNode::Apply {
                id,
                source_index: u32::try_from(index)?,
                class,
                operation,
                operands: inputs.clone(),
            });
        }
        ScalarProgram::new(
            program.cache_identity()?,
            SCALAR_PROGRAM_REVISION,
            nodes,
            root,
        )
        .map(Some)
    }

    #[cfg(test)]
    pub(crate) fn lower_checked_wgsl(
        checked: &ScalarProgram,
        buffer_type: ScalarType,
        expressions: HashMap<ValueId, String>,
    ) -> Result<Option<Self>> {
        Self::lower_checked_with_words(checked, buffer_type, expressions, HashMap::new())
    }

    pub(crate) fn lower_checked_with_words(
        checked: &ScalarProgram,
        buffer_type: ScalarType,
        mut expressions: HashMap<ValueId, String>,
        mut word_expressions: HashMap<ValueId, String>,
    ) -> Result<Option<Self>> {
        let mut body = String::new();
        let mut exact_power = false;
        let mut exact_product = false;
        let mut implementations = Vec::new();
        let capture_words = buffer_type == ScalarType::F32
            && checked.wire().nodes.iter().any(|node| {
                checked.exact_product(node.id()).is_some()
                    || matches!(
                        node,
                        ScalarNode::Apply {
                            operation: ScalarOperation::Operator(OperatorKind::ElementwisePower),
                            ..
                        }
                    )
            });
        let metadata_controls = checked.metadata_control_ids();
        for node in &checked.wire().nodes {
            match node {
                ScalarNode::Input { id, class } => {
                    require_provider_class(*class, buffer_type)?;
                    if !expressions.contains_key(id) {
                        return Err(anyhow!("missing checked scalar input expression"));
                    }
                }
                // Source text cannot promise NaN payload preservation merely
                // because the portable receipt retains its exact bits. The
                // established WGSL constant path remains in charge for now.
                ScalarNode::Literal {
                    id,
                    value: ScalarLiteral::F32Bits(bits),
                } if buffer_type == ScalarType::F32 && capture_words => {
                    let name = format!("control{id}_bits");
                    body.push_str(&format!("    let {name}: u32 = 0x{bits:08x}u;\n"));
                    expressions.insert(*id, format!("bitcast<f32>({name})"));
                    word_expressions.insert(*id, name);
                }
                ScalarNode::Literal { id, .. } => {
                    if buffer_type != ScalarType::F32 || !metadata_controls.contains(id) {
                        return Ok(None);
                    }
                }
                ScalarNode::Apply {
                    id,
                    source_index,
                    class,
                    operation,
                    operands,
                } => {
                    require_provider_class(*class, buffer_type)?;
                    if buffer_type == ScalarType::F32 {
                        if let Some((base, _, exact)) = checked.exact_product(*id) {
                            let bits = word_expressions.get(&base).ok_or_else(|| {
                                anyhow!("exact product requires word-preserving Single data")
                            })?;
                            let temporary = format!("typed{source_index}_bits");
                            body.push_str(&format!("    let {temporary}: u32 = rm_exact_product_bits({bits}, 0x{:08x}u);\n", exact.binary32_bits()));
                            expressions.insert(*id, format!("bitcast<f32>({temporary})"));
                            word_expressions.insert(*id, temporary);
                            exact_product = true;
                            implementations.push(super::product_implementation(*id));
                            continue;
                        }
                    }
                    if *operation == ScalarOperation::Operator(OperatorKind::ElementwisePower) {
                        if buffer_type != ScalarType::F32 {
                            return Ok(None);
                        }
                        let [base, exponent] = operands.as_slice() else {
                            return Err(anyhow!("certified POWER arity changed"));
                        };
                        let control = checked
                            .wire()
                            .nodes
                            .iter()
                            .find_map(|node| match node {
                                ScalarNode::Literal {
                                    id,
                                    value: ScalarLiteral::F64Bits(bits),
                                } if id == exponent => Some(
                                    runmat_accelerate_api::PowerScalarControl::DoubleBits(*bits),
                                ),
                                _ => None,
                            })
                            .and_then(runmat_accelerate_api::ExactSingleBasePower::for_control)
                            .ok_or_else(|| anyhow!("certified POWER control is unavailable"))?;
                        let base_bits = word_expressions
                            .get(base)
                            .cloned()
                            .or_else(|| {
                                expressions
                                    .get(base)
                                    .map(|value| format!("bitcast<u32>({value})"))
                            })
                            .ok_or_else(|| anyhow!("missing certified POWER base"))?;
                        let name = crate::kernel_math::power_exact::entrypoint(control);
                        let temporary = format!("typed{source_index}_bits");
                        body.push_str(&format!(
                            "    let {temporary}: u32 = {name}({base_bits});\n"
                        ));
                        expressions.insert(*id, format!("bitcast<f32>({temporary})"));
                        word_expressions.insert(*id, temporary);
                        exact_power = true;
                        implementations.push(crate::kernel_math::power_exact::implementation(
                            control, *id,
                        ));
                        continue;
                    }
                    let scalar = match class {
                        NumericClass::Single => ScalarType::F32,
                        NumericClass::Double => ScalarType::F64,
                        _ => return Err(anyhow!("nonfloating scalar operation")),
                    };
                    let name = scalar_name(scalar)?;
                    let raw_arguments = operands
                        .iter()
                        .map(|id| {
                            expressions
                                .get(id)
                                .cloned()
                                .ok_or_else(|| anyhow!("missing checked scalar operand"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let arguments = raw_arguments
                        .iter()
                        .map(|expression| format!("{name}({expression})"))
                        .collect::<Vec<_>>();
                    let expression = match (operation, arguments.as_slice()) {
                        (ScalarOperation::RealArithmetic(recipe), _) => {
                            let [left, right] = raw_arguments.as_slice() else {
                                return Err(anyhow!("represented arithmetic has invalid arity"));
                            };
                            real_arithmetic_expr(*recipe, [left, right], buffer_type)?
                        }
                        (ScalarOperation::Operator(operator), _) => {
                            crate::fusion::scalar_operator_expr(*operator, &arguments)
                                .ok_or_else(|| anyhow!("scalar operation lacks WGSL lowering"))?
                        }
                        (ScalarOperation::UnaryMath(operation), [input]) => {
                            runmat_runtime::builtins::common::spec::unary_math_expression(
                                *operation, scalar, input,
                            )
                            .map_err(|error| anyhow!("unary math expression failed: {error:?}"))?
                        }
                        (
                            ScalarOperation::Cast(NumericClass::Single | NumericClass::Double),
                            [a],
                        ) => a.clone(),
                        _ => return Err(anyhow!("scalar operation lacks WGSL lowering")),
                    };
                    let temporary = format!("typed{source_index}");
                    body.push_str(&format!("    let {temporary}: {name} = {expression};\n"));
                    if capture_words && scalar == ScalarType::F32 {
                        // Bits of the existing native result, not an accuracy
                        // certificate for the operation which produced it.
                        let bits = format!("{temporary}_bits");
                        body.push_str(&format!(
                            "    let {bits}: u32 = bitcast<u32>({temporary});\n"
                        ));
                        word_expressions.insert(*id, bits);
                    }
                    expressions.insert(*id, temporary);
                }
            }
        }
        Ok(Some(Self {
            body,
            expressions,
            helpers: Vec::new(),
            checked_program: Some(checked.clone()),
            exact_power,
            exact_product,
            word_expressions,
            implementations,
        }))
    }
}

fn require_provider_class(class: NumericClass, buffer_type: ScalarType) -> Result<()> {
    if class == NumericClass::Double && buffer_type == ScalarType::F32 {
        return Err(anyhow!(
            "provider cannot represent a Double scalar operation"
        ));
    }
    Ok(())
}

pub(super) fn literal(value: &Value) -> Option<ScalarLiteral> {
    match value {
        Value::Num(number) => Some(ScalarLiteral::F64Bits(number.to_bits())),
        Value::Tensor(tensor) if tensor.len() == 1 => match tensor.numeric_value_at(0)? {
            NumericScalar::F32(number) => Some(ScalarLiteral::F32Bits(number.to_bits())),
            NumericScalar::F64(number) => Some(ScalarLiteral::F64Bits(number.to_bits())),
            _ => None,
        },
        _ => None,
    }
}

fn primitive_operation(op: PrimitiveOp) -> Option<ScalarOperation> {
    if !matches!(
        op,
        PrimitiveOp::Add
            | PrimitiveOp::Mul
            | PrimitiveOp::Sub
            | PrimitiveOp::ElemMul
            | PrimitiveOp::ElemDiv
            | PrimitiveOp::ElemLeftDiv
            | PrimitiveOp::Neg
            | PrimitiveOp::UPlus
    ) {
        return None;
    }
    Some(ScalarOperation::Operator(op.operator_kind()))
}

pub(super) fn resolved_operation(
    program: &ResolvedProgram,
    operation: &FusionOp,
) -> Result<Option<ScalarOperation>> {
    let (inputs, output) = match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs, output)
        }
    };
    let id = output.ok_or_else(|| anyhow!("scalar operation has no output"))?;
    let class = program.floating_class(id)?;
    // Only captured canonical arithmetic evidence can upgrade the existing
    // closed operator to the certified family. A name/template is not proof.
    if program.exact_single_base_power(id).is_some() {
        return Ok(Some(ScalarOperation::Operator(
            OperatorKind::ElementwisePower,
        )));
    }
    let selected_identity = if let Some(declaration) = program.declaration(id) {
        let entry = program
            .catalog_entry(id)
            .ok_or_else(|| anyhow!("missing captured scalar binding"))?;
        if entry.contract.purity != runmat_builtins::BuiltinPurity::Pure
            || entry.contract.workspace_effect.is_some()
            || entry.contract.environment_effect.is_some()
        {
            return Err(anyhow!("scalar program requires an effect-free binding"));
        }
        let precision = match class {
            NumericClass::Single => ScalarType::F32,
            NumericClass::Double => ScalarType::F64,
            _ => return Err(anyhow!("scalar operation has no floating precision")),
        };
        let scalar_matrix_product = declaration
            .semantics
            .and_then(|semantics| semantics.scalar_operation)
            == Some(ScalarOperation::Operator(OperatorKind::MatrixMultiply))
            && matches!(inputs.as_slice(), [left, right]
                if runmat_types::RealFloatingRecipe::for_facts(OperatorKind::MatrixMultiply,
                    program.fact(*left)?, program.fact(*right)?).is_some());
        if !scalar_matrix_product
            && !declaration
                .spec
                .elementwise
                .as_ref()
                .is_some_and(|template| template.scalar_precisions.contains(&precision))
        {
            return Err(anyhow!("binding has no matching scalar template"));
        }
        declaration
            .semantics
            .and_then(|semantics| semantics.scalar_operation)
    } else {
        let FusionOp::Primitive { op, .. } = operation else {
            return Err(anyhow!("builtin has no captured scalar binding"));
        };
        primitive_operation(*op)
    };
    // Primitive and captured canonical operators share the represented host
    // recipe. A name/template alone cannot assert computation precision.
    let identity = if let Some(ScalarOperation::Operator(operator)) = selected_identity {
        if matches!(
            operator,
            OperatorKind::Add
                | OperatorKind::MatrixMultiply
                | OperatorKind::Subtract
                | OperatorKind::ElementwiseMultiply
                | OperatorKind::ElementwiseDivide
                | OperatorKind::ElementwiseLeftDivide
        ) {
            let [left, right] = inputs.as_slice() else {
                return Err(anyhow!("real scalar arithmetic has invalid arity"));
            };
            let recipe = runmat_types::RealFloatingRecipe::for_facts(
                operator,
                program.fact(*left)?,
                program.fact(*right)?,
            )
            .ok_or_else(|| anyhow!("real scalar arithmetic has no represented recipe"))?;
            if class != recipe.result_class() {
                return Err(anyhow!("real scalar arithmetic result class mismatch"));
            }
            Some(ScalarOperation::RealArithmetic(recipe))
        } else {
            selected_identity
        }
    } else {
        selected_identity
    };
    if let Some(identity) = identity {
        if identity.arity() != Some(inputs.len()) {
            return Err(anyhow!("captured scalar operation arity mismatch"));
        }
        if matches!(identity, ScalarOperation::Cast(target) if target != class) {
            return Err(anyhow!("captured scalar cast result class mismatch"));
        }
        if matches!(
            identity,
            ScalarOperation::Operator(_) | ScalarOperation::UnaryMath(_)
        ) {
            for input in inputs {
                if program.floating_class(*input)? != class {
                    return Err(anyhow!("captured scalar operand class mismatch"));
                }
            }
        }
    }
    Ok(identity)
}

/// The only WGSL renderer for represented real arithmetic, shared by checked
/// and legacy/multi-output paths. Never cast Double data to the result class
/// before the operation, and never run Double computation in an F32 data ABI.
pub(super) fn real_arithmetic_expr(
    recipe: runmat_types::RealFloatingRecipe,
    arguments: [&str; 2],
    buffer_type: ScalarType,
) -> Result<String> {
    require_provider_class(recipe.computation_class(), buffer_type)?;
    let computation = match recipe.computation_class() {
        NumericClass::Single => "f32",
        NumericClass::Double => "f64",
        _ => return Err(anyhow!("unsupported represented computation class")),
    };
    let result = match recipe.result_class() {
        NumericClass::Single => "f32",
        NumericClass::Double => "f64",
        _ => return Err(anyhow!("unsupported represented result class")),
    };
    let arguments = arguments.map(|argument| format!("{computation}({argument})"));
    let expression = crate::fusion::scalar_operator_expr(recipe.operator(), &arguments)
        .ok_or_else(|| anyhow!("represented arithmetic lacks WGSL lowering"))?;
    if recipe.computation_class() == recipe.result_class() {
        Ok(expression)
    } else {
        Ok(format!("{result}({expression})"))
    }
}
