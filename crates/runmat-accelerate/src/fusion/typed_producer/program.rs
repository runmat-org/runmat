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
        for id in leaf_ids {
            if !leaves.contains_key(&id) {
                return Err(anyhow!("scalar producer has an unbound leaf"));
            }
            let class = program.floating_class(id)?;
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

    pub(crate) fn lower_checked_wgsl(
        checked: &ScalarProgram,
        buffer_type: ScalarType,
        mut expressions: HashMap<ValueId, String>,
    ) -> Result<Option<Self>> {
        let mut body = String::new();
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
                ScalarNode::Literal { .. } => {
                    return Ok(None);
                }
                ScalarNode::Apply {
                    id,
                    source_index,
                    class,
                    operation,
                    operands,
                } => {
                    require_provider_class(*class, buffer_type)?;
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
                        (ScalarOperation::MixedRealAddToSingle, _) => {
                            let [left, right] = raw_arguments.as_slice() else {
                                return Err(anyhow!("mixed real addition has invalid arity"));
                            };
                            format!("f32((f64({left}) + f64({right})))")
                        }
                        (ScalarOperation::Operator(operator), _) => {
                            crate::fusion::scalar_operator_expr(*operator, &arguments)
                                .ok_or_else(|| anyhow!("scalar operation lacks WGSL lowering"))?
                        }
                        (
                            ScalarOperation::Cast(NumericClass::Single | NumericClass::Double),
                            [a],
                        ) => a.clone(),
                        _ => return Err(anyhow!("scalar operation lacks WGSL lowering")),
                    };
                    let temporary = format!("typed{source_index}");
                    body.push_str(&format!("    let {temporary}: {name} = {expression};\n"));
                    expressions.insert(*id, temporary);
                }
            }
        }
        Ok(Some(Self {
            body,
            expressions,
            helpers: Vec::new(),
            checked_program: Some(checked.clone()),
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

fn literal(value: &Value) -> Option<ScalarLiteral> {
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
        if !declaration
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
    // A primitive Add and a selected canonical declaration of that same
    // closed scalar operation share one represented-arithmetic owner. A
    // textual builtin name or an unselected implementation cannot claim it.
    let identity = if selected_identity == Some(ScalarOperation::Operator(OperatorKind::Add)) {
        let [left, right] = inputs.as_slice() else {
            return Err(anyhow!("scalar Add has invalid arity"));
        };
        match runmat_types::real_floating_add_mode(program.fact(*left)?, program.fact(*right)?) {
            Some(runmat_types::RealFloatingAddMode::MixedDoubleThenSingle) => {
                if class != NumericClass::Single {
                    return Err(anyhow!("mixed real Add result class mismatch"));
                }
                Some(ScalarOperation::MixedRealAddToSingle)
            }
            _ => selected_identity,
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
        if matches!(identity, ScalarOperation::Operator(_)) {
            for input in inputs {
                if program.floating_class(*input)? != class {
                    return Err(anyhow!("captured scalar operand class mismatch"));
                }
            }
        }
    }
    Ok(identity)
}
