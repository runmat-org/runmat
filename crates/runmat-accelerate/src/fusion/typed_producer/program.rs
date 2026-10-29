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
        let leaf_capacity = plan
            .inputs
            .len()
            .checked_add(plan.const_values.len())
            .ok_or_else(|| anyhow!("scalar producer leaf count overflows"))?;
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(leaf_capacity)?;
        leaves.extend(plan.inputs.iter().chain(plan.const_values.keys()).copied());
        leaves.sort_unstable();
        leaves.dedup();
        let selected = producer_indices(plan, &leaves, &[root])?;
        // Check every selected advertised binding before an unsupported legacy
        // operation can request fallback. A malformed closed binding must not
        // execute through its older expression template by accident.
        let resolved = selected_resolution_receipt(plan, program, &selected)?;
        if selected.iter().any(|index| resolved[index].is_none()) {
            return Ok(None);
        }
        let mut produced = Vec::new();
        produced.try_reserve_exact(selected.len())?;
        for index in selected.iter() {
            produced.push(
                (match &plan.operations[index] {
                    FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                        output.ok_or_else(|| anyhow!("scalar operation has no result"))
                    }
                })?,
            );
        }
        produced.sort_unstable();
        let leaf_capacity = plan
            .operations
            .iter()
            .try_fold(1_usize, |count, operation| {
                count.checked_add(operands(operation).0.len())
            })
            .ok_or_else(|| anyhow!("scalar producer leaf count overflows"))?;
        let mut leaf_ids = Vec::new();
        leaf_ids.try_reserve_exact(leaf_capacity)?;
        if selected.is_empty() {
            leaf_ids.push(root);
        }
        for index in selected.iter() {
            let inputs = match &plan.operations[index] {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            };
            leaf_ids.extend(
                inputs
                    .iter()
                    .filter(|&&id| produced.binary_search(&id).is_err())
                    .copied(),
            );
        }
        leaf_ids.sort_unstable();
        leaf_ids.dedup();
        let mut nodes = Vec::new();
        nodes.try_reserve_exact(
            leaf_capacity
                .checked_add(selected.len())
                .ok_or_else(|| anyhow!("scalar program node count overflows"))?,
        )?;
        let controls = program.embedded_controls_sorted_for_precision(
            plan,
            &[root],
            if buffer_type == ScalarType::F32 {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            },
        )?;
        for id in leaf_ids {
            if leaves.binary_search(&id).is_err() {
                return Err(anyhow!("scalar producer has an unbound leaf"));
            }
            let mut class = program.floating_class(id)?;
            if controls.binary_search(&id).is_ok() {
                let bits = program
                    .embedded_scalar_control_bits(id)
                    .ok_or_else(|| anyhow!("certified control lost its original Double bits"))?;
                nodes.push(ScalarNode::Literal {
                    id,
                    value: ScalarLiteral::F64Bits(bits),
                });
                continue;
            }
            if class == NumericClass::Double && buffer_type == ScalarType::F32 {
                if let Some(proof) = program.explicit_host_single_input(plan, id) {
                    class = proof.represented_class_for(program, id)?;
                }
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
                .get(index)
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
        let source = checked.wire();
        // Tests adapt their expression maps to the same indexed owner used
        // by production. No map-based lowering path exists in production.
        let mut ids = source.nodes.iter().map(ScalarNode::id).collect::<Vec<_>>();
        ids.extend(expressions.keys().copied());
        ids.sort_unstable();
        ids.dedup();
        let mut values = ExpressionSlots::from_sorted_ids(&ids);
        let words = ExpressionSlots::from_sorted_ids(&ids);
        for (id, expression) in expressions {
            values.insert(id, expression)?;
        }
        Self::lower_checked_indexed_with_words(checked, buffer_type, values, words, None)
    }

    pub(crate) fn lower_checked_indexed_with_words(
        checked: &ScalarProgram,
        buffer_type: ScalarType,
        expressions: ExpressionSlots,
        word_expressions: ExpressionSlots,
        body_capacity: Option<usize>,
    ) -> Result<Option<Self>> {
        Self::lower_checked_indexed_with_words_owned(
            checked.clone(),
            buffer_type,
            expressions,
            word_expressions,
            body_capacity,
        )
    }

    pub(crate) fn lower_checked_indexed_with_words_owned(
        checked: ScalarProgram,
        buffer_type: ScalarType,
        mut expressions: ExpressionSlots,
        mut word_expressions: ExpressionSlots,
        body_capacity: Option<usize>,
    ) -> Result<Option<Self>> {
        let mut body = super::super::wgsl_source::BoundedWgslBody::new(body_capacity)?;
        let mut exact_power = false;
        let mut exact_product = false;
        let mut implementations = Vec::new();
        implementations.try_reserve_exact(checked.wire().nodes.len())?;
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
        let metadata_controls = checked.metadata_control_ids_sorted()?;
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
                    let name =
                        super::checked_resources::exact_format(format_args!("control{id}_bits"))?;
                    writeln!(&mut body, "    let {name}: u32 = 0x{bits:08x}u;")?;
                    expressions.insert(
                        *id,
                        super::checked_resources::exact_format(format_args!(
                            "bitcast<f32>({name})"
                        ))?,
                    )?;
                    word_expressions.insert(*id, name)?;
                }
                ScalarNode::Literal { id, .. } => {
                    if buffer_type != ScalarType::F32
                        || metadata_controls.binary_search(id).is_err()
                    {
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
                            let temporary = super::checked_resources::exact_format(format_args!(
                                "typed{source_index}_bits"
                            ))?;
                            writeln!(&mut body, "    let {temporary}: u32 = rm_exact_product_bits({bits}, 0x{:08x}u);", exact.binary32_bits())?;
                            expressions.insert(
                                *id,
                                super::checked_resources::exact_format(format_args!(
                                    "bitcast<f32>({temporary})"
                                ))?,
                            )?;
                            word_expressions.insert(*id, temporary)?;
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
                        let base_bits = if let Some(bits) = word_expressions.get(base) {
                            bits.clone()
                        } else if let Some(value) = expressions.get(base) {
                            super::checked_resources::exact_format(format_args!(
                                "bitcast<u32>({value})"
                            ))?
                        } else {
                            return Err(anyhow!("missing certified POWER base"));
                        };
                        let name = crate::kernel_math::power_exact::entrypoint(control);
                        let temporary = super::checked_resources::exact_format(format_args!(
                            "typed{source_index}_bits"
                        ))?;
                        writeln!(&mut body, "    let {temporary}: u32 = {name}({base_bits});")?;
                        expressions.insert(
                            *id,
                            super::checked_resources::exact_format(format_args!(
                                "bitcast<f32>({temporary})"
                            ))?,
                        )?;
                        word_expressions.insert(*id, temporary)?;
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
                    let mut raw_arguments = Vec::new();
                    raw_arguments.try_reserve_exact(operands.len())?;
                    for id in operands {
                        raw_arguments.push(
                            expressions
                                .get(id)
                                .map(String::as_str)
                                .ok_or_else(|| anyhow!("missing checked scalar operand"))?,
                        );
                    }
                    let mut arguments = Vec::new();
                    arguments.try_reserve_exact(raw_arguments.len())?;
                    for expression in &raw_arguments {
                        arguments.push(super::checked_resources::exact_format(format_args!(
                            "{name}({expression})"
                        ))?);
                    }
                    let expression = match (operation, arguments.as_slice()) {
                        (ScalarOperation::RealArithmetic(recipe), _) => {
                            let [left, right] = raw_arguments.as_slice() else {
                                return Err(anyhow!("represented arithmetic has invalid arity"));
                            };
                            real_arithmetic_expr(*recipe, [*left, *right], buffer_type)?
                        }
                        (ScalarOperation::Operator(operator), _) => {
                            crate::fusion::scalar_operator_expr(*operator, &arguments)?
                                .ok_or_else(|| anyhow!("scalar operation lacks WGSL lowering"))?
                        }
                        (ScalarOperation::UnaryMath(operation), [input]) => {
                            runmat_runtime::builtins::common::spec::unary_math_expression(
                                *operation, scalar, input,
                            )
                            .map_err(anyhow::Error::from)?
                        }
                        (
                            ScalarOperation::Cast(NumericClass::Single | NumericClass::Double),
                            [a],
                        ) => a.clone(),
                        _ => return Err(anyhow!("scalar operation lacks WGSL lowering")),
                    };
                    let temporary = super::checked_resources::exact_format(format_args!(
                        "typed{source_index}"
                    ))?;
                    writeln!(&mut body, "    let {temporary}: {name} = {expression};")?;
                    if capture_words && scalar == ScalarType::F32 {
                        // Bits of the existing native result, not an accuracy
                        // certificate for the operation which produced it.
                        let bits = super::checked_resources::exact_format(format_args!(
                            "{temporary}_bits"
                        ))?;
                        writeln!(
                            &mut body,
                            "    let {bits}: u32 = bitcast<u32>({temporary});"
                        )?;
                        word_expressions.insert(*id, bits)?;
                    }
                    expressions.insert(*id, temporary)?;
                }
            }
        }
        Ok(Some(Self {
            body: body.into_source(),
            expressions,
            helpers: Vec::new(),
            checked_program: Some(checked),
            exact_power,
            exact_product,
            word_expressions,
            implementations,
        }))
    }
}

/// Validate every selected closed binding before any legacy expression can
/// decline. An uncounted legacy template without a closed scalar operation is
/// not a malformed closed binding: it may still use another admitted ABI.
/// Keep the represented
/// facts local to this scan; only the indexed identities survive emission.
pub(super) fn selected_resolution_receipt(
    plan: &FusionGroupPlan,
    program: &ResolvedProgram,
    selected: &SelectedProducerIndices,
) -> Result<Vec<Option<ScalarOperation>>> {
    let mut resolved = Vec::new();
    resolved.try_reserve_exact(plan.operations.len())?;
    resolved.resize(plan.operations.len(), None);
    let mut represented = Vec::new();
    represented.try_reserve_exact(selected.len())?;
    for (index, operation) in plan.operations.iter().enumerate() {
        if !selected.contains(&index) {
            continue;
        }
        let output = match operation {
            FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                output.ok_or_else(|| anyhow!("scalar producer has no output"))?
            }
        };
        let identity = match resolved_operation_with_represented(program, operation, &represented) {
            Ok(identity) => identity,
            Err(error)
                if program
                    .declaration(output)
                    .and_then(|declaration| declaration.semantics)
                    .is_none_or(|semantics| semantics.scalar_operation.is_none())
                    && error
                        .downcast_ref::<super::geometry::CountedRecipeUnavailable>()
                        .is_some() =>
            {
                None
            }
            Err(error) => return Err(error),
        };
        if let Some(fact) = represented_output_fact(program, operation, identity)? {
            let position = represented
                .binary_search_by_key(&output, |(id, _)| *id)
                .unwrap_or_else(|position| position);
            if represented
                .get(position)
                .is_some_and(|(id, _)| *id == output)
            {
                return Err(anyhow!("scalar producer has duplicate output identity"));
            }
            represented.insert(position, (output, fact));
        }
        resolved[index] = identity;
    }
    Ok(resolved)
}

pub(super) fn require_provider_class(class: NumericClass, buffer_type: ScalarType) -> Result<()> {
    if class == NumericClass::Double && buffer_type == ScalarType::F32 {
        return Err(super::geometry::CountedRecipeUnavailable(
            "provider cannot represent a Double scalar operation",
        )
        .into());
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

pub(super) fn resolved_operation_with_represented(
    program: &ResolvedProgram,
    operation: &FusionOp,
    represented: &[(ValueId, runmat_types::ValueFact)],
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
                    represented_operand(program, represented, *left)?,
                    represented_operand(program, represented, *right)?).is_some());
        if !scalar_matrix_product
            && !declaration
                .spec
                .elementwise
                .as_ref()
                .is_some_and(|template| template.scalar_precisions.contains(&precision))
        {
            return Err(super::geometry::CountedRecipeUnavailable(
                "binding has no matching scalar template",
            )
            .into());
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
                represented_operand(program, represented, *left)?,
                represented_operand(program, represented, *right)?,
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

fn represented_operand<'a>(
    program: &'a ResolvedProgram,
    represented: &'a [(ValueId, runmat_types::ValueFact)],
    id: ValueId,
) -> Result<&'a runmat_types::ValueFact> {
    match represented.binary_search_by_key(&id, |(value_id, _)| *value_id) {
        Ok(index) => Ok(&represented[index].1),
        Err(_) => program.fact(id),
    }
}

/// A selected scalar operation constructs a fresh numerical intermediate;
/// its eventual host/device publication is not an input-carrier fact. This
/// local projection is shared by checked receipts and legacy WGSL emission.
pub(super) fn represented_output_fact(
    program: &ResolvedProgram,
    operation: &FusionOp,
    identity: Option<ScalarOperation>,
) -> Result<Option<runmat_types::ValueFact>> {
    let output = match operation {
        FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
            output.ok_or_else(|| anyhow!("represented operation has no output"))?
        }
    };
    let selected_legacy_template = identity.is_none()
        && matches!(operation, FusionOp::Builtin { .. })
        && program.declaration(output).is_some_and(|declaration| {
            let precision = match program.floating_class(output) {
                Ok(NumericClass::Single) => ScalarType::F32,
                Ok(NumericClass::Double) => ScalarType::F64,
                _ => return false,
            };
            declaration
                .spec
                .elementwise
                .as_ref()
                .is_some_and(|template| template.scalar_precisions.contains(&precision))
        });
    if identity.is_none() && !selected_legacy_template {
        return Ok(None);
    }
    let mut fact = program.fact(output)?.clone();
    if fact.numeric().is_some_and(|numeric| {
        numeric.domain == runmat_types::NumericDomain::Real
            && matches!(numeric.class, NumericClass::Single | NumericClass::Double)
    }) && matches!(
        fact.storage,
        runmat_types::StorageFact::Unknown | runmat_types::StorageFact::Opaque
    ) {
        fact.storage = runmat_types::StorageFact::Dense;
        return Ok(Some(fact));
    }
    Ok(None)
}

/// The only WGSL renderer for represented real arithmetic, shared by checked
/// and legacy/multi-output paths. Never cast Double data to the result class
/// before the operation, and never run Double computation in an F32 data ABI.
pub(super) fn real_arithmetic_expr(
    recipe: runmat_types::RealFloatingRecipe,
    arguments: [&str; 2],
    buffer_type: ScalarType,
) -> Result<String> {
    let mut measured = crate::fusion::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
    write_real_arithmetic_expr(&mut measured, recipe, arguments, buffer_type)?;
    let mut expression = String::new();
    expression.try_reserve_exact(measured.bytes())?;
    write_real_arithmetic_expr(&mut expression, recipe, arguments, buffer_type)?;
    if expression.len() != measured.bytes() {
        return Err(anyhow!("represented arithmetic expression changed length"));
    }
    Ok(expression)
}

pub(super) fn write_real_arithmetic_expr(
    writer: &mut impl std::fmt::Write,
    recipe: runmat_types::RealFloatingRecipe,
    arguments: [impl std::fmt::Display; 2],
    buffer_type: ScalarType,
) -> Result<()> {
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
    // The recipe constructor admits exactly these operators. Keep a malformed
    // serialized recipe structural, before writing any source bytes.
    if !matches!(
        recipe.operator(),
        OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::ElementwiseDivide
            | OperatorKind::ElementwiseLeftDivide
    ) {
        return Err(anyhow!("represented arithmetic lacks WGSL lowering"));
    }
    struct Cast<'a, T>(&'a str, &'a T);
    impl<T: std::fmt::Display> std::fmt::Display for Cast<'_, T> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}({})", self.0, self.1)
        }
    }
    let casted = [
        Cast(computation, &arguments[0]),
        Cast(computation, &arguments[1]),
    ];
    if recipe.computation_class() != recipe.result_class() {
        write!(writer, "{result}(")?;
    }
    // This error now comes only from the counted/bounded writer, not from an
    // unsupported operator; retain it for finite admission classification.
    crate::fusion::write_scalar_operator_expr(writer, recipe.operator(), &casted)?;
    if recipe.computation_class() != recipe.result_class() {
        writer.write_char(')')?;
    }
    Ok(())
}
