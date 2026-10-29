//! Allocation-free WGSL operation geometry for a resolved, admitted program.
//! This deliberately counts through the same declaration and scalar writers
//! used by emission; no per-builtin byte multiplier is an authority here.

use anyhow::{anyhow, Result};
use runmat_runtime::builtins::common::spec::{
    write_unary_math_scalar_expression, FusionMathHelper, ScalarType,
};
use runmat_types::{NumericClass, ScalarOperation, ValueFact};
use std::fmt::Write as _;
use std::mem::size_of;

use super::program;
use super::{scalar_name, FusionGroupPlan, FusionOp, ResolvedProgram};
use crate::fusion::wgsl_source::{CountedWgslSource, MeasuredFragment};

/// This ABI has no selected counted expression. Unlike a writer failure or
/// checked-size overflow, another admitted ABI may still be usable.
#[derive(Debug)]
pub(crate) struct CountedRecipeUnavailable(pub(crate) &'static str);

impl std::fmt::Display for CountedRecipeUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for CountedRecipeUnavailable {}

fn measured(write: impl FnOnce(&mut CountedWgslSource) -> Result<()>) -> Result<usize> {
    let mut writer = CountedWgslSource::with_preamble_bytes(0);
    write(&mut writer)?;
    Ok(writer.bytes())
}

fn represented_type(class: NumericClass) -> Result<ScalarType> {
    match class {
        NumericClass::Single => Ok(ScalarType::F32),
        NumericClass::Double => Ok(ScalarType::F64),
        _ => Err(anyhow!("finite producer has a nonfloating operation")),
    }
}

/// Maximum expression bytes across the actual checked scalar spelling and
/// the captured template spelling. Only one is emitted, but checked lowering
/// may decline and retain the legacy path for this same operation.
fn expression_bytes(
    program: &ResolvedProgram,
    operation: &FusionOp,
    buffer_type: ScalarType,
    operand_bytes: usize,
    identity: Option<ScalarOperation>,
) -> Result<usize> {
    let (inputs, output) = match operation {
        FusionOp::Primitive { inputs, output, .. } | FusionOp::Builtin { inputs, output, .. } => {
            (inputs, output)
        }
    };
    if inputs.len() > 2 {
        return Err(anyhow!("finite producer expression has unsupported arity"));
    }
    let output = output.ok_or_else(|| anyhow!("finite producer has no output"))?;
    let scalar = represented_type(program.floating_class(output)?)?;
    let casted_bytes = measured(|writer| {
        write!(
            writer,
            "{}({})",
            scalar_name(scalar)?,
            MeasuredFragment(operand_bytes)
        )?;
        Ok(())
    })?;
    let mut upper = 0usize;
    if let Some(identity) = identity {
        let checked = match identity {
            ScalarOperation::RealArithmetic(recipe) => measured(|writer| {
                program::write_real_arithmetic_expr(
                    writer,
                    recipe,
                    [
                        MeasuredFragment(operand_bytes),
                        MeasuredFragment(operand_bytes),
                    ],
                    buffer_type,
                )
            })?,
            ScalarOperation::Operator(runmat_types::OperatorKind::ElementwisePower) => 0,
            ScalarOperation::Operator(operator) => measured(|writer| {
                let operands = [
                    MeasuredFragment(casted_bytes),
                    MeasuredFragment(casted_bytes),
                ];
                crate::fusion::write_scalar_operator_expr(
                    writer,
                    operator,
                    &operands[..inputs.len()],
                )?;
                Ok(())
            })?,
            ScalarOperation::UnaryMath(operation) => measured(|writer| {
                write_unary_math_scalar_expression(
                    operation,
                    scalar,
                    MeasuredFragment(casted_bytes),
                    writer,
                )?;
                Ok(())
            })?,
            ScalarOperation::Cast(_) => casted_bytes,
        };
        upper = upper.max(checked);
    }
    if let Some(declaration) = program.declaration(output) {
        if let Some(template) = declaration.spec.elementwise.as_ref() {
            let lengths = [casted_bytes, casted_bytes];
            upper = upper.max(
                template
                    .wgsl_body
                    .measured_bytes_from_lengths(scalar, &lengths[..inputs.len()])?,
            );
        }
    }
    if upper == 0 && program.exact_single_base_power(output).is_none() {
        return Err(
            CountedRecipeUnavailable("finite producer has no counted scalar expression").into(),
        );
    }
    Ok(upper)
}

pub(in crate::fusion) fn max_operand_bytes(
    plan: &FusionGroupPlan,
    input_count: usize,
) -> Result<usize> {
    let max_id = plan
        .inputs
        .iter()
        .chain(plan.const_values.keys())
        .chain(
            plan.operations
                .iter()
                .filter_map(|operation| match operation {
                    FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                        output.as_ref()
                    }
                }),
        )
        .copied()
        .max()
        .unwrap_or(0);
    let last_input = input_count.saturating_sub(1);
    let last_operation = plan.operations.len().saturating_sub(1);
    let candidates = [
        measured(|writer| {
            write!(writer, "input{last_input}.data[i{last_input}]")?;
            Ok(())
        })?,
        measured(|writer| {
            write!(
                writer,
                "bitcast<f32>(input{last_input}.data[i{last_input}])"
            )?;
            Ok(())
        })?,
        measured(|writer| {
            write!(writer, "control{max_id}")?;
            Ok(())
        })?,
        measured(|writer| {
            write!(writer, "bitcast<f32>(control{max_id}_bits)")?;
            Ok(())
        })?,
        measured(|writer| {
            write!(writer, "typed{last_operation}")?;
            Ok(())
        })?,
        measured(|writer| {
            write!(writer, "bitcast<f32>(typed{last_operation}_bits)")?;
            Ok(())
        })?,
    ];
    Ok(candidates.into_iter().max().unwrap_or(0))
}

/// A checked upper bound on the one retained producer body. Every source
/// operation is counted even when the final root slice selects fewer. An
/// authenticated exact-product branch replaces its unrepresentable ordinary
/// expression exactly as it does in the emitter; exact POWER and remaining
/// alternatives retain their conservative maximum because only one branch is
/// emitted per operation.
/// Measure in source order with the same selected represented-output facts as
/// checked and legacy emission. This temporary projection is charged to the
/// live phase-one receipt before the phase-two producer reservation exists.
pub(in crate::fusion) fn body_upper_bytes(
    plan: &FusionGroupPlan,
    outputs: &[crate::graph::ValueId],
    program: &ResolvedProgram,
    buffer_type: ScalarType,
    input_count: usize,
) -> Result<(usize, usize)> {
    let operand_bytes = max_operand_bytes(plan, input_count)?;
    let mut total = 0usize;
    let mut max_expression = 0usize;
    let mut represented: Vec<(crate::graph::ValueId, ValueFact)> = Vec::new();
    represented.try_reserve_exact(plan.operations.len())?;
    let embedded_controls = if buffer_type == ScalarType::F32 {
        program.embedded_controls_sorted_for_precision(
            plan,
            outputs,
            runmat_accelerate_api::ProviderPrecision::F32,
        )?
    } else {
        Vec::new()
    };
    for &id in plan.const_values.keys() {
        let word = measured(|writer| {
            write!(writer, "    let control{id}_bits: u32 = 0xffffffffu;\n    let control{id} = bitcast<f32>(control{id}_bits);\n")?;
            Ok(())
        })?;
        let double = measured(|writer| {
            writeln!(
                writer,
                "    let control{id} = runmat_f64_from_words(0xffffffffu, 0xffffffffu);"
            )?;
            Ok(())
        })?;
        total = total
            .checked_add(word.max(double))
            .ok_or_else(|| anyhow!("finite producer constant body overflows"))?;
    }
    for (index, operation) in plan.operations.iter().enumerate() {
        let output = match operation {
            FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                output.ok_or_else(|| anyhow!("finite producer operation has no output"))?
            }
        };
        let scalar = represented_type(program.floating_class(output)?)?;
        let name = scalar_name(scalar)?;
        let identity =
            program::resolved_operation_with_represented(program, operation, &represented)?;
        let exact_product = (buffer_type == ScalarType::F32)
            .then(|| program.exact_product(output))
            .flatten()
            .filter(|receipt| embedded_controls.binary_search(&receipt.control).is_ok());
        if exact_product.is_none() {
            program::require_provider_class(program.floating_class(output)?, buffer_type)?;
        }
        let mut upper = if exact_product.is_some() {
            // The emitter consumes the authenticated Double control as
            // metadata and writes the word-preserving product directly. Its
            // ordinary mixed-Double expression is unreachable in this ABI
            // and may legitimately be unrepresentable by an F32 provider.
            max_expression = max_expression.max(operand_bytes);
            measured(|writer| {
                writeln!(
                    writer,
                    "    let typed{index}_bits: u32 = rm_exact_product_bits({}, 0xffffffffu);",
                    MeasuredFragment(operand_bytes)
                )?;
                Ok(())
            })?
        } else {
            let ordinary =
                expression_bytes(program, operation, buffer_type, operand_bytes, identity)?;
            max_expression = max_expression.max(ordinary);
            measured(|writer| {
                writeln!(
                    writer,
                    "    let typed{index}: {name} = {};",
                    MeasuredFragment(ordinary)
                )?;
                Ok(())
            })?
        };
        if buffer_type == ScalarType::F32 && exact_product.is_none() {
            let capture = measured(|writer| {
                writeln!(
                    writer,
                    "    let typed{index}_bits: u32 = bitcast<u32>(typed{index});"
                )?;
                Ok(())
            })?;
            upper = upper
                .checked_add(capture)
                .ok_or_else(|| anyhow!("finite producer word capture overflows"))?;
            if let Some(control) = program.exact_single_base_power(output) {
                let entry = crate::kernel_math::power_exact::entrypoint(control);
                let exact_power = measured(|writer| {
                    writeln!(
                        writer,
                        "    let typed{index}_bits: u32 = {entry}({});",
                        MeasuredFragment(operand_bytes)
                    )?;
                    Ok(())
                })?;
                upper = upper.max(exact_power);
            }
        }
        total = total
            .checked_add(upper)
            .ok_or_else(|| anyhow!("finite producer operation body overflows"))?;
        if let Some(fact) = program::represented_output_fact(program, operation, identity)? {
            let position = represented
                .binary_search_by_key(&output, |(id, _)| *id)
                .unwrap_or_else(|position| position);
            if represented
                .get(position)
                .is_some_and(|(id, _)| *id == output)
            {
                return Err(anyhow!("finite producer has duplicate output identity"));
            }
            represented.insert(position, (output, fact));
        }
    }
    Ok((total, max_expression))
}

/// Shared helper writers count their own formatted bytes. Include every
/// operation helper once per precision regardless of the eventual root slice.
pub(in crate::fusion) fn preamble_upper_bytes(
    plan: &FusionGroupPlan,
    program: &ResolvedProgram,
) -> Result<usize> {
    let mut writer = CountedWgslSource::with_preamble_bytes(0);
    writeln!(&mut writer, "// {}", program.cache_identity()?)?;
    let mut exact_power = false;
    let mut exact_product = false;
    let mut helpers = [[false; 2]; 3];
    for operation in &plan.operations {
        let output = match operation {
            FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                output.ok_or_else(|| anyhow!("finite producer operation has no output"))?
            }
        };
        exact_power |= program.exact_single_base_power(output).is_some();
        exact_product |= program.exact_product(output).is_some();
        if let Some(declaration) = program.declaration(output) {
            if let Some(semantics) = declaration.semantics {
                let class = represented_type(program.floating_class(output)?)?;
                let precision = usize::from(class == ScalarType::F64);
                for helper in semantics.helpers {
                    let kind = match helper {
                        FusionMathHelper::RealPower => 0,
                        FusionMathHelper::Round => 1,
                        FusionMathHelper::FloatingRemainder => 2,
                    };
                    helpers[kind][precision] = true;
                }
            }
        }
    }
    if exact_power || exact_product {
        writer.write_str(crate::kernel_math::power_exact::functions())?;
    }
    if exact_product {
        writer.write_str(crate::kernel_math::product_exact::functions())?;
    }
    for precision in [
        runmat_accelerate_api::ProviderPrecision::F32,
        runmat_accelerate_api::ProviderPrecision::F64,
    ] {
        let index = usize::from(precision == runmat_accelerate_api::ProviderPrecision::F64);
        if helpers[0][index] {
            crate::kernel_math::power::write_functions(precision, &mut writer)?;
        }
        if helpers[1][index] {
            crate::kernel_math::rounding::write_functions(precision, &mut writer)?;
        }
        if helpers[2][index] {
            crate::kernel_math::floating_remainder::write_functions(precision, &mut writer)?;
        }
    }
    writer.write_char('\n')?;
    Ok(writer.bytes())
}

#[derive(Default)]
struct RequestedBytes(u64);

impl RequestedBytes {
    fn add(&mut self, bytes: usize) -> Result<()> {
        self.0 = self
            .0
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow!("finite producer requested bytes overflow"))?;
        Ok(())
    }

    fn vector<T>(&mut self, count: usize) -> Result<()> {
        self.add(
            count
                .checked_mul(size_of::<T>())
                .ok_or_else(|| anyhow!("finite producer vector bytes overflow"))?,
        )
    }
}

/// Requested capacities of producer-owned vectors and retained short strings.
/// All classes of storage have a named source owner; source/body/preamble text
/// is accounted separately by the same WGSL writers above. V3's selected
/// represented-output facts are included even before central composition.
pub(in crate::fusion) fn structural_upper_bytes(
    plan: &FusionGroupPlan,
    outputs: usize,
    max_rank: usize,
    max_operand: usize,
    max_expression: usize,
) -> Result<u64> {
    use crate::graph::ValueId;
    use runmat_accelerate_api::scalar_program::{EmittedScalarImplementation, ScalarNode};
    use runmat_runtime::builtins::common::spec::FusionMathHelper;
    use runmat_types::{DimensionFact, NumericClass, ScalarOperation, ValueFact};

    let input_count = plan.inputs.len();
    let operation_count = plan.operations.len();
    let constant_count = plan.const_values.len();
    let slots = input_count
        .checked_add(constant_count)
        .and_then(|count| count.checked_add(operation_count))
        .ok_or_else(|| anyhow!("finite producer slot count overflows"))?;
    let operands = plan
        .operations
        .iter()
        .try_fold(0usize, |count, operation| {
            let inputs = match operation {
                FusionOp::Primitive { inputs, .. } | FusionOp::Builtin { inputs, .. } => inputs,
            };
            count.checked_add(inputs.len())
        })
        .ok_or_else(|| anyhow!("finite producer operand count overflows"))?;
    let pending = outputs
        .checked_add(operands)
        .ok_or_else(|| anyhow!("finite producer traversal count overflows"))?;
    let used = outputs
        .checked_add(slots)
        .and_then(|count| count.checked_add(operands))
        .ok_or_else(|| anyhow!("finite producer constant-use count overflows"))?;
    let nodes = input_count
        .checked_add(constant_count)
        .and_then(|count| count.checked_add(operation_count))
        .ok_or_else(|| anyhow!("finite scalar node count overflows"))?;
    let node_capacity = operands
        .checked_add(1)
        .and_then(|count| count.checked_add(operation_count))
        .ok_or_else(|| anyhow!("finite scalar node capacity overflows"))?;
    let mut bytes = RequestedBytes::default();
    // Two live value/word tables, plus their two checked-lowering clones.
    for _ in 0..4 {
        bytes.vector::<(ValueId, Option<String>)>(slots)?;
        bytes.add(
            slots
                .checked_mul(max_operand)
                .ok_or_else(|| anyhow!("finite expression bytes overflow"))?,
        )?;
    }
    // Typed producer selection and the nested checked-program selection.
    for _ in 0..2 {
        bytes.vector::<ValueId>(slots)?;
        bytes.vector::<(ValueId, usize)>(operation_count)?;
        bytes.vector::<bool>(operation_count)?;
        bytes.vector::<(ValueId, usize)>(pending)?;
    }
    bytes.vector::<ValueId>(used)?;
    bytes.vector::<ValueId>(constant_count)?;
    // Checked scalar receipt and validation coexist with the source tables.
    bytes.vector::<Option<ScalarOperation>>(operation_count)?;
    bytes.vector::<ValueId>(operation_count)?;
    bytes.vector::<ValueId>(
        operands
            .checked_add(1)
            .ok_or_else(|| anyhow!("finite leaf capacity overflows"))?,
    )?;
    bytes.vector::<ScalarNode>(node_capacity)?;
    bytes.vector::<ScalarNode>(nodes)?;
    for _ in 0..2 {
        bytes.vector::<ValueId>(operands)?;
    }
    bytes.vector::<(ValueId, usize)>(nodes)?;
    bytes.vector::<bool>(nodes)?;
    bytes.vector::<NumericClass>(2)?;
    // Sorted metadata and physical-control sets plus their excluded operands.
    bytes.vector::<ValueId>(
        nodes
            .checked_mul(2)
            .ok_or_else(|| anyhow!("finite control count overflows"))?,
    )?;
    bytes.vector::<ValueId>(
        operands
            .checked_add(1)
            .ok_or_else(|| anyhow!("finite exclusion count overflows"))?,
    )?;
    for _ in 0..2 {
        bytes.vector::<ValueId>(
            operation_count
                .checked_mul(2)
                .ok_or_else(|| anyhow!("finite control count overflows"))?,
        )?;
        bytes.vector::<ValueId>(operands)?;
    }
    // V3's local represented real output facts: metadata, shapes, origin.
    bytes.vector::<(ValueId, ValueFact)>(operation_count)?;
    for _ in 0..operation_count {
        bytes.vector::<DimensionFact>(max_rank)?;
        bytes.add(17)?;
    }
    bytes.vector::<(FusionMathHelper, ScalarType)>(6)?;
    for count in [operation_count, nodes, nodes] {
        bytes.vector::<EmittedScalarImplementation>(count)?;
    }
    bytes.vector::<NumericClass>(outputs)?;
    for _ in 0..2 {
        bytes.vector::<usize>(input_count)?;
    }
    // One operation's casted operands and rendered expression can coexist
    // with the retained source body; later operations reuse this envelope.
    bytes.vector::<(ValueId, String)>(2)?;
    // Checked lowering holds borrowed raw arguments and their casted Strings
    // concurrently; the legacy pair above cannot stand in for both vectors.
    bytes.vector::<&str>(2)?;
    bytes.vector::<String>(2)?;
    bytes.add(
        max_operand
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(12))
            .ok_or_else(|| anyhow!("finite casted operand bytes overflow"))?,
    )?;
    bytes.add(max_expression)?;
    // One captured literal's formatted scalar expression is live while it is
    // written into the retained body and replaced by its short control name.
    let literal_expression = measured(|writer| {
        writer.write_str("runmat_f64_from_words(0xffffffffu, 0xffffffffu)")?;
        Ok(())
    })?;
    bytes.add(literal_expression)?;
    Ok(bytes.0)
}
