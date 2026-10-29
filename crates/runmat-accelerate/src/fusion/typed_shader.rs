use anyhow::Result;
use runmat_accelerate_api::ProviderShaderRequirements;
use runmat_runtime::builtins::common::spec::ScalarType;

use super::semantics::ResolvedProgram;
use super::typed_producer::{producer_indices, scalar_name, ExpressionSlots, TypedProducer};
use super::wgsl_source::WgslShaderFragments;
use super::FusionGroupPlan;
use crate::graph::ValueId;

/// A measured producer or shader shell disagreed with the source it emitted.
/// These are admission invariants, not unavailable precision recipes.
#[derive(Debug)]
pub(crate) enum CountedGeometryViolation {
    BodyCapacity { admitted: usize, actual: usize },
    ShellCapacity { admitted: usize, actual: usize },
    SourceLength { expected: usize, actual: usize },
}

impl std::fmt::Display for CountedGeometryViolation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BodyCapacity { admitted, actual } => write!(
                formatter,
                "fusion producer exceeded pre-admitted body geometry ({actual} > {admitted} bytes)"
            ),
            Self::ShellCapacity { admitted, actual } => write!(
                formatter,
                "fusion WGSL shell exceeds pre-emission geometry ({actual} > {admitted} bytes)"
            ),
            Self::SourceLength { expected, actual } => write!(
                formatter,
                "counted fusion WGSL writer changed source length during emission ({actual} != {expected} bytes)"
            ),
        }
    }
}

impl std::error::Error for CountedGeometryViolation {}

#[derive(Clone, Copy)]
pub(crate) struct TypedShaderPreparationBound {
    pub requested_bytes: u64,
    pub body_bytes: usize,
    pub retained_bytes: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct TypedProducerPreparationBound {
    pub requested_bytes: u64,
    pub body_bytes: usize,
}

impl FusionGroupPlan {
    /// Only the selected producer, not an imaginary elementwise shader shell.
    /// The same declaration/scalar writers count every emitted expression.
    pub(crate) fn measured_typed_producer_preparation_bytes(
        &self,
        root: ValueId,
        program: &ResolvedProgram,
        scalar: ScalarType,
        input_count: usize,
    ) -> Result<TypedProducerPreparationBound> {
        let outputs = [root];
        let (body, expression) =
            super::typed_producer::body_upper_bytes(self, &outputs, program, scalar, input_count)?;
        let preamble = super::typed_producer::preamble_upper_bytes(self, program)?;
        let operand = super::typed_producer::max_operand_bytes(self, input_count)?;
        let mut rank = 2usize;
        for id in self.const_values.keys() {
            if let Ok(fact) = program.fact(*id) {
                rank = rank.max(fact.shape.rank().unwrap_or(0));
            }
        }
        for id in self
            .inputs
            .iter()
            .copied()
            .chain(self.operations.iter().filter_map(|op| match op {
                super::FusionOp::Primitive { output, .. }
                | super::FusionOp::Builtin { output, .. } => *output,
            }))
        {
            rank = rank.max(program.fact(id)?.shape.rank().unwrap_or(0));
        }
        let mut bytes =
            super::typed_producer::structural_upper_bytes(self, 1, rank, operand, expression)?;
        // Retained body/preamble and the two copied root expressions coexist
        // with lowering tables. The final shape projections are owned below.
        for count in [body, preamble, operand, operand] {
            bytes = bytes
                .checked_add(u64::try_from(count)?)
                .ok_or_else(|| anyhow::anyhow!("reduction producer storage overflow"))?;
        }
        let operations = self.operations.len();
        let arrays = [
            operations.checked_add(1).and_then(|n| {
                n.checked_mul(std::mem::size_of::<
                    runmat_accelerate_api::scalar_program::EmittedScalarImplementation,
                >())
            }),
            input_count.checked_mul(std::mem::size_of::<runmat_types::NumericClass>()),
            input_count.checked_mul(std::mem::size_of::<(ValueId, u64)>()),
            operations.checked_mul(std::mem::size_of::<Vec<usize>>()),
            operations.checked_mul(std::mem::size_of::<[Vec<usize>; 2]>()),
            operations
                .checked_mul(rank)
                .and_then(|n| n.checked_mul(3))
                .and_then(|n| n.checked_mul(std::mem::size_of::<usize>())),
        ];
        for count in arrays {
            bytes = bytes
                .checked_add(u64::try_from(count.ok_or_else(|| {
                    anyhow::anyhow!("reduction retained metadata overflows")
                })?)?)
                .ok_or_else(|| anyhow::anyhow!("reduction retained producer storage overflows"))?;
        }
        Ok(TypedProducerPreparationBound {
            requested_bytes: bytes,
            body_bytes: body,
        })
    }
}

/// Selected source and its shader feature requirement share one producer.
pub(crate) struct GeneratedElementwiseShader {
    source: String,
    requirements: ProviderShaderRequirements,
    input_indices: Vec<usize>,
    abi: runmat_accelerate_api::ProviderFusedElementwiseAbi,
    exact_product_words: bool,
    implementations: Vec<runmat_accelerate_api::scalar_program::EmittedScalarImplementation>,
}

impl GeneratedElementwiseShader {
    #[cfg(test)]
    pub(crate) fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }
    pub(crate) fn abi(&self) -> runmat_accelerate_api::ProviderFusedElementwiseAbi {
        self.abi
    }
    pub(crate) fn uses_exact_product_words(&self) -> bool {
        self.exact_product_words
    }
    pub(crate) fn implementations(
        &self,
    ) -> &[runmat_accelerate_api::scalar_program::EmittedScalarImplementation] {
        &self.implementations
    }
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn requirements(&self) -> ProviderShaderRequirements {
        self.requirements
    }

    pub(crate) fn into_parts(self) -> (String, ProviderShaderRequirements, Vec<usize>) {
        (self.source, self.requirements, self.input_indices)
    }
}

impl FusionGroupPlan {
    /// Phase-two producer preparation: resolved facts are borrowed from the
    /// phase-one receipt, while every new producer-owned source and table is
    /// charged here before `prepare_typed_shader_with_inputs` may allocate.
    pub(crate) fn measured_typed_shader_preparation_bytes(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        input_count: usize,
    ) -> Result<TypedShaderPreparationBound> {
        let buffer_name = scalar_name(buffer_type)?;
        let storage_name = if buffer_type == ScalarType::F32 {
            "u32"
        } else {
            buffer_name
        };
        let (body, max_expression) = super::typed_producer::body_upper_bytes(
            self,
            outputs,
            program,
            buffer_type,
            input_count,
        )?;
        let preamble = super::typed_producer::preamble_upper_bytes(self, program)?;
        let max_operand = super::typed_producer::max_operand_bytes(self, input_count)?;
        let mut max_rank = 2usize;
        for id in &self.inputs {
            max_rank = max_rank.max(program.fact(*id)?.shape.rank().unwrap_or(0));
        }
        for id in self.const_values.keys() {
            if let Ok(fact) = program.fact(*id) {
                max_rank = max_rank.max(fact.shape.rank().unwrap_or(0));
            }
        }
        for operation in &self.operations {
            let output = match operation {
                super::FusionOp::Primitive { output, .. }
                | super::FusionOp::Builtin { output, .. } => {
                    output.ok_or_else(|| anyhow::anyhow!("finite producer has no output"))?
                }
            };
            max_rank = max_rank.max(program.fact(output)?.shape.rank().unwrap_or(0));
        }
        let binding_count = input_count
            .checked_add(outputs.len())
            .ok_or_else(|| anyhow::anyhow!("finite fusion binding count overflows"))?;
        let mut measured_bindings = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
        write_output_bindings(&mut measured_bindings, input_count, outputs.len())?;
        let bindings = measured_bindings.bytes();
        let writes = measured_output_stores_upper(
            outputs,
            buffer_name,
            buffer_type == ScalarType::F32,
            max_operand,
        )?;
        let shell = self
            .measured_wgsl_shell_bytes_with_inputs(
                (input_count, buffer_name, storage_name),
                bindings,
                binding_count,
                body,
                writes,
                buffer_type == ScalarType::F64,
            )
            .ok_or_else(|| anyhow::anyhow!("finite fusion shader shell overflows"))?;
        let structure = super::typed_producer::structural_upper_bytes(
            self,
            outputs.len(),
            max_rank,
            max_operand,
            max_expression,
        )?;
        let requested_bytes = [preamble, shell, preamble, body, body, bindings, writes]
            .into_iter()
            .try_fold(structure, |total, bytes| {
                total.checked_add(u64::try_from(bytes).ok()?)
            })
            .ok_or_else(|| anyhow::anyhow!("finite fusion producer peak overflows"))?;
        Ok(TypedShaderPreparationBound {
            requested_bytes,
            body_bytes: body,
            retained_bytes: u64::try_from(preamble)?
                .checked_add(u64::try_from(shell)?)
                .and_then(|bytes| {
                    u64::try_from(input_count.checked_mul(std::mem::size_of::<usize>())?)
                        .ok()
                        .and_then(|input_bytes| bytes.checked_add(input_bytes))
                })
                .and_then(|bytes| {
                    u64::try_from(self.operations.len().checked_mul(2)?.checked_mul(
                        std::mem::size_of::<
                            runmat_accelerate_api::scalar_program::EmittedScalarImplementation,
                        >(),
                    )?)
                    .ok()
                    .and_then(|output_bytes| bytes.checked_add(output_bytes))
                })
                .and_then(|bytes| {
                    u64::try_from(
                        outputs
                            .len()
                            .checked_mul(std::mem::size_of::<runmat_types::NumericClass>())?,
                    )
                    .ok()
                    .and_then(|classes| bytes.checked_add(classes))
                })
                .ok_or_else(|| anyhow::anyhow!("finite fusion retained shader size overflows"))?,
        })
    }

    /// Buffer precision is a provider ABI choice; each source operation retains
    /// its own language precision. In particular, widening a Single temporary
    /// into an F64 output buffer cannot undo its intermediate rounding.
    #[cfg(test)]
    pub(crate) fn generate_typed_wgsl(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
    ) -> Result<String> {
        Ok(self
            .prepare_typed_shader(outputs, program, buffer_type)?
            .source)
    }

    #[cfg(test)]
    pub(crate) fn prepare_typed_shader(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
    ) -> Result<GeneratedElementwiseShader> {
        let controls = program.embedded_controls_sorted_for_precision(
            self,
            outputs,
            if buffer_type == ScalarType::F32 {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            },
        )?;
        let input_indices = self
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (controls.binary_search(id).is_err()).then_some(index))
            .collect::<Vec<_>>();
        self.prepare_typed_shader_with_inputs(outputs, program, buffer_type, &input_indices)
    }

    /// Reuse one already prepared physical input projection across dispatch
    /// variants. Extra data bindings retain their original order; an omitted
    /// leaf must still be a receipt-authenticated control for these outputs.
    #[cfg(test)]
    pub(crate) fn prepare_typed_shader_with_inputs(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        input_indices: &[usize],
    ) -> Result<GeneratedElementwiseShader> {
        self.prepare_typed_shader_with_inputs_bounded(
            outputs,
            program,
            buffer_type,
            input_indices,
            None,
        )
    }

    pub(crate) fn prepare_typed_shader_with_inputs_bounded(
        &self,
        outputs: &[ValueId],
        program: &ResolvedProgram,
        buffer_type: ScalarType,
        input_indices: &[usize],
        body_capacity: Option<usize>,
    ) -> Result<GeneratedElementwiseShader> {
        let buffer_name = scalar_name(buffer_type)?;
        let checked_root = self.complete_single_output_root(outputs)?;
        let roots = checked_root.as_ref().map(std::slice::from_ref);
        let word_storage = buffer_type == ScalarType::F32
            && TypedProducer::requires_binary32_word_transport(self, program, roots)?;
        let storage_name = if word_storage { "u32" } else { buffer_name };
        let mut expressions = ExpressionSlots::new(self)?;
        let mut words = ExpressionSlots::new(self)?;
        let allowed_controls = program.embedded_controls_sorted_for_precision(
            self,
            outputs,
            if buffer_type == ScalarType::F32 {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            },
        )?;
        if input_indices
            .iter()
            .any(|&index| index >= self.inputs.len())
            || input_indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(anyhow::anyhow!("invalid fusion physical input projection"));
        }
        let controls = self
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (!input_indices.contains(&index)).then_some(*id))
            .collect::<Vec<_>>();
        if controls
            .iter()
            .any(|id| allowed_controls.binary_search(id).is_err())
        {
            return Err(anyhow::anyhow!(
                "fusion physical projection omits a data input"
            ));
        }
        for (binding, &index) in input_indices.iter().enumerate() {
            let load = format!("input{binding}.data[i{binding}]");
            if word_storage {
                words.insert(self.inputs[index], load.clone())?;
            }
            expressions.insert(
                self.inputs[index],
                if word_storage {
                    format!("bitcast<f32>({load})")
                } else {
                    load
                },
            )?;
        }
        for id in controls {
            expressions.insert(id, String::new())?;
        }
        for id in allowed_controls {
            expressions.insert_if_missing(id)?;
        }
        for id in outputs {
            if self.const_values.contains_key(id) && !program.is_invocation_input(*id) {
                expressions.insert_if_missing(*id)?;
            }
        }
        // A single public result may use checked lowering only when it covers
        // every source operation. A partial slice could hide an effect or an
        // invalid operation that complete elementwise emission must report.
        let producer = TypedProducer::emit_indexed_with_words_bounded(
            self,
            program,
            buffer_type,
            expressions,
            roots,
            words,
            body_capacity,
        )?;
        if let Some(admitted) = body_capacity {
            if producer.body.len() > admitted {
                return Err(CountedGeometryViolation::BodyCapacity {
                    admitted,
                    actual: producer.body.len(),
                }
                .into());
            }
        }
        // Checked scalar programs lower only arithmetic operators and casts;
        // legacy templates conservatively retain the IEEE/hypot helpers.
        let include_ieee_helpers = producer.checked_program().is_none();
        let requirements = match buffer_type {
            ScalarType::F32 => ProviderShaderRequirements::Basic,
            ScalarType::F64 if include_ieee_helpers => ProviderShaderRequirements::F64IeeeWords,
            ScalarType::F64 => ProviderShaderRequirements::Basic,
            ScalarType::I32 | ScalarType::Bool => {
                return Err(anyhow::anyhow!("typed fusion has no floating buffer class"));
            }
        };
        let binding_count = input_indices
            .len()
            .checked_add(outputs.len())
            .ok_or_else(|| anyhow::anyhow!("fusion output binding count overflows"))?;
        let mut measured_bindings = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
        write_output_bindings(&mut measured_bindings, input_indices.len(), outputs.len())?;
        let mut bindings = String::new();
        bindings.try_reserve_exact(measured_bindings.bytes())?;
        write_output_bindings(&mut bindings, input_indices.len(), outputs.len())?;
        if bindings.len() != measured_bindings.bytes() {
            return Err(anyhow::anyhow!(
                "fusion output bindings changed during emission"
            ));
        }
        let mut measured_writes = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
        write_output_stores(
            &mut measured_writes,
            &producer,
            outputs,
            buffer_name,
            word_storage,
        )?;
        let mut writes = String::new();
        writes.try_reserve_exact(measured_writes.bytes())?;
        write_output_stores(&mut writes, &producer, outputs, buffer_name, word_storage)?;
        if writes.len() != measured_writes.bytes() {
            return Err(anyhow::anyhow!(
                "fusion output stores changed during emission"
            ));
        }
        let mut source = producer.preamble(program)?;
        let fragments = WgslShaderFragments {
            bindings: bindings.as_str(),
            body: producer.body.as_str(),
            final_writes: writes.as_str(),
        };
        let source_bytes = self
            .measured_wgsl_shader_bytes_with_inputs(
                source.len(),
                (input_indices.len(), buffer_name, storage_name),
                fragments,
                binding_count,
                include_ieee_helpers,
            )
            .ok_or_else(|| anyhow::anyhow!("fusion WGSL source byte count overflows"))?;
        let shell_upper = self
            .measured_wgsl_shell_bytes_with_inputs(
                (input_indices.len(), buffer_name, storage_name),
                bindings.len(),
                binding_count,
                producer.body.len(),
                writes.len(),
                include_ieee_helpers,
            )
            .ok_or_else(|| anyhow::anyhow!("fusion WGSL shell byte count overflows"))?;
        let shell_bytes = source_bytes.checked_sub(source.len()).ok_or(
            CountedGeometryViolation::SourceLength {
                expected: source_bytes,
                actual: source.len(),
            },
        )?;
        if shell_bytes > shell_upper {
            return Err(CountedGeometryViolation::ShellCapacity {
                admitted: shell_upper,
                actual: shell_bytes,
            }
            .into());
        }
        source.try_reserve_exact(shell_bytes)?;
        self.append_wgsl_shader_with_inputs(
            &mut source,
            (input_indices.len(), buffer_name, storage_name),
            fragments,
            binding_count,
            include_ieee_helpers,
        );
        if source.len() != source_bytes {
            return Err(CountedGeometryViolation::SourceLength {
                expected: source_bytes,
                actual: source.len(),
            }
            .into());
        }
        Ok(GeneratedElementwiseShader {
            abi: runmat_accelerate_api::ProviderFusedElementwiseAbi::BroadcastReal {
                precision: match buffer_type {
                    ScalarType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                    ScalarType::F64 => runmat_accelerate_api::ProviderPrecision::F64,
                    _ => unreachable!("floating buffer validated"),
                },
            },
            source,
            requirements,
            input_indices: input_indices.to_vec(),
            exact_product_words: producer.uses_exact_product_words(),
            implementations: producer.implementations().to_vec(),
        })
    }

    pub(crate) fn complete_single_output_root(
        &self,
        outputs: &[ValueId],
    ) -> Result<Option<ValueId>> {
        let [root] = outputs else { return Ok(None) };
        let capacity = self
            .inputs
            .len()
            .checked_add(self.const_values.len())
            .ok_or_else(|| anyhow::anyhow!("fusion root leaf count overflows"))?;
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(capacity)?;
        leaves.extend(self.inputs.iter().chain(self.const_values.keys()).copied());
        leaves.sort_unstable();
        leaves.dedup();
        let complete = match producer_indices(self, &leaves, outputs) {
            Ok(indices) => indices.len() == self.operations.len(),
            Err(error)
                if error
                    .downcast_ref::<std::collections::TryReserveError>()
                    .is_some() =>
            {
                return Err(error);
            }
            Err(_) => false,
        };
        Ok(complete.then_some(*root))
    }
}

fn write_output_bindings(
    writer: &mut impl std::fmt::Write,
    input_count: usize,
    output_count: usize,
) -> Result<()> {
    for index in 0..output_count {
        let binding = input_count
            .checked_add(index)
            .ok_or_else(|| anyhow::anyhow!("fusion output binding index overflows"))?;
        writeln!(
            writer,
            "@group(0) @binding({binding}) var<storage, read_write> output{index}: Tensor;"
        )?;
    }
    Ok(())
}

fn write_output_stores(
    writer: &mut impl std::fmt::Write,
    producer: &TypedProducer,
    outputs: &[ValueId],
    buffer_name: &str,
    word_storage: bool,
) -> Result<()> {
    write_output_stores_with(writer, outputs, |writer, id| {
        let expression = producer.expression(id)?;
        if word_storage {
            if let Some(words) = producer.word_expression(id) {
                writer.write_str(words)?;
            } else {
                write!(writer, "bitcast<u32>({expression})")?;
            }
        } else {
            write!(writer, "{buffer_name}({expression})")?;
        }
        Ok(())
    })
}

fn write_output_stores_with(
    writer: &mut impl std::fmt::Write,
    outputs: &[ValueId],
    mut write_expression: impl FnMut(&mut dyn std::fmt::Write, ValueId) -> Result<()>,
) -> Result<()> {
    for (index, id) in outputs.iter().enumerate() {
        write!(writer, "    output{index}.data[g] = ")?;
        write_expression(writer, *id)?;
        writer.write_str(";\n")?;
    }
    Ok(())
}

fn measured_output_stores_upper(
    outputs: &[ValueId],
    buffer_name: &str,
    word_storage: bool,
    max_expression_bytes: usize,
) -> Result<usize> {
    let mut writer = super::wgsl_source::CountedWgslSource::with_preamble_bytes(0);
    write_output_stores_with(&mut writer, outputs, |writer, _| {
        let expression = super::wgsl_source::MeasuredFragment(max_expression_bytes);
        if word_storage {
            write!(writer, "bitcast<u32>({expression})")?;
        } else {
            write!(writer, "{buffer_name}({expression})")?;
        }
        Ok(())
    })?;
    Ok(writer.bytes())
}
