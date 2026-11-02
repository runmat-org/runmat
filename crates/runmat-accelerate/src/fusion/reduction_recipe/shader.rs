//! Both input-reading strategies embed exactly the same producer and indexing.
use super::*;
use crate::fusion::typed_producer::{
    checked_format, CheckedProducerResources, ExpressionSlots, TypedProducer,
};
use crate::wgsl::host_buffers::ClaimedString;
use runmat_accelerate_api::scalar_program::ScalarTransport;
use runmat_execution::resource::ProviderHostMetadataAdmission;
use runmat_runtime::builtins::common::spec::ScalarType;

#[derive(Clone, Copy)]
enum Pass {
    Single,
    First,
    Second,
}

pub(super) fn sources(
    program: &ResolvedReductionProgram<'_>,
    descriptor: &ReductionRecipeDescriptor,
    semantic_identity: Digest,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<(ClaimedString, ClaimedString, ClaimedString)> {
    render(
        descriptor,
        semantic_identity,
        program.producer_preamble(),
        program.producer_body(),
        program.producer_expression(),
        program.producer_word_expression(),
        metadata,
    )
}

/// Re-lower a checked recipe and compare every physical shader stage. This
/// checks internal consistency, not provenance of the carried scalar program.
pub(super) fn validate_lowered_sources(
    recipe: &ProviderReductionRecipe,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<()> {
    let checked = recipe.checked_scalar().ok_or_else(|| {
        runmat_accelerate_api::unsupported_provider_operation(
            "reduction recipe has no checked scalar producer",
        )
    })?;
    let scalar = match recipe.descriptor().execution_precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    let widening = recipe.descriptor().producer_transport == ScalarTransport::Binary32Words
        && recipe.descriptor().partial.precision == ProviderPrecision::F64;
    let resources = CheckedProducerResources::for_program(
        checked,
        scalar,
        recipe.scalar_inputs().len(),
        widening,
    )?;
    // This request-local aggregate precedes every checked lowering block and
    // drops only after its producer/slot/clone backing has been destroyed.
    let _checked_claim = crate::wgsl::host_buffers::claim_bytes(metadata, resources.scratch_bytes)?;
    let mut ids = runmat_value::exact_capacity_vec(checked.wire().nodes.len())?;
    ids.extend(checked.wire().nodes.iter().map(|node| node.id()));
    ids.sort_unstable();
    let mut expressions = ExpressionSlots::from_sorted_ids(&ids);
    let mut words = ExpressionSlots::from_sorted_ids(&ids);
    for (slot, &id) in recipe.scalar_inputs().iter().enumerate() {
        let load = checked_format(format_args!("input{slot}.data[i{slot}]"))?;
        if recipe.descriptor().producer_transport == ScalarTransport::Binary32Words {
            words.insert(id, load.clone())?;
            expressions.insert(id, checked_format(format_args!("bitcast<f32>({load})"))?)?;
        } else {
            expressions.insert(id, load)?;
        }
    }
    let producer = TypedProducer::lower_checked_indexed_with_words(
        checked,
        scalar,
        expressions,
        words,
        Some(resources.body_bytes),
    )?
    .ok_or_else(|| anyhow!("checked reduction producer has no WGSL lowering"))?;
    let preamble = writer::render(metadata, |out| producer.write_checked_preamble(out))?;
    let capacity = producer
        .implementations()
        .len()
        .checked_add(usize::from(widening))
        .ok_or_else(|| anyhow!("checked implementation count overflow"))?;
    let mut implementations = runmat_value::exact_capacity_vec(capacity)?;
    implementations.extend_from_slice(producer.implementations());
    if widening {
        implementations.push(crate::wgsl::ieee_words::binary32_widening_implementation(
            checked.wire().root,
        ));
    }
    ensure!(
        implementations == recipe.descriptor().producer_implementations,
        "checked reduction implementations differ from actual lowering"
    );
    let expected = render(
        recipe.descriptor(),
        recipe.semantic_identity(),
        preamble.as_str(),
        &producer.body,
        producer.expression(checked.wire().root)?,
        producer.word_expression(checked.wire().root),
        metadata,
    )?;
    ensure!(
        recipe.sources().single == expected.0.as_str()
            && recipe.sources().pass1 == expected.1.as_str()
            && recipe.sources().pass2 == expected.2.as_str(),
        "checked reduction source differs from typed semantics"
    );
    Ok(())
}

fn render(
    descriptor: &ReductionRecipeDescriptor,
    identity: Digest,
    preamble: &str,
    body: &str,
    expression: &str,
    word_expression: Option<&str>,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<(ClaimedString, ClaimedString, ClaimedString)> {
    let contract = &descriptor.contract;
    let inputs = &descriptor.inputs;
    let geometry = &descriptor.geometry;
    let partial = descriptor.partial;
    let config = descriptor.config;
    let word_identity = descriptor.producer_transport == ScalarTransport::Binary32Words
        && contract.identity
        && partial.precision == ProviderPrecision::F32
        && descriptor.output_precision == ProviderPrecision::F32;
    let word_widening = descriptor.producer_transport == ScalarTransport::Binary32Words
        && partial.precision == ProviderPrecision::F64;
    let state = writer::render(metadata, |out| {
        state::write(out, contract, partial, geometry, word_identity)?;
        if word_widening && contract.identity {
            write!(out, "\n{}", crate::wgsl::ieee_words::helpers())?;
        }
        Ok(())
    })?;
    let producer = writer::render(metadata, |out| {
        indexing::write_source(out, contract, inputs)?;
        writeln!(
            out,
            "fn rr_value(slice: u32, r: u32) -> {} {{",
            if word_identity {
                "u32"
            } else {
                scalar(partial.precision)
            }
        )?;
        for slot in 0..inputs.len() {
            writeln!(out, "  let i{slot} = rr_index{slot}(slice, r);")?;
        }
        out.write_str(body)?;
        if word_identity {
            let bits = word_expression
                .ok_or_else(|| anyhow!("word identity root has no captured Single result"))?;
            write!(out, "  return {bits};\n}}\n")?;
        } else if word_widening {
            let bits = word_expression
                .ok_or_else(|| anyhow!("word reduction root has no captured Single result"))?;
            write!(out, "  return runmat_f32_words_to_f64({bits});\n}}\n")?;
        } else {
            write!(
                out,
                "  return {}({expression});\n}}\n",
                scalar(partial.precision)
            )?;
        }
        Ok(())
    })?;
    let context = Shader {
        contract,
        identity,
        preamble,
        inputs,
        geometry,
        config,
        state: state.as_str(),
        producer: producer.as_str(),
        word_storage: descriptor.producer_transport == ScalarTransport::Binary32Words,
        word_identity,
        word_widening,
    };
    Ok((
        writer::render(metadata, |out| context.write(Pass::Single, out))?,
        writer::render(metadata, |out| context.write(Pass::First, out))?,
        writer::render(metadata, |out| context.write(Pass::Second, out))?,
    ))
}

struct Shader<'a> {
    contract: &'a FusionReductionContract,
    identity: Digest,
    preamble: &'a str,
    inputs: &'a [ReductionInputBinding],
    geometry: &'a ReductionGeometry,
    config: ReductionRecipeConfig,
    state: &'a str,
    producer: &'a str,
    word_storage: bool,
    word_identity: bool,
    word_widening: bool,
}
impl Shader<'_> {
    fn write(&self, pass: Pass, source: &mut dyn std::fmt::Write) -> Result<()> {
        let output = if self.word_identity {
            "u32"
        } else {
            scalar(precision_of_output(self.contract)?)
        };
        let wg = self.config.workgroup_size;
        write!(
            source,
            "// typed reduction semantic receipt {:?}\n{}",
            self.identity, self.state
        )?;
        let output_binding = if matches!(pass, Pass::Second) {
            source.write_str("struct RRPartials { data: array<RRPartial>, };\n@group(0) @binding(0) var<storage, read> partials: RRPartials;\n")?;
            1
        } else {
            source.write_str(self.preamble)?;
            if self.word_widening {
                source.write_str("\n")?;
                crate::wgsl::ieee_words::write_binary32_widening(source)?;
            }
            for (slot, input) in self.inputs.iter().enumerate() {
                let storage = if self.word_storage {
                    "u32"
                } else {
                    scalar(input.precision)
                };
                write!(source, "struct RRInput{slot} {{ data: array<{storage}>, }};\n@group(0) @binding({slot}) var<storage, read> input{slot}: RRInput{slot};\n")?;
            }
            source.write_str(self.producer)?;
            self.inputs.len()
        };
        if matches!(pass, Pass::First) {
            write!(source, "struct RRPartials {{ data: array<RRPartial>, }};\n@group(0) @binding({output_binding}) var<storage, read_write> partials: RRPartials;\n")?;
        } else {
            write!(source, "struct RROutput {{ data: array<{output}>, }};\n@group(0) @binding({output_binding}) var<storage, read_write> output: RROutput;\n")?;
        }
        if !self.contract.identity {
            writeln!(source, "var<workgroup> rr_tile: array<RRPartial, {wg}>;")?;
        }
        write!(source, "@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n")?;
        let dispatch = match pass {
            Pass::Single => self.geometry.single,
            Pass::First => self.geometry.pass1,
            Pass::Second => self.geometry.pass2,
        };
        if dispatch.logical_groups == 0 {
            source.write_str("  return;\n}\n")?;
            return Ok(());
        }
        let [x, y, _] = dispatch.workgroups;
        let groups = dispatch.logical_groups;
        // Guard padded rows/columns BEFORE multiplication/addition can wrap.
        write!(source, "  let rr_row = wid.y + wid.z * {y}u;\n  if rr_row > {}u {{ return; }}\n  let rr_base = rr_row * {x}u;\n  if wid.x >= {groups}u - rr_base {{ return; }}\n  let rr_group = rr_base + wid.x;\n", (groups - 1) / x)?;
        let chunks = self.geometry.chunks;
        let length = self.geometry.reduce_elements;
        match pass {
            Pass::Single => write!(source, "  let slice = rr_group;\n  let start = 0u;\n  let end = {length}u;\n")?,
            Pass::First => write!(source, "  let slice = rr_group / {chunks}u;\n  let chunk = rr_group % {chunks}u;\n  let start = chunk * {}u;\n  let end = start + min({}u, {length}u - start);\n", self.config.chunk_elements, self.config.chunk_elements)?,
            Pass::Second => write!(source, "  let slice = rr_group;\n  let start = 0u;\n  let end = {chunks}u;\n")?,
        }
        if self.contract.identity {
            source.write_str("  if lid.x == 0u { ")?;
            match pass {
                Pass::Single => write!(
                    source,
                    "output.data[slice] = {output}(rr_value(slice, 0u));"
                )?,
                Pass::First => {
                    source.write_str("partials.data[rr_group] = rr_sample(rr_value(slice, 0u));")?
                }
                Pass::Second => {
                    source.write_str("output.data[slice] = rr_finish(partials.data[slice]);")?
                }
            }
            source.write_str(" }\n}\n")?;
            return Ok(());
        }
        // Avoid start + lid overflow and the last induction-step overflow.
        source.write_str("  var acc = rr_empty();\n  if lid.x < end - start {\n    var i = start + lid.x;\n    loop {\n      acc = rr_combine(acc, ")?;
        if matches!(pass, Pass::Second) {
            write!(source, "partials.data[slice * {chunks}u + i]")?;
        } else {
            source.write_str("rr_sample(rr_value(slice, i))")?;
        }
        write!(source, ");\n      if end - i <= {wg}u {{ break; }}\n      i = i + {wg}u;\n    }}\n  }}\n  rr_tile[lid.x] = acc;\n  workgroupBarrier();\n  var offset = {}u;\n  loop {{\n    if offset == 0u {{ break; }}\n    if lid.x < offset {{ rr_tile[lid.x] = rr_combine(rr_tile[lid.x], rr_tile[lid.x + offset]); }}\n    workgroupBarrier();\n    offset = offset / 2u;\n  }}\n", wg / 2)?;
        let write = if matches!(pass, Pass::First) {
            "partials.data[rr_group] = rr_tile[0u];"
        } else {
            "output.data[slice] = rr_finish(rr_tile[0u]);"
        };
        write!(source, "  if lid.x == 0u {{ {write} }}\n}}\n")?;
        Ok(())
    }
}
