//! Both input-reading strategies embed exactly the same producer and indexing.
use super::*;
use crate::fusion::typed_producer::TypedProducer;
use crate::graph::ValueId;
use runmat_accelerate_api::scalar_program::ScalarTransport;
use runmat_runtime::builtins::common::spec::ScalarType;
use std::collections::HashMap;

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
) -> Result<(String, String, String)> {
    render(
        descriptor,
        semantic_identity,
        program.producer_preamble(),
        program.producer_body(),
        program.producer_expression(),
        program.producer_word_expression(),
    )
}

/// Re-lower a checked recipe and compare every physical shader stage. This
/// checks internal consistency, not provenance of the carried scalar program.
pub(super) fn validate_lowered_sources(recipe: &ProviderReductionRecipe) -> Result<()> {
    let checked = recipe.checked_scalar().ok_or_else(|| {
        runmat_accelerate_api::unsupported_provider_operation(
            "reduction recipe has no checked scalar producer",
        )
    })?;
    let scalar = match recipe.descriptor().execution_precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    let words: HashMap<ValueId, String> =
        if recipe.descriptor().producer_transport == ScalarTransport::Binary32Words {
            recipe
                .scalar_inputs()
                .iter()
                .enumerate()
                .map(|(slot, &id)| (id, format!("input{slot}.data[i{slot}]")))
                .collect()
        } else {
            HashMap::new()
        };
    let expressions: HashMap<ValueId, String> = recipe
        .scalar_inputs()
        .iter()
        .enumerate()
        .map(|(slot, &id)| {
            let load = format!("input{slot}.data[i{slot}]");
            (
                id,
                if words.contains_key(&id) {
                    format!("bitcast<f32>({load})")
                } else {
                    load
                },
            )
        })
        .collect();
    let producer = TypedProducer::lower_checked_with_words(checked, scalar, expressions, words)?
        .ok_or_else(|| anyhow!("checked reduction producer has no WGSL lowering"))?;
    let preamble = producer.checked_preamble()?;
    let mut implementations = producer.implementations().to_vec();
    if recipe.descriptor().producer_transport == ScalarTransport::Binary32Words
        && recipe.descriptor().partial.precision == ProviderPrecision::F64
    {
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
        &preamble,
        &producer.body,
        producer.expression(checked.wire().root)?,
        producer.word_expression(checked.wire().root),
    )?;
    ensure!(
        recipe.sources().single == expected.0
            && recipe.sources().pass1 == expected.1
            && recipe.sources().pass2 == expected.2,
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
) -> Result<(String, String, String)> {
    let contract = &descriptor.contract;
    let inputs = &descriptor.inputs;
    let geometry = &descriptor.geometry;
    let partial = descriptor.partial;
    let config = descriptor.config;
    let word_identity = descriptor.producer_transport == ScalarTransport::Binary32Words
        && contract.identity
        && partial.precision == ProviderPrecision::F32
        && descriptor.output_precision == ProviderPrecision::F32;
    let mut state = state::source(contract, partial, geometry, word_identity)?;
    let indices = indexing::source(contract, inputs)?;
    let mut producer = indices;
    producer.push_str(&format!(
        "fn rr_value(slice: u32, r: u32) -> {} {{\n",
        if word_identity {
            "u32"
        } else {
            scalar(partial.precision)
        }
    ));
    for slot in 0..inputs.len() {
        producer.push_str(&format!("  let i{slot} = rr_index{slot}(slice, r);\n"));
    }
    producer.push_str(body);
    let word_widening = descriptor.producer_transport == ScalarTransport::Binary32Words
        && partial.precision == ProviderPrecision::F64;
    if word_identity {
        let bits = word_expression
            .ok_or_else(|| anyhow!("word identity root has no captured Single result"))?;
        producer.push_str(&format!("  return {bits};\n}}\n"));
    } else if word_widening {
        state = crate::wgsl::ieee_words::inject(&state);
        // Decode the actual represented Single result, never a potentially
        // flushing f32-to-f64 conversion at this transport boundary.
        let bits = word_expression
            .ok_or_else(|| anyhow!("word reduction root has no captured Single result"))?;
        producer.push_str(&format!("  return runmat_f32_words_to_f64({bits});\n}}\n"));
    } else {
        producer.push_str(&format!(
            "  return {}({});\n}}\n",
            scalar(partial.precision),
            expression
        ));
    }
    let preamble = if word_widening {
        format!(
            "{preamble}\n{}",
            crate::wgsl::ieee_words::binary32_widening()
        )
    } else {
        preamble.to_owned()
    };
    let context = Shader {
        contract,
        identity,
        preamble: &preamble,
        inputs,
        geometry,
        config,
        state: &state,
        producer: &producer,
        word_storage: descriptor.producer_transport == ScalarTransport::Binary32Words,
        word_identity,
    };
    Ok((
        context.emit(Pass::Single)?,
        context.emit(Pass::First)?,
        context.emit(Pass::Second)?,
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
}
impl Shader<'_> {
    fn emit(&self, pass: Pass) -> Result<String> {
        let output = if self.word_identity {
            "u32"
        } else {
            scalar(precision_of_output(self.contract)?)
        };
        let wg = self.config.workgroup_size;
        let mut source = format!(
            "// typed reduction semantic receipt {:?}\n{}",
            self.identity, self.state
        );
        let output_binding = if matches!(pass, Pass::Second) {
            source.push_str("struct RRPartials { data: array<RRPartial>, };\n@group(0) @binding(0) var<storage, read> partials: RRPartials;\n");
            1
        } else {
            source.push_str(self.preamble);
            for (slot, input) in self.inputs.iter().enumerate() {
                let storage = if self.word_storage {
                    "u32"
                } else {
                    scalar(input.precision)
                };
                source.push_str(&format!("struct RRInput{slot} {{ data: array<{storage}>, }};\n@group(0) @binding({slot}) var<storage, read> input{slot}: RRInput{slot};\n"));
            }
            source.push_str(self.producer);
            self.inputs.len()
        };
        if matches!(pass, Pass::First) {
            source.push_str(&format!("struct RRPartials {{ data: array<RRPartial>, }};\n@group(0) @binding({output_binding}) var<storage, read_write> partials: RRPartials;\n"));
        } else {
            source.push_str(&format!("struct RROutput {{ data: array<{output}>, }};\n@group(0) @binding({output_binding}) var<storage, read_write> output: RROutput;\n"));
        }
        if !self.contract.identity {
            source.push_str(&format!(
                "var<workgroup> rr_tile: array<RRPartial, {wg}>;\n"
            ));
        }
        source.push_str(&format!("@compute @workgroup_size({wg})\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\n"));
        let dispatch = match pass {
            Pass::Single => self.geometry.single,
            Pass::First => self.geometry.pass1,
            Pass::Second => self.geometry.pass2,
        };
        if dispatch.logical_groups == 0 {
            source.push_str("  return;\n}\n");
            return Ok(source);
        }
        let [x, y, _] = dispatch.workgroups;
        let groups = dispatch.logical_groups;
        // Guard padded rows/columns BEFORE multiplication/addition can wrap.
        source.push_str(&format!("  let rr_row = wid.y + wid.z * {y}u;\n  if rr_row > {}u {{ return; }}\n  let rr_base = rr_row * {x}u;\n  if wid.x >= {groups}u - rr_base {{ return; }}\n  let rr_group = rr_base + wid.x;\n", (groups - 1) / x));
        let chunks = self.geometry.chunks;
        let length = self.geometry.reduce_elements;
        match pass {
            Pass::Single => source.push_str(&format!("  let slice = rr_group;\n  let start = 0u;\n  let end = {length}u;\n")),
            Pass::First => source.push_str(&format!("  let slice = rr_group / {chunks}u;\n  let chunk = rr_group % {chunks}u;\n  let start = chunk * {}u;\n  let end = start + min({}u, {length}u - start);\n", self.config.chunk_elements, self.config.chunk_elements)),
            Pass::Second => source.push_str(&format!("  let slice = rr_group;\n  let start = 0u;\n  let end = {chunks}u;\n")),
        }
        if self.contract.identity {
            let write = match pass {
                Pass::Single => format!("output.data[slice] = {output}(rr_value(slice, 0u));"),
                Pass::First => "partials.data[rr_group] = rr_sample(rr_value(slice, 0u));".into(),
                Pass::Second => "output.data[slice] = rr_finish(partials.data[slice]);".into(),
            };
            source.push_str(&format!("  if lid.x == 0u {{ {write} }}\n}}\n"));
            return Ok(source);
        }
        let sample = if matches!(pass, Pass::Second) {
            format!("partials.data[slice * {chunks}u + i]")
        } else {
            "rr_sample(rr_value(slice, i))".into()
        };
        // Avoid start + lid overflow and the last induction-step overflow.
        source.push_str(&format!("  var acc = rr_empty();\n  if lid.x < end - start {{\n    var i = start + lid.x;\n    loop {{\n      acc = rr_combine(acc, {sample});\n      if end - i <= {wg}u {{ break; }}\n      i = i + {wg}u;\n    }}\n  }}\n  rr_tile[lid.x] = acc;\n  workgroupBarrier();\n  var offset = {}u;\n  loop {{\n    if offset == 0u {{ break; }}\n    if lid.x < offset {{ rr_tile[lid.x] = rr_combine(rr_tile[lid.x], rr_tile[lid.x + offset]); }}\n    workgroupBarrier();\n    offset = offset / 2u;\n  }}\n", wg / 2));
        let write = if matches!(pass, Pass::First) {
            "partials.data[rr_group] = rr_tile[0u];"
        } else {
            "output.data[slice] = rr_finish(rr_tile[0u]);"
        };
        source.push_str(&format!("  if lid.x == 0u {{ {write} }}\n}}\n"));
        Ok(source)
    }
}
