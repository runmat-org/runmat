//! Both input-reading strategies embed exactly the same producer and indexing.
use super::*;

#[derive(Clone, Copy)]
enum Pass {
    Single,
    First,
    Second,
}

pub(super) fn sources(
    program: &ResolvedReductionProgram<'_>,
    inputs: &[ReductionInputLayout],
    geometry: &ReductionGeometry,
    partial: ReductionPartialLayout,
    config: ReductionRecipeConfig,
) -> Result<(String, String, String)> {
    let state = state::source(program.contract(), partial, geometry)?;
    let indices = indexing::source(program, inputs)?;
    let mut producer = indices;
    producer.push_str(&format!(
        "fn rr_value(slice: u32, r: u32) -> {} {{\n",
        scalar(partial.precision)
    ));
    for slot in 0..inputs.len() {
        producer.push_str(&format!("  let i{slot} = rr_index{slot}(slice, r);\n"));
    }
    producer.push_str(program.producer_body());
    producer.push_str(&format!(
        "  return {}({});\n}}\n",
        scalar(partial.precision),
        program.producer_expression()
    ));
    let context = Shader {
        program,
        inputs,
        geometry,
        config,
        state: &state,
        producer: &producer,
    };
    Ok((
        context.emit(Pass::Single)?,
        context.emit(Pass::First)?,
        context.emit(Pass::Second)?,
    ))
}

struct Shader<'a, 'value> {
    program: &'a ResolvedReductionProgram<'value>,
    inputs: &'a [ReductionInputLayout],
    geometry: &'a ReductionGeometry,
    config: ReductionRecipeConfig,
    state: &'a str,
    producer: &'a str,
}
impl Shader<'_, '_> {
    fn emit(&self, pass: Pass) -> Result<String> {
        let output = scalar(precision_of_output(self.program.contract())?);
        let wg = self.config.workgroup_size;
        let mut source = format!(
            "// typed reduction semantic receipt {:?}\n{}",
            self.program.identity(),
            self.state
        );
        let output_binding = if matches!(pass, Pass::Second) {
            source.push_str("struct RRPartials { data: array<RRPartial>, };\n@group(0) @binding(0) var<storage, read> partials: RRPartials;\n");
            1
        } else {
            source.push_str(self.program.producer_preamble());
            for (slot, input) in self.inputs.iter().enumerate() {
                source.push_str(&format!("struct RRInput{slot} {{ data: array<{}>, }};\n@group(0) @binding({slot}) var<storage, read> input{slot}: RRInput{slot};\n", scalar(input.bound_precision)));
            }
            source.push_str(self.producer);
            self.inputs.len()
        };
        if matches!(pass, Pass::First) {
            source.push_str(&format!("struct RRPartials {{ data: array<RRPartial>, }};\n@group(0) @binding({output_binding}) var<storage, read_write> partials: RRPartials;\n"));
        } else {
            source.push_str(&format!("struct RROutput {{ data: array<{output}>, }};\n@group(0) @binding({output_binding}) var<storage, read_write> output: RROutput;\n"));
        }
        if !self.program.contract().identity {
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
        if self.program.contract().identity {
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
