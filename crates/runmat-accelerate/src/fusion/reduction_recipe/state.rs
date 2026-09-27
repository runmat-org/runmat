//! The same accumulator/state functions are embedded in all three shaders.
use super::*;
use runmat_runtime::builtins::common::spec::{
    FusionReductionCount, FusionReductionEmpty, FusionReductionTies, ReductionNaN,
};

pub(super) fn layout(precision: ProviderPrecision) -> ReductionPartialLayout {
    let width = bytes(precision) as u32;
    ReductionPartialLayout {
        precision,
        alignment: width,
        stride: width + 8,
        count_offset: width,
        flags_offset: width + 4,
    }
}

pub(super) fn source(
    contract: &FusionReductionContract,
    partial: ReductionPartialLayout,
    geometry: &ReductionGeometry,
) -> Result<String> {
    let acc = scalar(partial.precision);
    let out = scalar(precision_of_output(contract)?);
    let (nan, negative) = match partial.precision {
        ProviderPrecision::F32 => (
            "bitcast<f32>(0x7fc00000u)",
            "(bitcast<u32>(value) & 0x80000000u) != 0u",
        ),
        ProviderPrecision::F64 => (
            "bitcast<f64>(vec2<u32>(0u, 0x7ff80000u))",
            "(bitcast<vec2<u32>>(value).y & 0x80000000u) != 0u",
        ),
    };
    let mut source = format!("struct RRPartial {{ value: {acc}, count: u32, flags: u32, }};\nfn rr_nan() -> {acc} {{ return {nan}; }}\nfn rr_negative(value: {acc}) -> bool {{ return {negative}; }}\nfn rr_empty() -> RRPartial {{ return RRPartial({acc}(0.0), 0u, 0u); }}\n");
    if contract.identity {
        source.push_str(&format!("fn rr_sample(value: {acc}) -> RRPartial {{ return RRPartial(value, 1u, 1u); }}\nfn rr_combine(a: RRPartial, b: RRPartial) -> RRPartial {{ if (a.flags & 1u) != 0u {{ return a; }} return b; }}\nfn rr_finish(a: RRPartial) -> {out} {{ return {out}(a.value); }}\n"));
        return Ok(source);
    }
    let nan_flags = if contract.nan_mode == ReductionNaN::Include {
        2
    } else {
        0
    };
    source.push_str(&format!("fn rr_sample(value: {acc}) -> RRPartial {{\n  if value != value {{ return RRPartial({acc}(0.0), 0u, {nan_flags}u); }}\n  return RRPartial(value, 1u, 1u);\n}}\n"));
    let combine = if is_extrema(contract.operation) {
        let (comparison, tie) = match (contract.operation, contract.ties) {
            (FusionReductionOperation::Minimum, FusionReductionTies::FirstWithNegativeZero) => (
                "b.value < a.value",
                "rr_negative(b.value) && !rr_negative(a.value)",
            ),
            (FusionReductionOperation::Maximum, FusionReductionTies::FirstWithPositiveZero) => (
                "b.value > a.value",
                "!rr_negative(b.value) && rr_negative(a.value)",
            ),
            _ => return Err(anyhow!("unsupported extrema tie policy")),
        };
        format!("var value = a.value;\n  if (a.flags & 1u) == 0u {{ value = b.value; }}\n  else if (b.flags & 1u) != 0u {{\n    if {comparison} || (a.value == {acc}(0.0) && b.value == {acc}(0.0) && ({tie})) {{ value = b.value; }}\n  }}")
    } else {
        "let value = a.value + b.value;".into()
    };
    source.push_str(&format!("fn rr_combine(a: RRPartial, b: RRPartial) -> RRPartial {{\n  {combine}\n  return RRPartial(value, a.count + b.count, a.flags | b.flags);\n}}\n"));
    let empty = match contract.empty {
        FusionReductionEmpty::PositiveZero => format!("{out}(0.0)"),
        FusionReductionEmpty::NaN => format!("{out}(rr_nan())"),
    };
    let result = match contract.operation {
        FusionReductionOperation::Mean => {
            let denominator = match contract.count {
                FusionReductionCount::NonNan => "a.count".to_owned(),
                FusionReductionCount::SliceLength => format!("{}u", geometry.reduce_elements),
                FusionReductionCount::None => return Err(anyhow!("mean lacks denominator policy")),
            };
            format!("let denominator = {denominator};\n  if denominator == 0u {{ return {empty}; }}\n  return {out}(a.value / {acc}(denominator));")
        }
        _ => format!("if (a.flags & 1u) == 0u {{ return {empty}; }}\n  return {out}(a.value);"),
    };
    source.push_str(&format!("fn rr_finish(a: RRPartial) -> {out} {{\n  if (a.flags & 2u) != 0u {{ return {out}(rr_nan()); }}\n  {result}\n}}\n"));
    Ok(source)
}
