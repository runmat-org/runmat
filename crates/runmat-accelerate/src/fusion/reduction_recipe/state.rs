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

pub(super) fn write(
    source: &mut dyn std::fmt::Write,
    contract: &FusionReductionContract,
    partial: ReductionPartialLayout,
    geometry: &ReductionGeometry,
    word_identity: bool,
) -> Result<()> {
    if word_identity {
        ensure!(
            contract.identity
                && partial.precision == ProviderPrecision::F32
                && precision_of_output(contract)? == ProviderPrecision::F32,
            "word identity requires an unchanged Single result"
        );
        // Same 4+4+4-byte partial layout, with no floating round trip in either
        // strategy. An identity is transport, not new arithmetic permission.
        source.write_str("struct RRPartial { value: u32, count: u32, flags: u32, };\nfn rr_sample(value: u32) -> RRPartial { return RRPartial(value, 1u, 1u); }\nfn rr_finish(a: RRPartial) -> u32 { return a.value; }\n")?;
        return Ok(());
    }
    let acc = scalar(partial.precision);
    let out = scalar(precision_of_output(contract)?);
    let (nan, negative) = match partial.precision {
        ProviderPrecision::F32 => (
            "bitcast<f32>(nan_bits)",
            "(bitcast<u32>(value) & 0x80000000u) != 0u",
        ),
        ProviderPrecision::F64 => (
            // Naga's WGSL frontend cannot bitcast between a scalar f64 and
            // two u32 lanes. Extrema canonicalize NaNs; no payload is owed.
            "f64(bitcast<f32>(nan_bits))",
            // The helper guards zero before this conversion. Both signed
            // zeros are exactly representable in f32; no nonzero value or
            // accumulator is narrowed to decide a comparison.
            "(bitcast<u32>(f32(value)) & 0x80000000u) != 0u",
        ),
    };
    write!(source, "struct RRPartial {{ value: {acc}, count: u32, flags: u32, }};\nfn rr_nan() -> {acc} {{ let nan_bits: u32 = 0x7fc00000u; return {nan}; }}\nfn rr_negative_zero(value: {acc}) -> bool {{ if value != {acc}(0.0) {{ return false; }} return {negative}; }}\nfn rr_empty() -> RRPartial {{ return RRPartial({acc}(0.0), 0u, 0u); }}\n")?;
    if contract.identity {
        write!(source, "fn rr_sample(value: {acc}) -> RRPartial {{ return RRPartial(value, 1u, 1u); }}\nfn rr_combine(a: RRPartial, b: RRPartial) -> RRPartial {{ if (a.flags & 1u) != 0u {{ return a; }} return b; }}\nfn rr_finish(a: RRPartial) -> {out} {{ return {out}(a.value); }}\n")?;
        return Ok(());
    }
    // Arithmetic self-comparison can be folded under device fast-math. Use
    // the same exact IEEE-word predicate as the other reduction kernels.
    let predicate_precision = match partial.precision {
        ProviderPrecision::F32 => crate::wgsl::NumericPrecision::F32,
        ProviderPrecision::F64 => crate::wgsl::NumericPrecision::F64,
    };
    write!(source, "fn rr_is_nan(value: {acc}) -> bool {{ ")?;
    crate::wgsl::nan_predicate::write_body(source, predicate_precision)?;
    source.write_str(" }\n")?;
    let nan_flags = if contract.nan_mode == ReductionNaN::Include {
        2
    } else {
        0
    };
    write!(source, "fn rr_sample(value: {acc}) -> RRPartial {{\n  if rr_is_nan(value) {{ return RRPartial({acc}(0.0), 0u, {nan_flags}u); }}\n  return RRPartial(value, 1u, 1u);\n}}\n")?;
    write!(
        source,
        "fn rr_combine(a: RRPartial, b: RRPartial) -> RRPartial {{\n  "
    )?;
    if is_extrema(contract.operation) {
        let (comparison, tie) = match (contract.operation, contract.ties) {
            (FusionReductionOperation::Minimum, FusionReductionTies::FirstWithNegativeZero) => (
                "b.value < a.value",
                "rr_negative_zero(b.value) && !rr_negative_zero(a.value)",
            ),
            (FusionReductionOperation::Maximum, FusionReductionTies::FirstWithPositiveZero) => (
                "b.value > a.value",
                "!rr_negative_zero(b.value) && rr_negative_zero(a.value)",
            ),
            _ => return Err(anyhow!("unsupported extrema tie policy")),
        };
        write!(source, "var value = a.value;\n  if (a.flags & 1u) == 0u {{ value = b.value; }}\n  else if (b.flags & 1u) != 0u {{\n    if {comparison} || (a.value == {acc}(0.0) && b.value == {acc}(0.0) && ({tie})) {{ value = b.value; }}\n  }}")?;
    } else {
        source.write_str("let value = a.value + b.value;")?;
    }
    source.write_str("\n  return RRPartial(value, a.count + b.count, a.flags | b.flags);\n}\n")?;
    let empty = match contract.empty {
        FusionReductionEmpty::PositiveZero => "0.0",
        FusionReductionEmpty::NaN => "rr_nan()",
    };
    write!(source, "fn rr_finish(a: RRPartial) -> {out} {{\n  if (a.flags & 2u) != 0u {{ return {out}(rr_nan()); }}\n  ")?;
    match contract.operation {
        FusionReductionOperation::Mean => {
            source.write_str("let denominator = ")?;
            match contract.count {
                FusionReductionCount::NonNan => source.write_str("a.count")?,
                FusionReductionCount::SliceLength => {
                    write!(source, "{}u", geometry.reduce_elements)?
                }
                FusionReductionCount::None => return Err(anyhow!("mean lacks denominator policy")),
            }
            write!(source, ";\n  if denominator == 0u {{ return {out}({empty}); }}\n  return {out}(a.value / {acc}(denominator));")?;
        }
        _ => write!(
            source,
            "if (a.flags & 1u) == 0u {{ return {out}({empty}); }}\n  return {out}(a.value);"
        )?,
    }
    source.write_str("\n}\n")?;
    if partial.precision == ProviderPrecision::F64 {
        write!(source, "\n{}", crate::wgsl::ieee_words::helpers())?;
    }
    Ok(())
}
