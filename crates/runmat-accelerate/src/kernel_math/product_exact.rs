//! Two-factor binary32 product, using the existing POWER 48-bit RNE owner.
//! Callers include power_exact::functions() once; no floating arithmetic.

/// Structural cold-start work prior for one helper invocation. These are
/// algorithmic stages, not ISA instructions, cycles, or a GPU measurement.
/// At most four 16-bit limb products construct the 48-bit significand; exactly
/// one CLZ branch is taken and at most one round48 call is executed. Classify,
/// assemble/carry, exponent-normalize and final-pack each contribute one stage.
/// Shared measured calibration remains authoritative once observations exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProductWorkPrior {
    pub limb_products: u32,
    pub leading_zero_queries: u32,
    pub round48_calls: u32,
    pub classify_pairs: u32,
    pub assemble_carries: u32,
    pub normalize_exponents: u32,
    pub pack_results: u32,
}

impl ProductWorkPrior {
    /// Unit weight per named algorithmic step, deliberately not an estimate of
    /// equivalent native floating instructions or elapsed time.
    pub(crate) const fn algorithmic_units(self) -> u64 {
        (self.limb_products
            + self.leading_zero_queries
            + self.round48_calls
            + self.classify_pairs
            + self.assemble_carries
            + self.normalize_exponents
            + self.pack_results) as u64
    }
}

pub(crate) const fn work_prior() -> ProductWorkPrior {
    ProductWorkPrior {
        limb_products: 4,
        leading_zero_queries: 1,
        round48_calls: 1,
        classify_pairs: 1,
        assemble_carries: 1,
        normalize_exponents: 1,
        pack_results: 1,
    }
}
pub(crate) fn functions() -> &'static str {
    FUNCTIONS
}

const FUNCTIONS: &str = r#"
fn rm_exact_product_bits(a: u32, b: u32) -> u32 {
    let sign = (a ^ b) & 0x80000000u;
    let aa = a & 0x7fffffffu;
    let bb = b & 0x7fffffffu;
    if aa > 0x7f800000u || bb > 0x7f800000u { return 0x7fc00000u; }
    if aa == 0x7f800000u || bb == 0x7f800000u {
        if aa == 0u || bb == 0u { return 0x7fc00000u; }
        return sign | 0x7f800000u;
    }
    if aa == 0u || bb == 0u { return sign; }
    let ae = (aa >> 23u) & 255u;
    let be = (bb >> 23u) & 255u;
    let am = (aa & 0x7fffffu) | select(0u, 0x800000u, ae != 0u);
    let bm = (bb & 0x7fffffu) | select(0u, 0x800000u, be != 0u);
    let first = (am & 0xffffu) * (bm & 0xffffu);
    let cross = (am & 0xffffu) * (bm >> 16u) + (am >> 16u) * (bm & 0xffffu);
    let low = first + (cross << 16u);
    let high = (am >> 16u) * (bm >> 16u) + (cross >> 16u) + select(0u, 1u, low < first);
    var highest: u32;
    if high != 0u { highest = 63u - countLeadingZeros(high); }
    else { highest = 31u - countLeadingZeros(low); }
    let scale = i32(max(ae, 1u)) + i32(max(be, 1u)) - 300;
    var exponent = i32(highest) + scale;
    if exponent > 127 { return sign | 0x7f800000u; }
    if exponent < -126 {
        let shift = -149 - scale;
        if shift > 48 { return sign; }
        return sign | rm_exact_power_round48(low, high, u32(shift));
    }
    // A normal result has highest >=23. Exactly 23 requires no shift:
    // e.g. min-subnormal times a large power of two.
    var rounded = low;
    if highest > 23u { rounded = rm_exact_power_round48(low, high, highest - 23u); }
    if rounded == 0x1000000u { rounded = rounded >> 1u; exponent = exponent + 1; }
    if exponent > 127 { return sign | 0x7f800000u; }
    return sign | (u32(exponent + 127) << 23u) | (rounded & 0x7fffffu);
}
"#;
