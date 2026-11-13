// Shared paired-word unsigned division/remainder. The integer arithmetic
// shader and bounded random rejection shader consume this exact source.
fn ge64(a0: u32, a1: u32, b0: u32, b1: u32) -> bool {
    return a1 > b1 || (a1 == b1 && a0 >= b0);
}
fn sub64(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let borrow = select(0u, 1u, a0 < b0);
    return vec2<u32>(a0 - b0, a1 - b1 - borrow);
}
fn increment64(lo: u32, hi: u32) -> vec2<u32> {
    let out_lo = lo + 1u;
    return vec2<u32>(out_lo, hi + select(0u, 1u, out_lo == 0u));
}
fn divmod64(n0: u32, n1: u32, d0: u32, d1: u32) -> vec4<u32> {
    var q0 = 0u;
    var q1 = 0u;
    var r0 = 0u;
    var r1 = 0u;
    var bit = 64u;
    loop {
        if (bit == 0u) { break; }
        bit = bit - 1u;
        var input_bit = 0u;
        if (bit >= 32u) { input_bit = (n1 >> (bit - 32u)) & 1u; }
        else { input_bit = (n0 >> bit) & 1u; }
        let carry = r0 >> 31u;
        r0 = (r0 << 1u) | input_bit;
        r1 = (r1 << 1u) | carry;
        if (ge64(r0, r1, d0, d1)) {
            let remainder = sub64(r0, r1, d0, d1);
            r0 = remainder.x;
            r1 = remainder.y;
            if (bit >= 32u) { q1 = q1 | (1u << (bit - 32u)); }
            else { q0 = q0 | (1u << bit); }
        }
    }
    return vec4<u32>(q0, q1, r0, r1);
}
