//! Independent integer-oracle test for the Replicate inverse stride interval.
//! These are test arithmetic, not a second production lowering policy.

fn raw_contributors(low: i32, high: i32, base: i32, stride: i32, output_len: i32) -> Vec<i32> {
    (low..=high)
        .filter_map(|raw| {
            let numerator = raw + base;
            (numerator >= 0 && numerator % stride == 0)
                .then_some(numerator / stride)
                .filter(|&span| span < output_len)
        })
        .collect()
}

fn bounded_contributors(low: i32, high: i32, base: i32, stride: i32, output_len: i32) -> Vec<i32> {
    if high < low || output_len == 0 {
        return Vec::new();
    }
    let lo = low + base;
    let hi = high + base;
    if hi < 0 {
        return Vec::new();
    }
    let first = if lo <= 0 { 0 } else { (lo - 1) / stride + 1 };
    let last = (hi / stride).min(output_len - 1);
    if first > last {
        Vec::new()
    } else {
        (first..=last).collect()
    }
}

#[test]
fn replicate_inverse_interval_matches_raw_coordinates_and_order() {
    for len in [1, 2, 4, 8] {
        for stride in [1, 2, 4, 7] {
            for dilation in [1, 3, 5] {
                for filter_size in [1, 2, 5] {
                    for padding_left in [0, 1, 4, 8] {
                        for output_len in [1, 3, 7] {
                            let minimum = -padding_left;
                            let maximum = (output_len - 1) * stride + (filter_size - 1) * dilation
                                - padding_left;
                            for filter in 0..filter_size {
                                let base = padding_left - filter * dilation;
                                let intervals = if len == 1 {
                                    vec![(minimum, maximum)]
                                } else {
                                    vec![(minimum, maximum.min(0)), (minimum.max(len - 1), maximum)]
                                };
                                for (low, high) in intervals {
                                    assert_eq!(
                                        raw_contributors(low, high, base, stride, output_len),
                                        bounded_contributors(low, high, base, stride, output_len),
                                        "len={len} stride={stride} dilation={dilation} filter={filter} padding={padding_left} output={output_len} interval={low}..={high}",
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
