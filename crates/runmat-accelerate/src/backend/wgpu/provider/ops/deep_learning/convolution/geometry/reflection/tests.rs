use super::*;

#[test]
fn precomputed_reflection_stride_matches_all_bounded_output_contributors() {
    for period in 2u32..=18 {
        for stride in 1u32..=17 {
            let divisor = greatest_common_divisor(stride, period);
            let step = period / divisor;
            let inverse = inverse_mod(stride / divisor, step);
            for target in 0..period {
                let computed: Vec<_> = if target % divisor != 0 {
                    Vec::new()
                } else {
                    let first =
                        (u64::from(target / divisor) * u64::from(inverse) % u64::from(step)) as u32;
                    (first..73).step_by(step as usize).collect()
                };
                let direct: Vec<_> = (0..73)
                    .filter(|&span| span * stride % period == target)
                    .collect();
                assert_eq!(
                    computed, direct,
                    "period={period} stride={stride} target={target}"
                );
            }
        }
    }
}

#[test]
fn inverse_mod_handles_near_admitted_coordinate_limit() {
    let period = (i32::MAX as u32 / 4 - 1) * 2;
    for stride in [1, 2, 3, 97, period - 1] {
        let divisor = greatest_common_divisor(stride, period);
        let step = period / divisor;
        let inverse = inverse_mod(stride / divisor, step);
        assert_eq!(
            (u64::from(stride / divisor) * u64::from(inverse)) % u64::from(step),
            u64::from(step != 1)
        );
    }
}
