//! Operation counts in words for admission messages: "61.3 billion", "12,345".
use runmat_simulation_usage::OperationCount;

/// Three significant figures with a word scale above a million, rounded up: an estimate never
/// reads smaller than the count it states.
pub(super) fn readable_estimate(count: OperationCount) -> String {
    readable(count.operations(), true)
}

/// Like [`readable_estimate`], rounded down: a remaining allowance never reads larger.
pub(super) fn readable_available(count: OperationCount) -> String {
    readable(count.operations(), false)
}

fn readable(count: u64, round_up: bool) -> String {
    const SCALES: [(u64, &str); 4] = [
        (1_000_000_000_000_000, "quadrillion"),
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
    ];
    let Some(&(unit, word)) = SCALES.iter().find(|(unit, _)| count >= *unit) else {
        return grouped(count);
    };
    let whole = count / unit;
    let decimals: u32 = match whole {
        0..=9 => 2,
        10..=99 => 1,
        _ => 0,
    };
    let scale = 10_u128.pow(decimals);
    let numerator = u128::from(count) * scale;
    let unit = u128::from(unit);
    let scaled = if round_up {
        numerator.div_ceil(unit)
    } else {
        numerator / unit
    };
    let (integer, fraction) = (scaled / scale, scaled % scale);
    if decimals == 0 {
        format!("{integer} {word}")
    } else {
        format!(
            "{integer}.{fraction:0width$} {word}",
            width = decimals as usize
        )
    }
}

fn grouped(count: u64) -> String {
    let digits = count.to_string();
    let mut text = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            text.push(',');
        }
        text.push(digit);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_read_in_words_rounded_toward_the_safe_side() {
        let ops = OperationCount::from_operations;
        assert_eq!(readable_estimate(ops(0)), "0");
        assert_eq!(readable_estimate(ops(12_345)), "12,345");
        assert_eq!(readable_estimate(ops(999_999)), "999,999");
        assert_eq!(readable_estimate(ops(50_000_657_332)), "50.1 billion");
        assert_eq!(readable_available(ops(50_000_657_332)), "50.0 billion");
        assert_eq!(readable_available(ops(50_000_000_000)), "50.0 billion");
        assert_eq!(readable_estimate(ops(1_234_000_000)), "1.24 billion");
        assert_eq!(readable_available(ops(1_234_000_000)), "1.23 billion");
        assert_eq!(readable_estimate(ops(340_000_000)), "340 million");
        assert_eq!(readable_available(ops(2_500_000_000_000)), "2.50 trillion");
    }
}
