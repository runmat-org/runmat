pub fn displacement_increment_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displacement_increment_norm_matches_expected_delta() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 0.0, 3.0];
        let norm = displacement_increment_norm(&a, &b);
        assert!((norm - 2.0).abs() < 1.0e-12);
    }
}
