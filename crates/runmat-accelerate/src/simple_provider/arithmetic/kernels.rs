use super::Operation;

// Match the owning arithmetic domains: real/complex mixed operations preserve
// the untouched imaginary lane; multiplying/dividing by a real scalar does not
// synthesize a complex zero lane (which changes NaN/Infinity and signed zero).
macro_rules! native {
    ($name:ident, $float:ty, $complex:ty, $divide:ident) => {
        pub(super) fn $name(
            (ar, ai): ($float, $float),
            (br, bi): ($float, $float),
            ac: bool,
            bc: bool,
            op: Operation,
        ) -> ($float, $float) {
            match op {
                Operation::Add => (
                    ar + br,
                    match (ac, bc) {
                        (true, true) => ai + bi,
                        (true, false) => ai,
                        (false, true) => bi,
                        (false, false) => 0.0,
                    },
                ),
                Operation::Sub => (
                    ar - br,
                    match (ac, bc) {
                        (true, true) => ai - bi,
                        (true, false) => ai,
                        (false, true) => -bi,
                        (false, false) => 0.0,
                    },
                ),
                Operation::Mul => match (ac, bc) {
                    (true, true) => (ar * br - ai * bi, ar * bi + ai * br),
                    (true, false) => (ar * br, ai * br),
                    (false, true) => (ar * br, ar * bi),
                    (false, false) => (ar * br, 0.0),
                },
                Operation::Div => {
                    if bc {
                        if [ar, ai, br, bi].into_iter().all(<$float>::is_finite)
                            && (br != 0.0 || bi != 0.0)
                        {
                            runmat_value::$divide((ar, ai), (br, bi))
                        } else {
                            let quotient = <$complex>::new(ar, ai) / <$complex>::new(br, bi);
                            (quotient.re, quotient.im)
                        }
                    } else if ac {
                        (ar / br, ai / br)
                    } else {
                        (ar / br, 0.0)
                    }
                }
            }
        }
    };
}
native!(single, f32, num_complex::Complex32, divide_complex_f32);
native!(double, f64, num_complex::Complex64, divide_complex_f64);
