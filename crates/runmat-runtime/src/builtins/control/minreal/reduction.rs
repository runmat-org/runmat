//! Root-group matching and coefficient-preserving pole-zero deflation.

use num_complex::Complex64;

use super::{check, numerical, NAME};
use crate::builtins::control::tf_model::{polynomial_roots, TfModel};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

mod root_polish;

#[derive(Clone, Copy)]
struct RootGroup {
    first: usize,
    second: Option<usize>,
}

impl RootGroup {
    fn matches(
        self,
        other: Self,
        zeros: &[Complex64],
        poles: &[Complex64],
        tolerance: f64,
    ) -> bool {
        match (self.second, other.second) {
            (None, None) => (zeros[self.first] - poles[other.first]).norm() <= tolerance,
            (Some(z2), Some(p2)) => {
                let direct = (zeros[self.first] - poles[other.first])
                    .norm()
                    .max((zeros[z2] - poles[p2]).norm());
                let reversed = (zeros[self.first] - poles[p2])
                    .norm()
                    .max((zeros[z2] - poles[other.first]).norm());
                direct.min(reversed) <= tolerance
            }
            _ => false,
        }
    }

    fn factor(self, roots: &[Complex64], real_input: bool) -> BuiltinResult<Vec<Complex64>> {
        let first = roots[self.first];
        let factor = match self.second {
            None if real_input => vec![Complex64::new(1.0, 0.0), Complex64::new(-first.re, 0.0)],
            None => vec![Complex64::new(1.0, 0.0), -first],
            Some(other) => {
                let second = roots[other];
                let real = 0.5 * first.re + 0.5 * second.re;
                let imaginary = 0.5 * first.im.abs() + 0.5 * second.im.abs();
                let constant = real * real + imaginary * imaginary;
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(-2.0 * real, 0.0),
                    Complex64::new(constant, 0.0),
                ]
            }
        };
        if factor.iter().copied().all(finite) {
            Ok(factor)
        } else {
            Err(numerical("minreal root factor is not finite"))
        }
    }
}

fn finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn conjugacy_bound(root: Complex64, degree: usize) -> f64 {
    64.0 * f64::EPSILON * (degree as f64).max(1.0) * root.norm().max(1.0)
}

fn groups(
    roots: &[Complex64],
    real_input: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<RootGroup>> {
    let mut result = Vec::with_capacity(roots.len());
    let mut used = vec![false; roots.len()];
    for (index, root) in roots.iter().copied().enumerate() {
        check(runtime)?;
        if used[index] {
            continue;
        }
        if !finite(root) {
            return Err(numerical("minreal computed a nonfinite root"));
        }
        used[index] = true;
        if !real_input || root.im == 0.0 {
            result.push(RootGroup {
                first: index,
                second: None,
            });
            continue;
        }
        let mut mate = None;
        let mut nearest = f64::INFINITY;
        for (other, candidate) in roots.iter().copied().enumerate() {
            if other % 1024 == 0 {
                check(runtime)?;
            }
            let distance = (candidate - root.conj()).norm();
            if !used[other] && distance <= conjugacy_bound(root, roots.len()) && distance < nearest
            {
                mate = Some(other);
                nearest = distance;
            }
        }
        let mate = mate.ok_or_else(|| {
            numerical("minreal could not resolve a real polynomial's conjugate roots")
        })?;
        used[mate] = true;
        result.push(RootGroup {
            first: index,
            second: Some(mate),
        });
    }
    Ok(result)
}

fn maximum_matching(
    zero_groups: &[RootGroup],
    pole_groups: &[RootGroup],
    zeros: &[Complex64],
    poles: &[Complex64],
    tolerance: f64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<Option<usize>>> {
    let mut pole_owner: Vec<Option<usize>> = vec![None; pole_groups.len()];
    // Iterative alternating paths avoid a recursion depth proportional to
    // model order while finding a maximum-cardinality eligible matching.
    for start in 0..zero_groups.len() {
        check(runtime)?;
        let mut seen_zero = vec![false; zero_groups.len()];
        let mut seen_pole = vec![false; pole_groups.len()];
        let mut via_pole = vec![None; zero_groups.len()];
        let mut from_zero = vec![None; pole_groups.len()];
        let mut queue = vec![start];
        let mut head = 0;
        seen_zero[start] = true;
        let mut terminal = None;
        while head < queue.len() && terminal.is_none() {
            let zero_index = queue[head];
            head += 1;
            check(runtime)?;
            for pole_index in 0..pole_groups.len() {
                if pole_index % 1024 == 0 {
                    check(runtime)?;
                }
                if seen_pole[pole_index]
                    || !zero_groups[zero_index].matches(
                        pole_groups[pole_index],
                        zeros,
                        poles,
                        tolerance,
                    )
                {
                    continue;
                }
                seen_pole[pole_index] = true;
                from_zero[pole_index] = Some(zero_index);
                if let Some(next_zero) = pole_owner[pole_index] {
                    if !seen_zero[next_zero] {
                        seen_zero[next_zero] = true;
                        via_pole[next_zero] = Some(pole_index);
                        queue.push(next_zero);
                    }
                } else {
                    terminal = Some(pole_index);
                    break;
                }
            }
        }
        while let Some(pole_index) = terminal {
            let zero_index = from_zero[pole_index].expect("visited pole has predecessor");
            terminal = via_pole[zero_index];
            pole_owner[pole_index] = Some(zero_index);
        }
    }
    Ok(pole_owner)
}

fn deflate(
    coefficients: Vec<Complex64>,
    factor: &[Complex64],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<Complex64>> {
    if coefficients.len() < factor.len() {
        return Err(numerical("minreal cancellation exceeds polynomial order"));
    }
    let quotient_len = coefficients.len() - factor.len() + 1;
    let mut work = coefficients;
    let mut quotient = Vec::with_capacity(quotient_len);
    let mut scale = work
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    for index in 0..quotient_len {
        if index % 1024 == 0 {
            check(runtime)?;
        }
        let coefficient = work[index];
        quotient.push(coefficient);
        for (offset, factor_coefficient) in factor.iter().copied().enumerate() {
            let product = coefficient * factor_coefficient;
            scale = scale.max(product.norm());
            work[index + offset] -= product;
        }
    }
    let residual = work[quotient_len..]
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    let bound = 64.0 * f64::EPSILON.sqrt() * (work.len() as f64) * scale;
    if !scale.is_finite()
        || !residual.is_finite()
        || residual > bound
        || !quotient.iter().copied().all(finite)
    {
        return Err(numerical(
            "minreal root deflation is numerically unreliable",
        ));
    }
    Ok(quotient)
}

pub(super) fn reduce(
    model: &mut TfModel,
    tolerance: f64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    check(runtime)?;
    if model
        .numerator
        .iter()
        .all(|coefficient| coefficient.re == 0.0 && coefficient.im == 0.0)
    {
        // A zero transfer function has no observable poles. Preserve the
        // model's time domain and delays while avoiding companion-root work.
        model.numerator.truncate(1);
        model.denominator.truncate(1);
        model.denominator[0] = Complex64::new(1.0, 0.0);
        return Ok(());
    }
    let zeros = root_polish::polish(
        &model.numerator,
        polynomial_roots(&model.numerator, NAME)?,
        runtime,
    )?;
    check(runtime)?;
    let poles = root_polish::polish(
        &model.denominator,
        polynomial_roots(&model.denominator, NAME)?,
        runtime,
    )?;
    check(runtime)?;
    let real_input = model
        .numerator
        .iter()
        .chain(&model.denominator)
        .all(|value| value.im == 0.0);
    let zero_groups = groups(&zeros, real_input, runtime)?;
    let pole_groups = groups(&poles, real_input, runtime)?;
    let matches = maximum_matching(
        &zero_groups,
        &pole_groups,
        &zeros,
        &poles,
        tolerance,
        runtime,
    )?;
    if matches.iter().all(Option::is_none) {
        return Ok(());
    }
    // Work offstate. A failed deflation leaves the original model intact.
    let mut numerator = model.numerator.clone();
    let mut denominator = model.denominator.clone();
    for (pole_index, zero_index) in matches.into_iter().enumerate() {
        let Some(zero_index) = zero_index else {
            continue;
        };
        check(runtime)?;
        numerator = deflate(
            numerator,
            &zero_groups[zero_index].factor(&zeros, real_input)?,
            runtime,
        )?;
        denominator = deflate(
            denominator,
            &pole_groups[pole_index].factor(&poles, real_input)?,
            runtime,
        )?;
    }
    model.numerator = numerator;
    model.denominator = denominator;
    Ok(())
}
