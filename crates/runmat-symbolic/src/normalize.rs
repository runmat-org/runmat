//! Staged normalization pipeline for symbolic expressions
//!
//! Normalization transforms expressions into a canonical form for:
//! - Efficient equality comparison
//! - Better simplification opportunities
//! - Predictable output format

use crate::coeff::Coefficient;
use crate::expr::{SymExpr, SymExprKind};
use serde::{Deserialize, Serialize};

/// A normalization pass that can be applied to an expression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPass {
    /// Flatten nested Add/Mul operations
    Flatten,
    /// Sort terms/factors in canonical order
    Sort,
    /// Merge numeric constants
    MergeConstants,
    /// Simplify powers (x^0 = 1, x^1 = x)
    SimplifyPowers,
    /// Collect like terms (x + x = 2*x)
    CollectLikeTerms,
    /// Remove identity elements (x + 0 = x, x * 1 = x)
    RemoveIdentity,
    /// Apply double negation rule
    SimplifyNeg,
    /// Expand products of sums
    Expand,
}

/// A proof record of normalization steps
#[derive(Debug, Clone, Default)]
pub struct NormProof {
    pub steps: Vec<NormStep>,
    pub initial_size: usize,
    pub final_size: usize,
}

/// A single step in the normalization proof
#[derive(Debug, Clone)]
pub struct NormStep {
    pub pass: NormPass,
    pub before_size: usize,
    pub after_size: usize,
    pub changed: bool,
}

/// Staged normalization engine
#[derive(Debug, Clone)]
pub struct StagedNormalizer {
    passes: Vec<NormPass>,
    record_proof: bool,
}

impl StagedNormalizer {
    /// Create a new normalizer with the given passes
    pub fn new(passes: Vec<NormPass>) -> Self {
        StagedNormalizer {
            passes,
            record_proof: false,
        }
    }

    /// Create a default normalization pipeline
    pub fn default_pipeline() -> Self {
        Self::new(vec![
            NormPass::SimplifyNeg,
            NormPass::Flatten,
            NormPass::MergeConstants,
            NormPass::RemoveIdentity,
            NormPass::SimplifyPowers,
            NormPass::CollectLikeTerms,
            NormPass::Sort,
        ])
    }

    /// Create a minimal pipeline (just flatten and sort)
    pub fn minimal() -> Self {
        Self::new(vec![NormPass::Flatten, NormPass::Sort])
    }

    /// Create an aggressive pipeline including expansion
    pub fn aggressive() -> Self {
        Self::new(vec![
            NormPass::SimplifyNeg,
            NormPass::Flatten,
            NormPass::MergeConstants,
            NormPass::RemoveIdentity,
            NormPass::SimplifyPowers,
            NormPass::Expand,
            NormPass::Flatten,
            NormPass::CollectLikeTerms,
            NormPass::MergeConstants,
            NormPass::Sort,
        ])
    }

    /// Enable proof recording
    pub fn with_proof_recording(mut self) -> Self {
        self.record_proof = true;
        self
    }

    /// Normalize an expression using the configured passes
    pub fn normalize(&self, expr: SymExpr) -> (SymExpr, Option<NormProof>) {
        let mut current = expr;
        let mut proof = if self.record_proof {
            Some(NormProof {
                initial_size: current.node_count(),
                ..Default::default()
            })
        } else {
            None
        };

        for pass in &self.passes {
            let before_size = current.node_count();
            let next = apply_pass(&current, pass);
            let after_size = next.node_count();
            let changed = current != next;

            if let Some(ref mut p) = proof {
                p.steps.push(NormStep {
                    pass: pass.clone(),
                    before_size,
                    after_size,
                    changed,
                });
            }

            current = next;
        }

        if let Some(ref mut p) = proof {
            p.final_size = current.node_count();
        }

        (current, proof)
    }
}

impl Default for StagedNormalizer {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

/// Apply a single normalization pass
fn apply_pass(expr: &SymExpr, pass: &NormPass) -> SymExpr {
    match pass {
        NormPass::Flatten => flatten(expr),
        NormPass::Sort => sort_terms(expr),
        NormPass::MergeConstants => merge_constants(expr),
        NormPass::SimplifyPowers => simplify_powers(expr),
        NormPass::CollectLikeTerms => collect_like_terms(expr),
        NormPass::RemoveIdentity => remove_identity(expr),
        NormPass::SimplifyNeg => simplify_neg(expr),
        NormPass::Expand => expand(expr),
    }
}

/// Flatten nested Add and Mul operations
fn flatten(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            let mut new_terms = Vec::new();
            for t in terms {
                let flattened = flatten(t);
                if let SymExprKind::Add(inner) = flattened.kind.as_ref() {
                    new_terms.extend(inner.iter().cloned());
                } else {
                    new_terms.push(flattened);
                }
            }
            SymExpr::add(new_terms)
        }
        SymExprKind::Mul(factors) => {
            let mut new_factors = Vec::new();
            for f in factors {
                let flattened = flatten(f);
                if let SymExprKind::Mul(inner) = flattened.kind.as_ref() {
                    new_factors.extend(inner.iter().cloned());
                } else {
                    new_factors.push(flattened);
                }
            }
            SymExpr::mul(new_factors)
        }
        SymExprKind::Pow(base, exp) => SymExpr::pow(flatten(base), flatten(exp)),
        SymExprKind::Neg(inner) => SymExpr::neg(flatten(inner)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(flatten).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Sort terms in canonical order
fn sort_terms(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            let mut sorted: Vec<_> = terms.iter().map(sort_terms).collect();
            sorted.sort_by(compare_exprs);
            SymExpr::add(sorted)
        }
        SymExprKind::Mul(factors) => {
            let mut sorted: Vec<_> = factors.iter().map(sort_terms).collect();
            sorted.sort_by(compare_exprs);
            SymExpr::mul(sorted)
        }
        SymExprKind::Pow(base, exp) => SymExpr::pow(sort_terms(base), sort_terms(exp)),
        SymExprKind::Neg(inner) => SymExpr::neg(sort_terms(inner)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(sort_terms).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Compare expressions for sorting
fn compare_exprs(a: &SymExpr, b: &SymExpr) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;

    // Order: numbers < variables < powers < products < sums < functions
    let type_order = |e: &SymExpr| -> u8 {
        match e.kind.as_ref() {
            SymExprKind::Num(_) => 0,
            SymExprKind::Var(_) => 1,
            SymExprKind::Pow(_, _) => 2,
            SymExprKind::Mul(_) => 3,
            SymExprKind::Add(_) => 4,
            SymExprKind::Neg(_) => 5,
            SymExprKind::Func(_, _) => 6,
        }
    };

    let ord_a = type_order(a);
    let ord_b = type_order(b);

    if ord_a != ord_b {
        return ord_a.cmp(&ord_b);
    }

    match (a.kind.as_ref(), b.kind.as_ref()) {
        (SymExprKind::Num(ca), SymExprKind::Num(cb)) => ca.cmp(cb),
        (SymExprKind::Var(sa), SymExprKind::Var(sb)) => sa.name.cmp(&sb.name),
        (SymExprKind::Func(na, _), SymExprKind::Func(nb, _)) => na.cmp(nb),
        (SymExprKind::Pow(ba, ea), SymExprKind::Pow(bb, eb)) => {
            compare_exprs(ba, bb).then_with(|| compare_exprs(ea, eb))
        }
        _ => Equal,
    }
}

/// Merge numeric constants in Add and Mul
fn merge_constants(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            let mut sum = Coefficient::int(0);
            let mut non_const = Vec::new();

            for t in terms {
                let merged = merge_constants(t);
                if let SymExprKind::Num(c) = merged.kind.as_ref() {
                    sum = sum + c.clone();
                } else {
                    non_const.push(merged);
                }
            }

            if !sum.is_zero() {
                non_const.insert(0, SymExpr::num(sum));
            }
            SymExpr::add(non_const)
        }
        SymExprKind::Mul(factors) => {
            let mut prod = Coefficient::int(1);
            let mut non_const = Vec::new();

            for f in factors {
                let merged = merge_constants(f);
                if let SymExprKind::Num(c) = merged.kind.as_ref() {
                    prod = prod * c.clone();
                } else {
                    non_const.push(merged);
                }
            }

            // If product is zero, everything is zero
            if prod.is_zero() {
                return SymExpr::int(0);
            }

            if !prod.is_one() {
                non_const.insert(0, SymExpr::num(prod));
            }
            SymExpr::mul(non_const)
        }
        SymExprKind::Pow(base, exp) => {
            let new_base = merge_constants(base);
            let new_exp = merge_constants(exp);

            // Constant^Constant = evaluated constant
            if let (Some(bc), Some(ec)) = (new_base.as_coeff(), new_exp.as_coeff()) {
                return SymExpr::num(bc.pow(ec));
            }

            SymExpr::pow(new_base, new_exp)
        }
        SymExprKind::Neg(inner) => SymExpr::neg(merge_constants(inner)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(merge_constants).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Simplify powers: x^0 = 1, x^1 = x
fn simplify_powers(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Pow(base, exp) => {
            let new_base = simplify_powers(base);
            let new_exp = simplify_powers(exp);

            // x^0 = 1
            if new_exp.is_zero() {
                return SymExpr::int(1);
            }

            // x^1 = x
            if new_exp.is_one() {
                return new_base;
            }

            // 0^n = 0 (for n > 0)
            if new_base.is_zero() {
                if let Some(c) = new_exp.as_coeff() {
                    if !c.is_negative() {
                        return SymExpr::int(0);
                    }
                }
            }

            // 1^x = 1
            if new_base.is_one() {
                return SymExpr::int(1);
            }

            SymExpr::pow(new_base, new_exp)
        }
        SymExprKind::Add(terms) => {
            let new_terms: Vec<_> = terms.iter().map(simplify_powers).collect();
            SymExpr::add(new_terms)
        }
        SymExprKind::Mul(factors) => {
            let new_factors: Vec<_> = factors.iter().map(simplify_powers).collect();
            SymExpr::mul(new_factors)
        }
        SymExprKind::Neg(inner) => SymExpr::neg(simplify_powers(inner)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(simplify_powers).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Collect like terms: x + x = 2*x
fn collect_like_terms(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            // Group terms by their non-coefficient part
            let mut groups: std::collections::HashMap<SymExpr, Coefficient> =
                std::collections::HashMap::new();
            let mut const_sum = Coefficient::int(0);

            for t in terms {
                let collected = collect_like_terms(t);

                if let Some(c) = collected.as_coeff() {
                    const_sum = const_sum + c.clone();
                    continue;
                }

                // Extract coefficient and base
                let (coeff, base) = extract_coeff_and_base(&collected);
                groups
                    .entry(base)
                    .and_modify(|c| *c = c.clone() + coeff.clone())
                    .or_insert(coeff);
            }

            let mut result = Vec::new();

            if !const_sum.is_zero() {
                result.push(SymExpr::num(const_sum));
            }

            for (base, coeff) in groups {
                if coeff.is_zero() {
                    continue;
                }
                if coeff.is_one() {
                    result.push(base);
                } else {
                    result.push(SymExpr::mul(vec![SymExpr::num(coeff), base]));
                }
            }

            SymExpr::add(result)
        }
        SymExprKind::Mul(factors) => {
            let new_factors: Vec<_> = factors.iter().map(collect_like_terms).collect();
            SymExpr::mul(new_factors)
        }
        SymExprKind::Pow(base, exp) => {
            SymExpr::pow(collect_like_terms(base), collect_like_terms(exp))
        }
        SymExprKind::Neg(inner) => SymExpr::neg(collect_like_terms(inner)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(collect_like_terms).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Extract coefficient and base from a term
fn extract_coeff_and_base(expr: &SymExpr) -> (Coefficient, SymExpr) {
    match expr.kind.as_ref() {
        SymExprKind::Num(c) => (c.clone(), SymExpr::int(1)),
        SymExprKind::Neg(inner) => {
            let (c, b) = extract_coeff_and_base(inner);
            (-c, b)
        }
        SymExprKind::Mul(factors) => {
            let mut coeff = Coefficient::int(1);
            let mut non_const = Vec::new();

            for f in factors {
                if let Some(c) = f.as_coeff() {
                    coeff = coeff * c.clone();
                } else {
                    non_const.push(f.clone());
                }
            }

            (coeff, SymExpr::mul(non_const))
        }
        _ => (Coefficient::int(1), expr.clone()),
    }
}

/// Remove identity elements: x + 0 = x, x * 1 = x
fn remove_identity(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Add(terms) => {
            let filtered: Vec<_> = terms
                .iter()
                .map(remove_identity)
                .filter(|t| !t.is_zero())
                .collect();

            if filtered.is_empty() {
                SymExpr::int(0)
            } else {
                SymExpr::add(filtered)
            }
        }
        SymExprKind::Mul(factors) => {
            // Check for zero factor first
            let processed: Vec<_> = factors.iter().map(remove_identity).collect();

            if processed.iter().any(|f| f.is_zero()) {
                return SymExpr::int(0);
            }

            let filtered: Vec<_> = processed.into_iter().filter(|f| !f.is_one()).collect();

            if filtered.is_empty() {
                SymExpr::int(1)
            } else {
                SymExpr::mul(filtered)
            }
        }
        SymExprKind::Pow(base, exp) => SymExpr::pow(remove_identity(base), remove_identity(exp)),
        SymExprKind::Neg(inner) => {
            let new_inner = remove_identity(inner);
            if new_inner.is_zero() {
                SymExpr::int(0)
            } else {
                SymExpr::neg(new_inner)
            }
        }
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(remove_identity).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Simplify double negation: -(-x) = x
fn simplify_neg(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Neg(inner) => {
            let simplified_inner = simplify_neg(inner);

            // -(-x) = x
            if let SymExprKind::Neg(double_inner) = simplified_inner.kind.as_ref() {
                return double_inner.as_ref().clone();
            }

            // -(num) = -num
            if let Some(c) = simplified_inner.as_coeff() {
                return SymExpr::num(-c.clone());
            }

            SymExpr::neg(simplified_inner)
        }
        SymExprKind::Add(terms) => {
            let new_terms: Vec<_> = terms.iter().map(simplify_neg).collect();
            SymExpr::add(new_terms)
        }
        SymExprKind::Mul(factors) => {
            let new_factors: Vec<_> = factors.iter().map(simplify_neg).collect();
            SymExpr::mul(new_factors)
        }
        SymExprKind::Pow(base, exp) => SymExpr::pow(simplify_neg(base), simplify_neg(exp)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(simplify_neg).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Expand products of sums: (a + b) * c = a*c + b*c
fn expand(expr: &SymExpr) -> SymExpr {
    match expr.kind.as_ref() {
        SymExprKind::Mul(factors) => {
            let expanded_factors: Vec<_> = factors.iter().map(expand).collect();

            // Find the first Add factor
            let add_idx = expanded_factors
                .iter()
                .position(|f| matches!(f.kind.as_ref(), SymExprKind::Add(_)));

            if let Some(idx) = add_idx {
                if let SymExprKind::Add(terms) = expanded_factors[idx].kind.as_ref() {
                    // Distribute: (a + b) * rest = a * rest + b * rest
                    let mut new_terms = Vec::new();
                    for t in terms {
                        let mut new_factors = expanded_factors.clone();
                        new_factors[idx] = t.clone();
                        new_terms.push(expand(&SymExpr::mul(new_factors)));
                    }
                    return SymExpr::add(new_terms);
                }
            }

            SymExpr::mul(expanded_factors)
        }
        SymExprKind::Pow(base, exp) => {
            let new_base = expand(base);
            let new_exp = expand(exp);

            // (a + b)^n for small positive integer n: expand binomially
            if let (SymExprKind::Add(terms), Some(c)) = (new_base.kind.as_ref(), new_exp.as_coeff())
            {
                if c.is_integer() && !c.is_negative() {
                    let n = c.to_f64() as u32;
                    if n <= 4 && terms.len() == 2 {
                        // Simple binomial expansion for small cases
                        return expand_binomial(&terms[0], &terms[1], n);
                    }
                }
            }

            SymExpr::pow(new_base, new_exp)
        }
        SymExprKind::Add(terms) => {
            let new_terms: Vec<_> = terms.iter().map(expand).collect();
            SymExpr::add(new_terms)
        }
        SymExprKind::Neg(inner) => SymExpr::neg(expand(inner)),
        SymExprKind::Func(name, args) => {
            let new_args: Vec<_> = args.iter().map(expand).collect();
            SymExpr::func(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Expand (a + b)^n using binomial theorem
fn expand_binomial(a: &SymExpr, b: &SymExpr, n: u32) -> SymExpr {
    if n == 0 {
        return SymExpr::int(1);
    }
    if n == 1 {
        return SymExpr::add(vec![a.clone(), b.clone()]);
    }

    let mut terms = Vec::new();
    for k in 0..=n {
        let binom = binomial(n, k);
        let a_pow = if n - k == 0 {
            SymExpr::int(1)
        } else if n - k == 1 {
            a.clone()
        } else {
            SymExpr::pow(a.clone(), SymExpr::int((n - k) as i64))
        };
        let b_pow = if k == 0 {
            SymExpr::int(1)
        } else if k == 1 {
            b.clone()
        } else {
            SymExpr::pow(b.clone(), SymExpr::int(k as i64))
        };

        let term = SymExpr::mul(vec![SymExpr::int(binom as i64), a_pow, b_pow]);
        terms.push(term);
    }

    SymExpr::add(terms)
}

/// Compute binomial coefficient C(n, k)
fn binomial(n: u32, k: u32) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten() {
        // (x + (y + z)) should flatten to (x + y + z)
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let z = SymExpr::var("z");

        let nested = SymExpr::add(vec![x, SymExpr::add(vec![y, z])]);
        let flat = flatten(&nested);

        if let SymExprKind::Add(terms) = flat.kind.as_ref() {
            assert_eq!(terms.len(), 3);
        } else {
            panic!("Expected Add");
        }
    }

    #[test]
    fn test_merge_constants() {
        // 1 + 2 + x should become 3 + x
        let expr = SymExpr::add(vec![SymExpr::int(1), SymExpr::int(2), SymExpr::var("x")]);
        let merged = merge_constants(&expr);

        // After merging, first term should be 3
        if let SymExprKind::Add(terms) = merged.kind.as_ref() {
            assert!(terms[0].is_num());
        }
    }

    #[test]
    fn test_simplify_powers() {
        // x^1 should become x
        let x = SymExpr::var("x");
        let x_pow_1 = SymExpr::pow(x.clone(), SymExpr::int(1));
        let simplified = simplify_powers(&x_pow_1);
        assert!(simplified.is_var());

        // x^0 should become 1
        let x_pow_0 = SymExpr::pow(x, SymExpr::int(0));
        let simplified = simplify_powers(&x_pow_0);
        assert!(simplified.is_one());
    }

    #[test]
    fn test_default_pipeline() {
        let normalizer = StagedNormalizer::default_pipeline();

        // x + 0 + x should normalize to 2*x
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![x.clone(), SymExpr::int(0), x]);

        let (result, _) = normalizer.normalize(expr);
        // Result should be simpler
        assert!(result.node_count() <= 3);
    }

    #[test]
    fn test_expand() {
        // (x + 1)^2 should expand to x^2 + 2*x + 1
        let x = SymExpr::var("x");
        let expr = SymExpr::pow(SymExpr::add(vec![x, SymExpr::int(1)]), SymExpr::int(2));
        let expanded = expand(&expr);

        if let SymExprKind::Add(terms) = expanded.kind.as_ref() {
            assert_eq!(terms.len(), 3); // x^2, 2*x, 1
        } else {
            panic!("Expected Add after expansion");
        }
    }
}
