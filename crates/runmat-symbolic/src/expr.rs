//! Symbolic expression representation
//!
//! The core expression type for symbolic computation.

use crate::coeff::Coefficient;
use crate::symbol::Symbol;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

/// A symbolic expression
///
/// Expressions are immutable and use reference counting for efficient cloning.
/// They support the standard mathematical operations.
#[derive(Debug, Clone)]
pub struct SymExpr {
    pub kind: Arc<SymExprKind>,
}

impl PartialEq for SymExpr {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for SymExpr {}

impl Hash for SymExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

/// The kind of symbolic expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymExprKind {
    /// Numeric coefficient (rational or float)
    Num(Coefficient),

    /// Symbolic variable
    Var(Symbol),

    /// Addition: sum of terms
    Add(Vec<SymExpr>),

    /// Multiplication: product of factors
    Mul(Vec<SymExpr>),

    /// Power: base^exponent
    Pow(Box<SymExpr>, Box<SymExpr>),

    /// Negation: -expr
    Neg(Box<SymExpr>),

    /// Function application: name(args...)
    Func(String, Vec<SymExpr>),
}

impl SymExpr {
    // ========== Constructors ==========

    /// Create a numeric constant from an integer
    pub fn int(n: i64) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Num(Coefficient::int(n))),
        }
    }

    /// Create a numeric constant from a float
    pub fn float(f: f64) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Num(Coefficient::from_f64_exact(f))),
        }
    }

    /// Create a numeric constant from a coefficient
    pub fn num(c: Coefficient) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Num(c)),
        }
    }

    /// Create a symbolic variable
    pub fn var(name: impl Into<String>) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Var(Symbol::new(name))),
        }
    }

    /// Create a symbolic variable from a Symbol
    pub fn symbol(s: Symbol) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Var(s)),
        }
    }

    /// Create an addition (sum) expression
    pub fn add(terms: Vec<SymExpr>) -> Self {
        if terms.is_empty() {
            return Self::int(0);
        }
        if terms.len() == 1 {
            return terms.into_iter().next().unwrap();
        }
        SymExpr {
            kind: Arc::new(SymExprKind::Add(terms)),
        }
    }

    /// Create a multiplication (product) expression
    pub fn mul(factors: Vec<SymExpr>) -> Self {
        if factors.is_empty() {
            return Self::int(1);
        }
        if factors.len() == 1 {
            return factors.into_iter().next().unwrap();
        }
        SymExpr {
            kind: Arc::new(SymExprKind::Mul(factors)),
        }
    }

    /// Create a power expression
    pub fn pow(base: SymExpr, exp: SymExpr) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Pow(Box::new(base), Box::new(exp))),
        }
    }

    /// Create a negation expression
    #[allow(clippy::should_implement_trait)]
    pub fn neg(expr: SymExpr) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Neg(Box::new(expr))),
        }
    }

    /// Create a function application
    pub fn func(name: impl Into<String>, args: Vec<SymExpr>) -> Self {
        SymExpr {
            kind: Arc::new(SymExprKind::Func(name.into(), args)),
        }
    }

    // ========== Common mathematical functions ==========

    /// sin(x)
    pub fn sin(arg: SymExpr) -> Self {
        Self::func("sin", vec![arg])
    }

    /// cos(x)
    pub fn cos(arg: SymExpr) -> Self {
        Self::func("cos", vec![arg])
    }

    /// tan(x)
    pub fn tan(arg: SymExpr) -> Self {
        Self::func("tan", vec![arg])
    }

    /// exp(x)
    pub fn exp(arg: SymExpr) -> Self {
        Self::func("exp", vec![arg])
    }

    /// log(x) - natural logarithm
    pub fn log(arg: SymExpr) -> Self {
        Self::func("log", vec![arg])
    }

    /// sqrt(x)
    pub fn sqrt(arg: SymExpr) -> Self {
        Self::func("sqrt", vec![arg])
    }

    /// abs(x)
    pub fn abs(arg: SymExpr) -> Self {
        Self::func("abs", vec![arg])
    }

    // ========== Predicates ==========

    /// Check if expression is a numeric constant
    pub fn is_num(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Num(_))
    }

    /// Check if expression is zero
    pub fn is_zero(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Num(c) if c.is_zero())
    }

    /// Check if expression is one
    pub fn is_one(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Num(c) if c.is_one())
    }

    /// Check if expression is a variable
    pub fn is_var(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Var(_))
    }

    /// Check if expression is an addition
    pub fn is_add(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Add(_))
    }

    /// Check if expression is a multiplication
    pub fn is_mul(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Mul(_))
    }

    /// Check if expression is a power
    pub fn is_pow(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Pow(_, _))
    }

    /// Check if expression is a function call
    pub fn is_func(&self) -> bool {
        matches!(self.kind.as_ref(), SymExprKind::Func(_, _))
    }

    // ========== Accessors ==========

    /// Get the numeric coefficient if this is a Num
    pub fn as_coeff(&self) -> Option<&Coefficient> {
        match self.kind.as_ref() {
            SymExprKind::Num(c) => Some(c),
            _ => None,
        }
    }

    /// Get the variable if this is a Var
    pub fn as_var(&self) -> Option<&Symbol> {
        match self.kind.as_ref() {
            SymExprKind::Var(s) => Some(s),
            _ => None,
        }
    }

    /// Get add terms if this is an Add
    pub fn as_add(&self) -> Option<&[SymExpr]> {
        match self.kind.as_ref() {
            SymExprKind::Add(terms) => Some(terms),
            _ => None,
        }
    }

    /// Get mul factors if this is a Mul
    pub fn as_mul(&self) -> Option<&[SymExpr]> {
        match self.kind.as_ref() {
            SymExprKind::Mul(factors) => Some(factors),
            _ => None,
        }
    }

    /// Get (base, exp) if this is a Pow
    pub fn as_pow(&self) -> Option<(&SymExpr, &SymExpr)> {
        match self.kind.as_ref() {
            SymExprKind::Pow(base, exp) => Some((base, exp)),
            _ => None,
        }
    }

    /// Get (name, args) if this is a Func
    pub fn as_func(&self) -> Option<(&str, &[SymExpr])> {
        match self.kind.as_ref() {
            SymExprKind::Func(name, args) => Some((name, args)),
            _ => None,
        }
    }

    // ========== Analysis ==========

    /// Collect all free variables in the expression
    pub fn free_vars(&self) -> HashSet<Symbol> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<Symbol>) {
        match self.kind.as_ref() {
            SymExprKind::Num(_) => {}
            SymExprKind::Var(s) => {
                vars.insert(s.clone());
            }
            SymExprKind::Add(terms) => {
                for t in terms {
                    t.collect_vars(vars);
                }
            }
            SymExprKind::Mul(factors) => {
                for f in factors {
                    f.collect_vars(vars);
                }
            }
            SymExprKind::Pow(base, exp) => {
                base.collect_vars(vars);
                exp.collect_vars(vars);
            }
            SymExprKind::Neg(inner) => {
                inner.collect_vars(vars);
            }
            SymExprKind::Func(_, args) => {
                for a in args {
                    a.collect_vars(vars);
                }
            }
        }
    }

    /// Count the number of nodes in the expression tree
    pub fn node_count(&self) -> usize {
        match self.kind.as_ref() {
            SymExprKind::Num(_) | SymExprKind::Var(_) => 1,
            SymExprKind::Add(terms) => 1 + terms.iter().map(|t| t.node_count()).sum::<usize>(),
            SymExprKind::Mul(factors) => 1 + factors.iter().map(|f| f.node_count()).sum::<usize>(),
            SymExprKind::Pow(base, exp) => 1 + base.node_count() + exp.node_count(),
            SymExprKind::Neg(inner) => 1 + inner.node_count(),
            SymExprKind::Func(_, args) => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
        }
    }

    // ========== Transformations ==========

    /// Substitute a variable with an expression
    pub fn substitute(&self, var: &str, replacement: &SymExpr) -> SymExpr {
        match self.kind.as_ref() {
            SymExprKind::Num(_) => self.clone(),
            SymExprKind::Var(s) if s.name == var => replacement.clone(),
            SymExprKind::Var(_) => self.clone(),
            SymExprKind::Add(terms) => {
                let new_terms: Vec<_> = terms
                    .iter()
                    .map(|t| t.substitute(var, replacement))
                    .collect();
                SymExpr::add(new_terms)
            }
            SymExprKind::Mul(factors) => {
                let new_factors: Vec<_> = factors
                    .iter()
                    .map(|f| f.substitute(var, replacement))
                    .collect();
                SymExpr::mul(new_factors)
            }
            SymExprKind::Pow(base, exp) => {
                let new_base = base.substitute(var, replacement);
                let new_exp = exp.substitute(var, replacement);
                SymExpr::pow(new_base, new_exp)
            }
            SymExprKind::Neg(inner) => {
                let new_inner = inner.substitute(var, replacement);
                SymExpr::neg(new_inner)
            }
            SymExprKind::Func(name, args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| a.substitute(var, replacement))
                    .collect();
                SymExpr::func(name.clone(), new_args)
            }
        }
    }

    /// Differentiate with respect to a variable
    pub fn diff(&self, var: &str) -> SymExpr {
        match self.kind.as_ref() {
            SymExprKind::Num(_) => SymExpr::int(0),
            SymExprKind::Var(s) if s.name == var => SymExpr::int(1),
            SymExprKind::Var(_) => SymExpr::int(0),

            // d/dx (a + b + ...) = da/dx + db/dx + ...
            SymExprKind::Add(terms) => {
                let dterms: Vec<_> = terms.iter().map(|t| t.diff(var)).collect();
                SymExpr::add(dterms)
            }

            // d/dx (a * b) = a'*b + a*b' (generalized product rule)
            SymExprKind::Mul(factors) => {
                if factors.is_empty() {
                    return SymExpr::int(0);
                }
                let mut sum_terms = Vec::new();
                for i in 0..factors.len() {
                    let mut prod_factors = Vec::new();
                    for (j, f) in factors.iter().enumerate() {
                        if i == j {
                            prod_factors.push(f.diff(var));
                        } else {
                            prod_factors.push(f.clone());
                        }
                    }
                    sum_terms.push(SymExpr::mul(prod_factors));
                }
                SymExpr::add(sum_terms)
            }

            // d/dx (base^exp) using chain rule
            // = base^exp * (exp' * ln(base) + exp * base'/base)
            SymExprKind::Pow(base, exp) => {
                let base_diff = base.diff(var);
                let exp_diff = exp.diff(var);

                // If exponent is constant: d/dx (f^n) = n * f^(n-1) * f'
                if exp_diff.is_zero() {
                    let n_minus_one = SymExpr::add(vec![exp.as_ref().clone(), SymExpr::int(-1)]);
                    let new_pow = SymExpr::pow(base.as_ref().clone(), n_minus_one);
                    SymExpr::mul(vec![exp.as_ref().clone(), new_pow, base_diff])
                }
                // If base is constant: d/dx (a^f) = a^f * ln(a) * f'
                else if base_diff.is_zero() {
                    let log_base = SymExpr::log(base.as_ref().clone());
                    SymExpr::mul(vec![self.clone(), log_base, exp_diff])
                }
                // General case: d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f)
                else {
                    let log_base = SymExpr::log(base.as_ref().clone());
                    let term1 = SymExpr::mul(vec![exp_diff, log_base]);
                    let term2 = SymExpr::mul(vec![
                        exp.as_ref().clone(),
                        base_diff,
                        SymExpr::pow(base.as_ref().clone(), SymExpr::int(-1)),
                    ]);
                    SymExpr::mul(vec![self.clone(), SymExpr::add(vec![term1, term2])])
                }
            }

            // d/dx (-f) = -f'
            SymExprKind::Neg(inner) => SymExpr::neg(inner.diff(var)),

            // Function derivatives (chain rule)
            SymExprKind::Func(name, args) if args.len() == 1 => {
                let arg = &args[0];
                let arg_diff = arg.diff(var);
                let outer_diff = match name.as_str() {
                    "sin" => SymExpr::cos(arg.clone()),
                    "cos" => SymExpr::neg(SymExpr::sin(arg.clone())),
                    "tan" => {
                        // sec^2(x) = 1/cos^2(x)
                        let cos_x = SymExpr::cos(arg.clone());
                        SymExpr::pow(cos_x, SymExpr::int(-2))
                    }
                    "exp" => SymExpr::exp(arg.clone()),
                    "log" => SymExpr::pow(arg.clone(), SymExpr::int(-1)),
                    "sqrt" => {
                        // 1/(2*sqrt(x))
                        let half = SymExpr::num(Coefficient::rational(1, 2));
                        let sqrt_x = SymExpr::sqrt(arg.clone());
                        SymExpr::mul(vec![half, SymExpr::pow(sqrt_x, SymExpr::int(-1))])
                    }
                    "abs" => {
                        // d/dx |x| = sign(x) - not differentiable at 0
                        SymExpr::func("sign", vec![arg.clone()])
                    }
                    _ => {
                        // Unknown function: return symbolic derivative
                        SymExpr::func(format!("D_{}", name), vec![arg.clone()])
                    }
                };
                SymExpr::mul(vec![outer_diff, arg_diff])
            }

            SymExprKind::Func(name, _args) => {
                // Multi-argument function: return symbolic derivative
                SymExpr::func(format!("D_{}", name), vec![self.clone()])
            }
        }
    }

    /// Evaluate numerically given variable values
    pub fn eval(&self, vars: &std::collections::HashMap<String, f64>) -> Option<f64> {
        match self.kind.as_ref() {
            SymExprKind::Num(c) => Some(c.to_f64()),
            SymExprKind::Var(s) => vars.get(&s.name).copied(),
            SymExprKind::Add(terms) => {
                let mut sum = 0.0;
                for t in terms {
                    sum += t.eval(vars)?;
                }
                Some(sum)
            }
            SymExprKind::Mul(factors) => {
                let mut prod = 1.0;
                for f in factors {
                    prod *= f.eval(vars)?;
                }
                Some(prod)
            }
            SymExprKind::Pow(base, exp) => {
                let b = base.eval(vars)?;
                let e = exp.eval(vars)?;
                Some(b.powf(e))
            }
            SymExprKind::Neg(inner) => Some(-inner.eval(vars)?),
            SymExprKind::Func(name, args) => {
                let arg_vals: Option<Vec<_>> = args.iter().map(|a| a.eval(vars)).collect();
                let vals = arg_vals?;
                match name.as_str() {
                    "sin" if vals.len() == 1 => Some(vals[0].sin()),
                    "cos" if vals.len() == 1 => Some(vals[0].cos()),
                    "tan" if vals.len() == 1 => Some(vals[0].tan()),
                    "exp" if vals.len() == 1 => Some(vals[0].exp()),
                    "log" if vals.len() == 1 => Some(vals[0].ln()),
                    "sqrt" if vals.len() == 1 => Some(vals[0].sqrt()),
                    "abs" if vals.len() == 1 => Some(vals[0].abs()),
                    "sign" if vals.len() == 1 => Some(vals[0].signum()),
                    _ => None, // Unknown function
                }
            }
        }
    }
}

// ========== Operator Overloads ==========

impl std::ops::Add for SymExpr {
    type Output = SymExpr;

    fn add(self, rhs: Self) -> Self::Output {
        SymExpr::add(vec![self, rhs])
    }
}

impl std::ops::Sub for SymExpr {
    type Output = SymExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        SymExpr::add(vec![self, SymExpr::neg(rhs)])
    }
}

impl std::ops::Mul for SymExpr {
    type Output = SymExpr;

    fn mul(self, rhs: Self) -> Self::Output {
        SymExpr::mul(vec![self, rhs])
    }
}

impl std::ops::Div for SymExpr {
    type Output = SymExpr;

    fn div(self, rhs: Self) -> Self::Output {
        SymExpr::mul(vec![self, SymExpr::pow(rhs, SymExpr::int(-1))])
    }
}

impl std::ops::Neg for SymExpr {
    type Output = SymExpr;

    fn neg(self) -> Self::Output {
        SymExpr::neg(self)
    }
}

// ========== Display ==========

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind.as_ref() {
            SymExprKind::Num(c) => write!(f, "{}", c),
            SymExprKind::Var(s) => write!(f, "{}", s.name),
            SymExprKind::Add(terms) => {
                if terms.is_empty() {
                    return write!(f, "0");
                }
                write!(f, "(")?;
                for (i, t) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            SymExprKind::Mul(factors) => {
                if factors.is_empty() {
                    return write!(f, "1");
                }
                write!(f, "(")?;
                for (i, t) in factors.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            SymExprKind::Pow(base, exp) => {
                write!(f, "{}^{}", base, exp)
            }
            SymExprKind::Neg(inner) => {
                write!(f, "(-{})", inner)
            }
            SymExprKind::Func(name, args) => {
                write!(f, "{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let two = SymExpr::int(2);

        let expr = x.clone() + y.clone() * two;
        assert!(expr.is_add());
    }

    #[test]
    fn test_free_vars() {
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let expr = x + y * SymExpr::int(2);

        let vars = expr.free_vars();
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_substitution() {
        let x = SymExpr::var("x");
        let expr = x.clone() + SymExpr::int(1);

        let result = expr.substitute("x", &SymExpr::int(2));
        // Should simplify to 2 + 1 (not normalized yet)
        assert!(result.is_add());
    }

    #[test]
    fn test_differentiation() {
        // d/dx (x^2) = 2*x
        let x = SymExpr::var("x");
        let x_squared = SymExpr::pow(x.clone(), SymExpr::int(2));
        let deriv = x_squared.diff("x");

        // Should produce 2 * x^1 * 1 (before normalization)
        assert!(deriv.is_mul());
    }

    #[test]
    fn test_evaluation() {
        let x = SymExpr::var("x");
        let expr = x.clone() * x.clone() + SymExpr::int(1);

        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 3.0);

        let result = expr.eval(&vars);
        assert_eq!(result, Some(10.0)); // 3^2 + 1 = 10
    }
}
