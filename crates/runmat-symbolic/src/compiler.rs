//! Symbolic-to-numeric bytecode compiler
//!
//! Compiles symbolic expressions to stack-based bytecode for efficient
//! numeric evaluation. This is a foundation for JIT compilation.

use crate::expr::{SymExpr, SymExprKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bytecode instruction for the expression evaluator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BytecodeOp {
    /// Push a constant onto the stack
    PushConst(usize), // Index into constants array
    /// Load a variable onto the stack
    LoadVar(usize), // Index into variables array
    /// Binary operations (pop 2, push 1)
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// Unary operations (pop 1, push 1)
    Neg,
    /// Function calls (pop n args, push 1 result)
    Call(String, usize), // Function name, argument count
    /// Duplicate top of stack
    Dup,
    /// Swap top two stack elements
    Swap,
    /// Pop and discard top of stack
    Pop,
}

/// Compiled bytecode representation of a symbolic expression
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    /// Bytecode instructions
    pub ops: Vec<BytecodeOp>,
    /// Constant values
    pub constants: Vec<f64>,
    /// Variable names (in order)
    pub variables: Vec<String>,
    /// Maximum stack depth required
    pub max_stack: usize,
}

impl CompiledExpr {
    /// Evaluate the compiled expression with given variable values
    pub fn eval(&self, var_values: &[f64]) -> Result<f64, String> {
        if var_values.len() != self.variables.len() {
            return Err(format!(
                "Expected {} variable values, got {}",
                self.variables.len(),
                var_values.len()
            ));
        }

        let mut stack: Vec<f64> = Vec::with_capacity(self.max_stack);

        for op in &self.ops {
            match op {
                BytecodeOp::PushConst(idx) => {
                    stack.push(self.constants[*idx]);
                }
                BytecodeOp::LoadVar(idx) => {
                    stack.push(var_values[*idx]);
                }
                BytecodeOp::Add => {
                    let b = stack.pop().ok_or("Stack underflow")?;
                    let a = stack.pop().ok_or("Stack underflow")?;
                    stack.push(a + b);
                }
                BytecodeOp::Sub => {
                    let b = stack.pop().ok_or("Stack underflow")?;
                    let a = stack.pop().ok_or("Stack underflow")?;
                    stack.push(a - b);
                }
                BytecodeOp::Mul => {
                    let b = stack.pop().ok_or("Stack underflow")?;
                    let a = stack.pop().ok_or("Stack underflow")?;
                    stack.push(a * b);
                }
                BytecodeOp::Div => {
                    let b = stack.pop().ok_or("Stack underflow")?;
                    let a = stack.pop().ok_or("Stack underflow")?;
                    stack.push(a / b);
                }
                BytecodeOp::Pow => {
                    let b = stack.pop().ok_or("Stack underflow")?;
                    let a = stack.pop().ok_or("Stack underflow")?;
                    stack.push(a.powf(b));
                }
                BytecodeOp::Neg => {
                    let a = stack.pop().ok_or("Stack underflow")?;
                    stack.push(-a);
                }
                BytecodeOp::Call(name, argc) => {
                    let mut args = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(stack.pop().ok_or("Stack underflow")?);
                    }
                    args.reverse();
                    let result = call_function(name, &args)?;
                    stack.push(result);
                }
                BytecodeOp::Dup => {
                    let a = *stack.last().ok_or("Stack underflow")?;
                    stack.push(a);
                }
                BytecodeOp::Swap => {
                    let len = stack.len();
                    if len < 2 {
                        return Err("Stack underflow".to_string());
                    }
                    stack.swap(len - 1, len - 2);
                }
                BytecodeOp::Pop => {
                    stack.pop().ok_or("Stack underflow")?;
                }
            }
        }

        stack.pop().ok_or_else(|| "Empty stack at end".to_string())
    }

    /// Get variable names for binding
    pub fn variable_names(&self) -> &[String] {
        &self.variables
    }

    /// Evaluate the compiled expression for multiple input sets (vectorized)
    pub fn eval_batch(&self, var_values_batch: &[&[f64]]) -> Result<Vec<f64>, String> {
        let n_vars = self.variables.len();

        for (i, vars) in var_values_batch.iter().enumerate() {
            if vars.len() != n_vars {
                return Err(format!(
                    "Input set {} has {} values, expected {}",
                    i,
                    vars.len(),
                    n_vars
                ));
            }
        }

        let mut results = Vec::with_capacity(var_values_batch.len());
        let mut stack: Vec<f64> = Vec::with_capacity(self.max_stack);

        for var_values in var_values_batch {
            stack.clear();

            for op in &self.ops {
                match op {
                    BytecodeOp::PushConst(idx) => {
                        stack.push(self.constants[*idx]);
                    }
                    BytecodeOp::LoadVar(idx) => {
                        stack.push(var_values[*idx]);
                    }
                    BytecodeOp::Add => {
                        let b = stack.pop().ok_or("Stack underflow")?;
                        let a = stack.pop().ok_or("Stack underflow")?;
                        stack.push(a + b);
                    }
                    BytecodeOp::Sub => {
                        let b = stack.pop().ok_or("Stack underflow")?;
                        let a = stack.pop().ok_or("Stack underflow")?;
                        stack.push(a - b);
                    }
                    BytecodeOp::Mul => {
                        let b = stack.pop().ok_or("Stack underflow")?;
                        let a = stack.pop().ok_or("Stack underflow")?;
                        stack.push(a * b);
                    }
                    BytecodeOp::Div => {
                        let b = stack.pop().ok_or("Stack underflow")?;
                        let a = stack.pop().ok_or("Stack underflow")?;
                        stack.push(a / b);
                    }
                    BytecodeOp::Pow => {
                        let b = stack.pop().ok_or("Stack underflow")?;
                        let a = stack.pop().ok_or("Stack underflow")?;
                        stack.push(a.powf(b));
                    }
                    BytecodeOp::Neg => {
                        let a = stack.pop().ok_or("Stack underflow")?;
                        stack.push(-a);
                    }
                    BytecodeOp::Call(name, argc) => {
                        let mut args = Vec::with_capacity(*argc);
                        for _ in 0..*argc {
                            args.push(stack.pop().ok_or("Stack underflow")?);
                        }
                        args.reverse();
                        let result = call_function(name, &args)?;
                        stack.push(result);
                    }
                    BytecodeOp::Dup => {
                        let a = *stack.last().ok_or("Stack underflow")?;
                        stack.push(a);
                    }
                    BytecodeOp::Swap => {
                        let len = stack.len();
                        if len < 2 {
                            return Err("Stack underflow".to_string());
                        }
                        stack.swap(len - 1, len - 2);
                    }
                    BytecodeOp::Pop => {
                        stack.pop().ok_or("Stack underflow")?;
                    }
                }
            }

            results.push(
                stack
                    .pop()
                    .ok_or_else(|| "Empty stack at end".to_string())?,
            );
        }

        Ok(results)
    }

    /// Evaluate for a single variable over a range of values
    pub fn eval_range(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        if self.variables.len() != 1 {
            return Err(format!(
                "eval_range requires single-variable expression, got {} variables",
                self.variables.len()
            ));
        }

        let batch: Vec<&[f64]> = values.iter().map(std::slice::from_ref).collect();
        self.eval_batch(&batch)
    }
}

/// Evaluate a built-in function
fn call_function(name: &str, args: &[f64]) -> Result<f64, String> {
    match name {
        "sin" if args.len() == 1 => Ok(args[0].sin()),
        "cos" if args.len() == 1 => Ok(args[0].cos()),
        "tan" if args.len() == 1 => Ok(args[0].tan()),
        "asin" if args.len() == 1 => Ok(args[0].asin()),
        "acos" if args.len() == 1 => Ok(args[0].acos()),
        "atan" if args.len() == 1 => Ok(args[0].atan()),
        "sinh" if args.len() == 1 => Ok(args[0].sinh()),
        "cosh" if args.len() == 1 => Ok(args[0].cosh()),
        "tanh" if args.len() == 1 => Ok(args[0].tanh()),
        "exp" if args.len() == 1 => Ok(args[0].exp()),
        "log" if args.len() == 1 => Ok(args[0].ln()),
        "log10" if args.len() == 1 => Ok(args[0].log10()),
        "log2" if args.len() == 1 => Ok(args[0].log2()),
        "sqrt" if args.len() == 1 => Ok(args[0].sqrt()),
        "abs" if args.len() == 1 => Ok(args[0].abs()),
        "sign" if args.len() == 1 => Ok(args[0].signum()),
        "floor" if args.len() == 1 => Ok(args[0].floor()),
        "ceil" if args.len() == 1 => Ok(args[0].ceil()),
        "round" if args.len() == 1 => Ok(args[0].round()),
        "atan2" if args.len() == 2 => Ok(args[0].atan2(args[1])),
        "max" if args.len() == 2 => Ok(args[0].max(args[1])),
        "min" if args.len() == 2 => Ok(args[0].min(args[1])),
        _ => Err(format!("Unknown function: {}({})", name, args.len())),
    }
}

/// Bytecode compiler for symbolic expressions
#[derive(Debug, Default)]
pub struct BytecodeCompiler {
    constants: Vec<f64>,
    const_map: HashMap<u64, usize>, // f64 bits -> index
    variables: Vec<String>,
    var_map: HashMap<String, usize>,
}

impl BytecodeCompiler {
    pub fn new() -> Self {
        BytecodeCompiler {
            constants: Vec::new(),
            const_map: HashMap::new(),
            variables: Vec::new(),
            var_map: HashMap::new(),
        }
    }

    /// Compile a symbolic expression to bytecode
    pub fn compile(&mut self, expr: &SymExpr) -> CompiledExpr {
        let mut ops = Vec::new();
        let mut max_stack = 0;
        let mut current_stack = 0;

        self.compile_expr(expr, &mut ops, &mut current_stack, &mut max_stack);

        CompiledExpr {
            ops,
            constants: self.constants.clone(),
            variables: self.variables.clone(),
            max_stack,
        }
    }

    /// Compile with explicit variable ordering
    pub fn compile_with_vars(&mut self, expr: &SymExpr, var_order: &[&str]) -> CompiledExpr {
        // Pre-register variables in specified order
        for v in var_order {
            if !self.var_map.contains_key(*v) {
                let idx = self.variables.len();
                self.variables.push(v.to_string());
                self.var_map.insert(v.to_string(), idx);
            }
        }

        self.compile(expr)
    }

    fn add_constant(&mut self, value: f64) -> usize {
        let bits = value.to_bits();
        if let Some(&idx) = self.const_map.get(&bits) {
            return idx;
        }
        let idx = self.constants.len();
        self.constants.push(value);
        self.const_map.insert(bits, idx);
        idx
    }

    fn add_variable(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.var_map.get(name) {
            return idx;
        }
        let idx = self.variables.len();
        self.variables.push(name.to_string());
        self.var_map.insert(name.to_string(), idx);
        idx
    }

    fn compile_expr(
        &mut self,
        expr: &SymExpr,
        ops: &mut Vec<BytecodeOp>,
        current_stack: &mut usize,
        max_stack: &mut usize,
    ) {
        match expr.kind.as_ref() {
            SymExprKind::Num(c) => {
                let idx = self.add_constant(c.to_f64());
                ops.push(BytecodeOp::PushConst(idx));
                *current_stack += 1;
                *max_stack = (*max_stack).max(*current_stack);
            }

            SymExprKind::Var(s) => {
                let idx = self.add_variable(&s.name);
                ops.push(BytecodeOp::LoadVar(idx));
                *current_stack += 1;
                *max_stack = (*max_stack).max(*current_stack);
            }

            SymExprKind::Add(terms) => {
                if terms.is_empty() {
                    let idx = self.add_constant(0.0);
                    ops.push(BytecodeOp::PushConst(idx));
                    *current_stack += 1;
                    *max_stack = (*max_stack).max(*current_stack);
                    return;
                }

                self.compile_expr(&terms[0], ops, current_stack, max_stack);
                for t in &terms[1..] {
                    self.compile_expr(t, ops, current_stack, max_stack);
                    ops.push(BytecodeOp::Add);
                    *current_stack -= 1;
                }
            }

            SymExprKind::Mul(factors) => {
                if factors.is_empty() {
                    let idx = self.add_constant(1.0);
                    ops.push(BytecodeOp::PushConst(idx));
                    *current_stack += 1;
                    *max_stack = (*max_stack).max(*current_stack);
                    return;
                }

                self.compile_expr(&factors[0], ops, current_stack, max_stack);
                for f in &factors[1..] {
                    self.compile_expr(f, ops, current_stack, max_stack);
                    ops.push(BytecodeOp::Mul);
                    *current_stack -= 1;
                }
            }

            SymExprKind::Pow(base, exp) => {
                self.compile_expr(base, ops, current_stack, max_stack);
                self.compile_expr(exp, ops, current_stack, max_stack);
                ops.push(BytecodeOp::Pow);
                *current_stack -= 1;
            }

            SymExprKind::Neg(inner) => {
                self.compile_expr(inner, ops, current_stack, max_stack);
                ops.push(BytecodeOp::Neg);
            }

            SymExprKind::Func(name, args) => {
                for arg in args {
                    self.compile_expr(arg, ops, current_stack, max_stack);
                }
                ops.push(BytecodeOp::Call(name.clone(), args.len()));
                *current_stack -= args.len();
                *current_stack += 1; // result
                *max_stack = (*max_stack).max(*current_stack);
            }
        }
    }
}

/// Convenience function to compile an expression
pub fn compile(expr: &SymExpr) -> CompiledExpr {
    let mut compiler = BytecodeCompiler::new();
    compiler.compile(expr)
}

/// Compile with explicit variable ordering
pub fn compile_with_vars(expr: &SymExpr, var_order: &[&str]) -> CompiledExpr {
    let mut compiler = BytecodeCompiler::new();
    compiler.compile_with_vars(expr, var_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_constant() {
        let expr = SymExpr::int(42);
        let compiled = compile(&expr);

        assert_eq!(compiled.constants, vec![42.0]);
        assert!(compiled.variables.is_empty());

        let result = compiled.eval(&[]).unwrap();
        assert!((result - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_compile_variable() {
        let expr = SymExpr::var("x");
        let compiled = compile(&expr);

        assert_eq!(compiled.variables, vec!["x"]);

        let result = compiled.eval(&[5.0]).unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compile_polynomial() {
        // x^2 + 2*x + 1
        let x = SymExpr::var("x");
        let expr = SymExpr::add(vec![
            SymExpr::pow(x.clone(), SymExpr::int(2)),
            SymExpr::mul(vec![SymExpr::int(2), x.clone()]),
            SymExpr::int(1),
        ]);

        let compiled = compile(&expr);
        assert_eq!(compiled.variables, vec!["x"]);

        // Evaluate at x = 3: 9 + 6 + 1 = 16
        let result = compiled.eval(&[3.0]).unwrap();
        assert!((result - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_compile_function() {
        let x = SymExpr::var("x");
        let expr = SymExpr::sin(x);

        let compiled = compile(&expr);

        let result = compiled.eval(&[std::f64::consts::PI / 2.0]).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compile_with_var_order() {
        let x = SymExpr::var("x");
        let y = SymExpr::var("y");
        let expr = y.clone() - x.clone(); // y - x

        let compiled = compile_with_vars(&expr, &["x", "y"]);
        assert_eq!(compiled.variables, vec!["x", "y"]);

        // With x=1, y=5: 5 - 1 = 4
        let result = compiled.eval(&[1.0, 5.0]).unwrap();
        assert!((result - 4.0).abs() < 1e-10);
    }
}
