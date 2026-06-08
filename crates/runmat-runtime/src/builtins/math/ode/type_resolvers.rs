use runmat_builtins::{ResolveContext, Type};

pub fn ode_solution_type(_args: &[Type], _context: &ResolveContext) -> Type {
    // ODE solvers return an output matrix of shape [T, N], where T depends on
    // runtime integration behavior (adaptive stepping or tspan sample count).
    // Because T cannot be inferred statically from y0, keep the shape unknown.
    Type::tensor()
}
