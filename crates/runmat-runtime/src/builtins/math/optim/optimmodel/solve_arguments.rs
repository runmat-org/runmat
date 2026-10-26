//! One public solve grammar; typed families constrain numerical solver choices.
use super::*;
use runmat_builtins::{OptimSolver, OptimizationSolveParameter as Parameter};
#[derive(Clone, Copy)]
pub(super) enum Family {
    Optimization,
    Equation,
}
pub(super) struct Parsed<'a> {
    pub starting: Option<&'a Value>,
    pub options: Option<&'a Value>,
    pub selected: Option<OptimSolver>,
}
pub(super) async fn parse<'a>(
    arguments: crate::call::FunctionArgumentView<'a>,
    family: Family,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Parsed<'a>> {
    let starting = arguments
        .get(1)
        .filter(|value| matches!(value, Value::Struct(_)));
    let rest = arguments
        .range(if starting.is_some() { 2 } else { 1 }..)
        .map_err(|e| policy.invalid(e))?;
    if !rest.len().is_multiple_of(2) {
        return Err(policy.invalid("solve accepts an optional starting-point struct followed by Solver and Options name/value pairs"));
    }
    let mut options = None;
    let mut selected = None;
    let mut seen = [false; 2];
    for pair in rest.chunks_exact(2) {
        let name = policy.text(&pair[0], cooperation.runtime())?;
        let parameter = Parameter::parse(&name)
            .ok_or_else(|| policy.invalid("unsupported solve name/value parameter"))?;
        let index = Parameter::ALL
            .iter()
            .position(|&known| known == parameter)
            .expect("canonical solve parameter");
        if seen[index] {
            return Err(policy.invalid("duplicate solve name/value parameter"));
        }
        seen[index] = true;
        match parameter {
            Parameter::Solver => {
                let solver = policy.text(&pair[1], cooperation.runtime())?;
                selected = OptimSolver::parse(&solver);
                let admitted = match family {
                    Family::Optimization => matches!(
                        selected,
                        Some(
                            OptimSolver::Linprog
                                | OptimSolver::Lsqlin
                                | OptimSolver::Quadprog
                                | OptimSolver::Fmincon
                                | OptimSolver::Fminunc
                                | OptimSolver::Lsqnonlin
                                | OptimSolver::Intlinprog
                        )
                    ),
                    Family::Equation => matches!(
                        selected,
                        Some(
                            OptimSolver::Lsqlin
                                | OptimSolver::Fzero
                                | OptimSolver::Fsolve
                                | OptimSolver::Lsqnonlin
                        )
                    ),
                };
                if !admitted {
                    return Err(policy.invalid(match family {
                        Family::Optimization => "this model objective family supports linprog, lsqlin, quadprog, fmincon, fminunc, lsqnonlin and intlinprog; other solver consumers are not implemented",
                        Family::Equation => "equation solve supports lsqlin, fzero, fsolve and lsqnonlin",
                    }));
                }
            }
            Parameter::Options => options = pair.get(1),
        }
        cooperation.completed(1).await?;
    }
    Ok(Parsed {
        starting,
        options,
        selected,
    })
}
