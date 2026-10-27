//! Shared typed publication, with actual binary64 revalidation before outputs.
use super::*;
pub(in crate::builtins::math::optim::intlinprog) async fn prepare<V: compiled::Validation>(
    mut outcome: engine::Outcome,
    workspace: &model::Workspace,
    controls: &controls::Controls,
    information: bool,
    validation: &mut V,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Published> {
    let (mut flag,mut message)=match outcome.terminal {
        Terminal::Optimal=>(1,"Certified branch-and-bound gap tolerance reached."),
        Terminal::Infeasible if controls.cutoff.is_some()=>(-2,"No feasible integer point satisfies ObjectiveCutOff."),
        Terminal::Infeasible=>(-2,"All branch domains are exactly certified infeasible."),
        Terminal::Unbounded=>(-3,"Exact feasibility and an integer-compatible rational recession certificate establish an unbounded objective."),
        Terminal::Stopped(stop)=>{
            let flag=if matches!(stop,Stop::Observer){-1}else if outcome.incumbent.is_some(){2}else{0};
            (flag,match stop {
                Stop::Nodes=>"MaxNodes reached; unresolved branch domains and their certified bound are retained.",
                Stop::Time=>"MaxTime reached in the selected invocation clock; unresolved domains remain.",
                Stop::FeasiblePoints=>"MaxFeasiblePoints reached; unresolved domains remain.",
                Stop::Numerical=>"A node lacks a sufficient exact certificate or a progressing representable branch; unresolved domains remain.",
                Stop::Observer=>"Stopped by an output function or plot function; incumbent and certified bounds are retained.",
            })
        }
    };
    let upper = outcome.incumbent.as_ref().map(|v| v.objective.clone());
    let mut x = None;
    let mut fval = None;
    let mut violation = f64::NAN;
    if let Some(point) = outcome.incumbent.as_mut() {
        if let Some(binary) = BinaryPoint::prepare(
            &point.point,
            &workspace.integers,
            &workspace.semis,
            cooperation,
        )
        .await?
        {
            let evaluated = binary
                .evaluate(
                    &mut point.point,
                    &workspace.node(),
                    controls,
                    validation,
                    cooperation,
                )
                .await?;
            violation = evaluated.violation;
            if evaluated.admitted {
                x = Some(binary.values);
                fval = Some(evaluated.objective);
            } else {
                flag = -9;
                message="The certified rational incumbent cannot be published as an integer-feasible binary64 point within ConstraintTolerance and ObjectiveCutOff.";
            }
        } else {
            flag = -9;
            message = "The certified rational incumbent exceeds the finite binary64 output range.";
        }
    }
    let (absolute, relative) = match upper.as_ref().zip(outcome.lower.as_ref()) {
        Some((upper, lower)) if lower <= upper => (
            nonnegative(&(upper - lower)),
            gap::relative(upper, lower)
                .as_ref()
                .map(nonnegative)
                .unwrap_or(f64::INFINITY),
        ),
        _ => (f64::INFINITY, f64::INFINITY),
    };
    let information = if information {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        let mixed = workspace.is_mixed();
        for (name, value) in [
            ("relativegap", relative),
            ("absolutegap", absolute),
            ("numfeaspoints", controls.feasible_points as f64),
            ("numnodes", controls.nodes as f64),
        ] {
            fields.insert(name, if mixed { Value::Num(value) } else { empty() });
        }
        fields.insert("constrviolation", Value::Num(violation));
        fields.insert("message", Value::from(message));
        fields.insert(
            "lowerbound",
            Value::Num(outcome.lower.as_ref().map(number).unwrap_or_else(|| {
                if matches!(outcome.terminal, Terminal::Infeasible) {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            })),
        );
        fields.insert(
            "certifiedobjective",
            Value::Num(upper.as_ref().map(number).unwrap_or(f64::INFINITY)),
        );
        Some(fields.finish())
    } else {
        None
    };
    if matches!(
        controls.display,
        runmat_builtins::OptimDisplay::Final | runmat_builtins::OptimDisplay::Iter
    ) {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!(
                "intlinprog: {message} Nodes: {}. Certified feasible candidates: {}.",
                controls.nodes, controls.feasible_points
            ),
        )?;
    }
    cooperation.completed(0).await?;
    Ok(Published {
        x,
        fval,
        flag,
        information,
    })
}
