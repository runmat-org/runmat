//! Original public argument roles, admitted before provider materialization.
use super::{common, invalid, numeric, ScalarCooperation, NAME};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticPopulation;
use runmat_value::Value;
pub(super) mod raw;
use common::constrained_numeric::Geometry;

pub(super) struct Problem {
    pub objective: Value,
    pub nonlinear: Option<Value>,
    pub variables: usize,
    pub linear: Option<super::linear::Data>,
    pub rng: Option<crate::builtins::common::random::RngSnapshot>,
    pub public: Value,
    pub has_linear: bool,
    pub has_bounds: bool,
    pub mixed_integer: bool,
}

fn public_problem(raw: &raw::Raw) -> Value {
    // One shallow public view of the supplied problem, not a duplicate sparse
    // model. Actual normalized rows and feasibility proofs have a different
    // internal owner; callbacks are not given a mutable mirror of that owner.
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in [
        ("fitnessfcn", &raw.objective),
        ("FitnessFcn", &raw.objective),
        ("nvars", &raw.variables),
        ("Aineq", &raw.inequalities),
        ("Bineq", &raw.inequality_rhs),
        ("Aeq", &raw.equalities),
        ("Beq", &raw.equality_rhs),
        ("lb", &raw.lower),
        ("ub", &raw.upper),
        ("nonlcon", &raw.nonlinear),
        ("intcon", &raw.integers),
    ] {
        fields.insert(name, value.clone());
    }
    fields.insert("options", raw::empty());
    fields.insert("solver", Value::from(NAME));
    Value::Struct(fields.finish())
}

pub(super) fn parse(
    args: crate::call::BuiltinArguments,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<raw::Raw> {
    raw::parse(args, cooperation.runtime())
}

/// PopulationType is resolved without numeric transfer first. Check every
/// active data role before numeric controls or inputs can be materialized.
pub(super) fn preflight(
    raw: &raw::Raw,
    population: OptimGeneticPopulation,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    numeric::preflight(
        &raw.variables,
        "nvars",
        Geometry::Scalar,
        cooperation.runtime(),
    )?;
    numeric::preflight(
        &raw.integers,
        "intcon",
        Geometry::Vector,
        cooperation.runtime(),
    )?;
    if population != OptimGeneticPopulation::DoubleVector {
        if !raw::unspecified(&raw.integers) {
            return Err(invalid(
                "integer constraints require doubleVector PopulationType",
            ));
        }
        return Ok(());
    }
    if !raw::unspecified(&raw.nonlinear) && !raw::callable(&raw.nonlinear) {
        return Err(invalid(
            "nonlcon must be one function handle, function name or []",
        ));
    }
    for (value, role, geometry) in [
        (&raw.inequalities, "A", Geometry::MatrixAny),
        (&raw.inequality_rhs, "b", Geometry::Vector),
        (&raw.equalities, "Aeq", Geometry::MatrixAny),
        (&raw.equality_rhs, "beq", Geometry::Vector),
        (&raw.lower, "lb", Geometry::Vector),
        (&raw.upper, "ub", Geometry::Vector),
    ] {
        numeric::preflight(value, role, geometry, cooperation.runtime())?;
    }
    Ok(())
}

/// Called only after the selected option owner has resolved necessary control
/// selectors and admitted/materialized all active controls. No input transfer
/// can therefore precede a late active-option compatibility/intent refusal.
pub(super) async fn prepare(
    mut raw: raw::Raw,
    population: OptimGeneticPopulation,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    let public = public_problem(&raw);
    if population != OptimGeneticPopulation::DoubleVector {
        // These inputs are documented inactive for bitstring/custom genomes.
        // Preserve the supplied public problem view without inspecting or
        // downloading inactive data or resolving a dormant nonlcon callback.
        raw.inequalities = raw::empty();
        raw.inequality_rhs = raw::empty();
        raw.equalities = raw::empty();
        raw.equality_rhs = raw::empty();
        raw.lower = raw::empty();
        raw.upper = raw::empty();
        raw.nonlinear = raw::empty();
    }
    let variables = numeric::variables(raw.variables, cooperation).await?;
    let inequalities = numeric::pair(
        raw.inequalities,
        raw.inequality_rhs,
        variables,
        "linear inequalities",
        cooperation,
    )
    .await?;
    let equalities = numeric::pair(
        raw.equalities,
        raw.equality_rhs,
        variables,
        "linear equalities",
        cooperation,
    )
    .await?;
    let has_linear = inequalities.0.m != 0 || equalities.0.m != 0;
    let lower = numeric::bound(raw.lower, variables, f64::NEG_INFINITY, "lb", cooperation).await?;
    let upper = numeric::bound(raw.upper, variables, f64::INFINITY, "ub", cooperation).await?;
    let has_bounds = lower.iter().chain(&upper).any(|value| value.is_finite());
    let integers = numeric::indices(raw.integers, variables, cooperation).await?;
    let mixed_integer = !integers.is_empty();
    let linear = super::linear::Data {
        inequality: inequalities,
        equality: equalities,
        lower,
        upper,
        integers,
    };
    let nonlinear = (!raw::unspecified(&raw.nonlinear)).then_some(raw.nonlinear);
    let rng = raw
        .rng
        .as_ref()
        .map(|value| {
            crate::builtins::stats::random::rng::restore_snapshot_value(
                value,
                cooperation.runtime(),
            )
        })
        .transpose()?;
    Ok(Problem {
        objective: raw.objective,
        nonlinear,
        variables,
        linear: Some(linear),
        rng,
        public,
        has_linear,
        has_bounds,
        mixed_integer,
    })
}
