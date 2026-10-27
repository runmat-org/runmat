//! Complete positional/problem grammar; no recursive callback facts.
use super::super::{model, numeric, parse};
use super::{bernstein::invalid, oracle::admit_double};
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::Value;
pub(super) struct Problem {
    pub objective: Value,
    pub seminfcon: Value,
    pub families: usize,
    pub x: Vec<f64>,
    pub shape: Vec<usize>,
    pub scalar: bool,
    pub geometry: model::Geometry,
    pub inconsistent_bounds: bool,
    pub options: Value,
}
pub(super) async fn parse(
    args: crate::call::BuiltinArguments,
    policy: &numeric::Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    let mut values = if args.len() == 1 {
        let Value::Struct(p) = args.into_iter().next().unwrap() else {
            return Err(invalid(
                "single argument form requires a scalar problem structure",
            ));
        };
        for field in [
            "objective",
            "x0",
            "ntheta",
            "seminfcon",
            "solver",
            "options",
        ] {
            if !p.fields.contains_key(field) {
                return Err(invalid(format!("problem requires {field}")));
            }
        }
        if !common::field_name(&p.fields["solver"])?.eq_ignore_ascii_case("fseminf") {
            return Err(invalid("problem solver must be fseminf"));
        }
        [
            "objective",
            "x0",
            "ntheta",
            "seminfcon",
            "Aineq",
            "bineq",
            "Aeq",
            "beq",
            "lb",
            "ub",
            "options",
        ]
        .into_iter()
        .map(|key| {
            p.fields
                .get(key)
                .cloned()
                .unwrap_or_else(super::super::empty)
        })
        .collect::<Vec<_>>()
    } else {
        if !matches!(args.len(), 4 | 6 | 8 | 10 | 11) {
            return Err(invalid("expected fun,x0,ntheta,seminfcon with complete A/b, Aeq/beq and lb/ub pairs and optional options"));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(11, super::super::empty);
        values
    };
    if !parse::callable(&values[0]) || !parse::callable(&values[3]) {
        return Err(invalid(
            "fun and seminfcon must be single function handles or function names",
        ));
    }
    // All documented numeric inputs are host double. Do not traverse a denied
    // struct/object/callback carrier or transfer any provider before admission.
    for (index, role, geometry) in [
        (1, "x0", numeric::Geometry::Array),
        (2, "ntheta", numeric::Geometry::Scalar),
        (4, "A", numeric::Geometry::MatrixAny),
        (5, "b", numeric::Geometry::Vector),
        (6, "Aeq", numeric::Geometry::MatrixAny),
        (7, "beq", numeric::Geometry::Vector),
        (8, "lb", numeric::Geometry::Array),
        (9, "ub", numeric::Geometry::Array),
    ] {
        admit_double(&values[index])?;
        if !numeric::preflight(
            &values[index],
            role,
            geometry,
            policy,
            cooperation.runtime(),
        )? {
            return Err(invalid(format!("{role} must be real")));
        }
    }
    let families = common::option_usize_value("fseminf", Some(&values[2]), "ntheta", 0)?;
    if families == 0 {
        return Err(invalid("ntheta must be a positive integer"));
    }
    // Checked shape, including the ntheta+3 callback tuple cardinality.
    families
        .checked_mul(2)
        .and_then(|_| families.checked_add(3))
        .ok_or_else(|| invalid("ntheta extent overflow"))?;
    let options = values.pop().unwrap();
    let mut values = values.into_iter();
    let objective = values.next().unwrap();
    let x0 = values.next().unwrap();
    values.next();
    let seminfcon = values.next().unwrap();
    let scalar = matches!(&x0, Value::Num(_));
    let numeric::Numeric::Real { shape, values: x } =
        numeric::read(x0, "x0", numeric::Geometry::Array, policy, cooperation).await?
    else {
        return Err(invalid("x0 must be finite real double"));
    };
    if x.is_empty() {
        return Err(invalid("x0 must be nonempty"));
    }
    let inequality = parse::pair(
        values.next().unwrap(),
        values.next().unwrap(),
        x.len(),
        "A/b",
        policy,
        cooperation,
    )
    .await?;
    let equality = parse::pair(
        values.next().unwrap(),
        values.next().unwrap(),
        x.len(),
        "Aeq/beq",
        policy,
        cooperation,
    )
    .await?;
    let lower = parse::bound(
        values.next().unwrap(),
        x.len(),
        f64::NEG_INFINITY,
        "lb",
        policy,
        cooperation,
    )
    .await?;
    let upper = parse::bound(
        values.next().unwrap(),
        x.len(),
        f64::INFINITY,
        "ub",
        policy,
        cooperation,
    )
    .await?;
    let mut inconsistent_bounds = false;
    for (&l, &u) in lower.iter().zip(&upper) {
        inconsistent_bounds |= l > u || l == f64::INFINITY || u == f64::NEG_INFINITY;
        cooperation.completed(1).await?;
    }
    Ok(Problem {
        objective,
        seminfcon,
        families,
        x,
        shape,
        scalar,
        geometry: model::Geometry {
            variables: lower.len(),
            inequality,
            equality,
            lower,
            upper,
        },
        inconsistent_bounds,
        options,
    })
}
