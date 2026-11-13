//! Supplied score classes cross the same GA numeric/compatibility boundary;
//! their conversion does not manufacture an objective callback observation.
use super::{
    common::{self, constrained_numeric as numeric, population::Fitness},
    invalid, ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticStateField;
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueKindFact};
use runmat_value::{NumericDType, Tensor, Value};
#[derive(Clone, Copy)]
pub(super) enum Role {
    Initial,
    Score,
    Fitness,
}
impl Role {
    fn name(self) -> &'static str {
        match self {
            Self::Initial => "InitialScoresMatrix",
            Self::Score => OptimGeneticStateField::Score.name(),
            Self::Fitness => OptimGeneticStateField::Fitness.name(),
        }
    }
}
pub(super) async fn decode(
    value: Value,
    role: Role,
    count: usize,
    check: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Fitness> {
    let fact = crate::value_fact::value_fact(&value);
    let complex = matches!(fact.kind, ValueKindFact::Numeric(number) if number.class == NumericClass::Double && number.domain == NumericDomain::Complex);
    if complex && !matches!(role, Role::Initial) {
        common::preflight_host_residency("ga", &value, None, cooperation.runtime())?;
        return Fitness::decode(POLICY, value, count, check, cooperation).await;
    }
    let policy = &super::numeric::INPUT_POLICY;
    let value = numeric::prepare(
        value,
        role.name(),
        numeric::Geometry::Vector,
        policy,
        cooperation,
    )
    .await?
    .ok_or_else(|| invalid("supplied fitness must be real numeric data"))?;
    let value = match value {
        Value::Num(_) => value,
        Value::Tensor(ref tensor) if tensor.numeric_dtype() == NumericDType::F64 => value,
        value => {
            let numeric::Numeric::Real { values, .. } =
                numeric::materialize_raw(value, role.name(), policy, cooperation).await?
            else {
                return Err(invalid("supplied fitness must be real numeric data"));
            };
            Value::Tensor(Tensor::new(values, vec![count, 1]).map_err(invalid)?)
        }
    };
    Fitness::decode(POLICY, value, count, check, cooperation).await
}

pub(super) fn metadata(
    value: &Value,
    rows: Option<usize>,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    let fact = crate::value_fact::value_fact(value);
    let complex_double = matches!(fact.kind, ValueKindFact::Numeric(number) if number.class == NumericClass::Double && number.domain == NumericDomain::Complex);
    if complex_double {
        common::preflight_host_residency("ga", value, None, cooperation.runtime())?;
    } else {
        super::numeric::preflight(
            value,
            "state score",
            common::constrained_numeric::Geometry::Vector,
            cooperation.runtime(),
        )?;
    }
    if fact.storage == StorageFact::Sparse
        || rows.is_some_and(|rows| {
            fact.shape
                .element_count()
                .is_some_and(|count| count != rows)
        })
        || fact.shape.known_dims().is_some_and(|shape| {
            shape.len() > 2
                || shape
                    .iter()
                    .filter(|axis| axis.is_some_and(|count| count != 1))
                    .count()
                    > 1
        })
    {
        return Err(invalid(
            "state scores require one full numeric entry per population row",
        ));
    }
    Ok(())
}
