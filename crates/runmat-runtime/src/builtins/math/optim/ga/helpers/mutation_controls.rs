//! Only a standalone mutation's consumed option/state roles enter admission.
use super::{
    super::{common, invalid, options, ScalarCooperation, POLICY},
    admission,
};
use crate::BuiltinResult;
use runmat_builtins::{
    OptimGeneticMutation as Kind, OptimGeneticStateField as StateField, OptimOptionField as Field,
};
use runmat_value::{CellArray, Value};

pub(super) struct State<'a> {
    pub generation: Option<&'a Value>,
    pub best: Option<&'a Value>,
}
impl<'a> State<'a> {
    pub fn new(
        kind: Kind,
        value: &'a Value,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let Value::Struct(value) = value else {
            return Err(invalid("mutation state must be a scalar structure"));
        };
        let generation = if kind == Kind::Gaussian {
            let generation = value
                .fields
                .get(StateField::Generation.name())
                .ok_or_else(|| invalid("Gaussian mutation state requires Generation"))?;
            admission::count_metadata(generation, cooperation.runtime())?;
            Some(generation)
        } else {
            None
        };
        let best = if matches!(kind, Kind::AdaptiveFeasible | Kind::PositiveBasis) {
            let best = value
                .fields
                .get(StateField::Best.name())
                .ok_or_else(|| invalid("feasible mutation state requires Best history"))?;
            admission::vector_metadata(best, cooperation.runtime())?;
            Some(best)
        } else {
            None
        };
        Ok(Self { generation, best })
    }
}
pub(super) fn options(
    kind: Kind,
    value: &Value,
    members: usize,
    parameters: &[Value],
    cooperation: &ScalarCooperation,
) -> BuiltinResult<Value> {
    admission::options(value)?;
    let fields = common::ScalarOptionFields::from_struct_for_solver(
        if let Value::Struct(value) = value {
            Some(value)
        } else {
            None
        },
        runmat_builtins::OptimSolver::Ga,
    );
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    for field in [Field::PopulationType, Field::InitialPopulationRange] {
        if let Some(value) = fields.exact(field) {
            result.insert(field.name(), value.clone());
        }
    }
    if kind == Kind::Gaussian {
        if let Some(value) = fields.exact(Field::MaxIter) {
            result.insert(Field::MaxIter.name(), value.clone());
        }
    }
    if matches!(kind, Kind::AdaptiveFeasible | Kind::PositiveBasis) {
        if let Some(value) = fields.exact(Field::TolCon) {
            result.insert(Field::TolCon.name(), value.clone());
        }
    }
    let count = u64::try_from(members)
        .ok()
        .and_then(|value| runmat_value::IntValue::U64(value).to_exact_f64())
        .ok_or_else(|| invalid("population row count cannot be represented as an exact control"))?;
    result.insert(Field::PopulationSize.name(), Value::Num(count));
    let selector = if parameters.is_empty() {
        Value::from(kind.name())
    } else {
        let mut values = POLICY.capacity(
            parameters
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid("mutation parameter extent overflow"))?,
            cooperation.runtime(),
        )?;
        values.push(Value::from(kind.name()));
        values.extend(parameters.iter().cloned());
        let n = values.len();
        Value::Cell(CellArray::new(values, 1, n).map_err(invalid)?)
    };
    result.insert(Field::MutationFcn.name(), selector);
    Ok(Value::Struct(result.finish()))
}

pub(super) fn matrix_rows(value: &Value) -> BuiltinResult<usize> {
    let fact = crate::value_fact::value_fact(value);
    let dimensions = fact
        .shape
        .known_dims()
        .ok_or_else(|| invalid("mutation population requires known matrix geometry"))?;
    if dimensions.len() != 2 {
        return Err(invalid("mutation population must be a matrix"));
    }
    dimensions[0]
        .filter(|n| *n != 0)
        .ok_or_else(|| invalid("mutation population must have at least one row"))
}

pub(super) async fn restore(
    mutation: &mut super::super::reproduction::Mutation,
    state: State<'_>,
    options: &options::Options,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if let Some(generation) = state.generation {
        let completed = admission::count(generation, "Generation", cooperation).await?;
        let next = completed
            .checked_add(1)
            .ok_or_else(|| invalid("mutation generation overflow"))?;
        if next > options.generations {
            return Err(invalid(
                "Gaussian mutation requires an active generation below MaxGenerations",
            ));
        }
        for generation in 1..=next {
            if !mutation
                .begin(generation, options.generations, cooperation)
                .await?
            {
                return Err(invalid("Gaussian deviation cannot be represented"));
            }
            cooperation.completed(1).await?;
        }
    }
    if let Some(best) = state.best {
        let history = admission::vector(best.clone(), cooperation).await?;
        for pair in history.windows(2) {
            mutation.completed(pair[1].is_finite() && (!pair[0].is_finite() || pair[1] < pair[0]));
            cooperation.completed(1).await?;
        }
    }
    Ok(())
}
