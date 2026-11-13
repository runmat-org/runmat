//! Only the original linear geometry crosses a feasible-start worker call;
//! unused fitness callbacks, scores and observer state are not captured.
use super::super::{
    common::constrained_numeric::Geometry, inputs::raw, invalid, numeric, ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::{StructValue, Value};

#[derive(Clone, Copy)]
enum Field {
    A,
    B,
    Aeq,
    Beq,
    Lower,
    Upper,
}
impl Field {
    const ALL: [Self; 6] = [
        Self::A,
        Self::B,
        Self::Aeq,
        Self::Beq,
        Self::Lower,
        Self::Upper,
    ];
    const fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "b",
            Self::Aeq => "Aeq",
            Self::Beq => "beq",
            Self::Lower => "lb",
            Self::Upper => "ub",
        }
    }
    const fn public(self) -> &'static str {
        match self {
            Self::A => "Aineq",
            Self::B => "Bineq",
            Self::Aeq => "Aeq",
            Self::Beq => "Beq",
            Self::Lower => "lb",
            Self::Upper => "ub",
        }
    }
    const fn geometry(self) -> Geometry {
        match self {
            Self::A | Self::Aeq => Geometry::MatrixAny,
            _ => Geometry::Vector,
        }
    }
}
pub(super) fn from_problem(problem: &Value) -> BuiltinResult<Value> {
    let Value::Struct(problem) = problem else {
        return Err(invalid("creation lost its problem structure"));
    };
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    for field in Field::ALL {
        let value = problem
            .fields
            .get(field.public())
            .ok_or_else(|| invalid("creation problem is missing original geometry"))?;
        result.insert(field.name(), value.clone());
    }
    Ok(Value::Struct(result.finish()))
}
pub(super) struct Original<'a>(&'a StructValue);
impl<'a> Original<'a> {
    pub fn preflight(value: &'a Value, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        let Value::Struct(value) = value else {
            return Err(invalid("worker geometry must be a scalar structure"));
        };
        if value.fields.len() != Field::ALL.len() {
            return Err(invalid("worker geometry has unknown or missing fields"));
        }
        for field in Field::ALL {
            let value = value
                .fields
                .get(field.name())
                .ok_or_else(|| invalid("worker geometry is missing a field"))?;
            numeric::preflight(value, field.name(), field.geometry(), cooperation.runtime())?;
        }
        Ok(Self(value))
    }
    pub async fn materialize(
        self,
        variables: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<super::super::super::fmincon::feasibility::Input> {
        let value = |field: Field| {
            self.0
                .fields
                .get(field.name())
                .cloned()
                .expect("preflighted geometry field")
        };
        let inequality = numeric::pair(
            value(Field::A),
            value(Field::B),
            variables,
            "linear inequalities",
            cooperation,
        )
        .await?;
        let equality = numeric::pair(
            value(Field::Aeq),
            value(Field::Beq),
            variables,
            "linear equalities",
            cooperation,
        )
        .await?;
        let lower = numeric::bound(
            value(Field::Lower),
            variables,
            f64::NEG_INFINITY,
            "lb",
            cooperation,
        )
        .await?;
        let upper = numeric::bound(
            value(Field::Upper),
            variables,
            f64::INFINITY,
            "ub",
            cooperation,
        )
        .await?;
        Ok(super::super::super::fmincon::feasibility::Input {
            inequality,
            equality,
            lower,
            upper,
        })
    }
}

pub(super) fn callback(value: &Value) -> BuiltinResult<()> {
    if raw::callable(value) {
        Ok(())
    } else {
        Err(invalid("worker nonlcon must be one selected callable"))
    }
}
