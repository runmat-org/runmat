//! The immutable prepared InitialPoints template, not a trial-history copy.
use super::super::numeric::{ArchivedMatrix, BorrowedMatrix};
use super::*;

impl Initial {
    pub(in super::super) async fn same_template(
        &self,
        other: &Self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if self.rows != other.rows || self.variables != other.variables {
            return Ok(false);
        }
        for index in 0..self
            .rows
            .checked_mul(self.variables)
            .ok_or_else(|| invalid("initial template extent overflow"))?
        {
            if self.points.get(index).to_bits() != other.points.get(index).to_bits() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        fn values(measurements: &Measurements) -> (u8, &[Observation]) {
            match measurements {
                Measurements::Missing => (0, &[][..]),
                Measurements::Complete(values) => (1, values.as_slice()),
                Measurements::Partial {
                    supplied: Known::Objective,
                    values,
                } => (2, values.as_slice()),
                Measurements::Partial {
                    supplied: Known::Inequalities,
                    values,
                } => (3, values.as_slice()),
            }
        }
        let (left_role, left) = values(&self.measurements);
        let (right_role, right) = values(&other.measurements);
        if left_role != right_role || left.len() != right.len() {
            return Ok(false);
        }
        for (left, right) in left.iter().zip(right) {
            if left.schema() != right.schema() {
                return Ok(false);
            }
            match (left.objective(), right.objective()) {
                (observation::Objective::Absent, observation::Objective::Absent) => {}
                (
                    observation::Objective::Measured(left),
                    observation::Objective::Measured(right),
                ) if left.to_bits() == right.to_bits() => {}
                _ => return Ok(false),
            }
            for column in 0..left.schema().inequalities {
                if left.inequality(column).to_bits() != right.inequality(column).to_bits() {
                    return Ok(false);
                }
                c.completed(1).await?;
            }
        }
        Ok(true)
    }
}

pub(in super::super) struct ArchivedInitial {
    x: ArchivedMatrix,
    fval: Lane,
    ineq: Lane,
}
enum Lane {
    Missing,
    Empty,
    Array(ArchivedMatrix),
}
impl ArchivedInitial {
    pub(in super::super) async fn read(
        reader: &mut super::super::checkpoint::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        reader.token(b'{')?;
        reader.key("x", true)?;
        let x = ArchivedMatrix::read(reader, c).await?;
        reader.key("fval", false)?;
        let fval = Lane::read(reader, c).await?;
        reader.key("ineq", false)?;
        let ineq = Lane::read(reader, c).await?;
        reader.token(b'}')?;
        Ok(Self { x, fval, ineq })
    }
    pub(in super::super) async fn prepare(
        self,
        variables: usize,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Initial> {
        let value = {
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            fields.insert("X", self.x.into_value()?);
            for (name, value) in [("Fval", self.fval), ("Ineq", self.ineq)] {
                if !matches!(value, Lane::Missing) {
                    fields.insert(
                        name,
                        match value {
                            Lane::Array(value) => value.into_value()?,
                            Lane::Empty => public_values::empty(c)?,
                            Lane::Missing => unreachable!("checked supplied lane"),
                        },
                    );
                }
            }
            Value::Struct(fields.finish())
        };
        Initial::prepare(value, variables, tolerance, c).await
    }
}
impl Lane {
    async fn read(
        reader: &mut super::super::checkpoint::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        if reader.is(b'{') {
            reader.token(b'{')?;
            reader.key("Array", true)?;
            let value = ArchivedMatrix::read(reader, c).await?;
            reader.token(b'}')?;
            Ok(Self::Array(value))
        } else {
            match reader.metadata::<String>(9, c).await?.as_str() {
                "Missing" => Ok(Self::Missing),
                "Empty" => Ok(Self::Empty),
                _ => Err(invalid("invalid initial measurement lane in checkpoint")),
            }
        }
    }
}
pub(in super::super) struct BorrowedInitial<'a>(pub &'a Initial);
impl BorrowedInitial<'_> {
    pub(in super::super) async fn write(
        &self,
        writer: &mut super::super::checkpoint::json::Writer,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<()> {
        let first = match &self.0.measurements {
            Measurements::Missing => None,
            Measurements::Complete(values) | Measurements::Partial { values, .. } => values.first(),
        };
        let (objective, constraints) = first
            .map(Observation::initial_headers)
            .transpose()?
            .unwrap_or((None, None));
        let (has_objective, has_constraints) = match &self.0.measurements {
            Measurements::Missing => (false, false),
            Measurements::Complete(_) => (true, true),
            Measurements::Partial {
                supplied: Known::Objective,
                ..
            } => (true, false),
            Measurements::Partial {
                supplied: Known::Inequalities,
                ..
            } => (false, true),
        };
        writer.token(b"{")?;
        writer.key("x", true)?;
        BorrowedMatrix(&self.0.points).write(writer, c).await?;
        for (key, present, value) in [
            ("fval", has_objective, objective),
            ("ineq", has_constraints, constraints),
        ] {
            writer.key(key, false)?;
            if !present {
                writer.metadata("Missing", c).await?;
            } else if let Some(value) = value {
                writer.token(b"{")?;
                writer.key("Array", true)?;
                BorrowedMatrix(value).write(writer, c).await?;
                writer.token(b"}")?;
            } else {
                writer.metadata("Empty", c).await?;
            }
        }
        writer.token(b"}")
    }
}
