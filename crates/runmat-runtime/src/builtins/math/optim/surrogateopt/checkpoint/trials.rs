//! One canonical restored point matrix and sparse constraint matrix. Trial
//! rows retain headers into these owners rather than copying point payloads.

use super::super::{
    history::{History, MeasurementSource, Point, Trial},
    numeric::DoubleArray,
    observation::{self, Form, Objective, ObjectiveKind},
    queue::{Origin, PhaseId},
};
use super::error;
use crate::{
    builtins::math::optim::common::ScalarCooperation, execution::RootedValueSequence, BuiltinResult,
};
use runmat_value::{SparseTensor, Tensor, Value, ValueSequence};
use serde::Serializer;
use std::num::NonZeroUsize;

pub(super) struct Trials {
    variables: usize,
    objective: bool,
    inequalities: usize,
    x: Vec<f64>,
    fval: Vec<f64>,
    colptr: Vec<usize>,
    rowval: Vec<usize>,
    values: Vec<f64>,
    metadata: Vec<Metadata>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    phase: u64,
    #[serde(with = "origin")]
    origin: Origin,
    source: MeasurementSource,
}

impl Trials {
    pub(super) async fn read(
        reader: &mut super::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        reader.token(b'{')?;
        reader.key("variables", true)?;
        let variables = reader.metadata(20, c).await?;
        reader.key("objective", false)?;
        let objective = reader.metadata(5, c).await?;
        reader.key("inequalities", false)?;
        let inequalities = reader.metadata(20, c).await?;
        reader.key("x", false)?;
        let x = reader.bits(c).await?;
        reader.key("fval", false)?;
        let fval = reader.bits(c).await?;
        reader.key("colptr", false)?;
        let colptr = reader.array(20, c).await?;
        reader.key("rowval", false)?;
        let rowval = reader.array(20, c).await?;
        reader.key("values", false)?;
        let values = reader.bits(c).await?;
        reader.key("metadata", false)?;
        let metadata = reader.array(512, c).await?;
        reader.token(b'}')?;
        Ok(Self {
            variables,
            objective,
            inequalities,
            x,
            fval,
            colptr,
            rowval,
            values,
            metadata,
        })
    }
    pub(super) async fn decode(
        self,
        variables: usize,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<Trial>> {
        let rows = self.metadata.len();
        if self.variables != variables
            || variables == 0
            || rows.checked_mul(variables) != Some(self.x.len())
            || self.fval.len() != if self.objective { rows } else { 0 }
            || self.inequalities.checked_add(1) != Some(self.colptr.len())
            || self.colptr.first() != Some(&0)
            || self.colptr.last() != Some(&self.values.len())
            || self.rowval.len() != self.values.len()
            || self.x.iter().any(|value| !value.is_finite())
        {
            return Err(error(
                "checkpoint trial storage has an invalid original extent",
            ));
        }
        for column in 0..self.inequalities {
            let first = self.colptr[column];
            let end = self.colptr[column + 1];
            if first > end
                || end > self.values.len()
                || self.rowval[first..end].iter().any(|&row| row >= rows)
                || self.rowval[first..end]
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
            {
                return Err(error("checkpoint constraint column is invalid"));
            }
            c.completed((end - first) as u64 + 1).await?;
        }
        if rows == 0 {
            if self.objective || self.inequalities != 0 {
                return Err(error("empty checkpoint trials have a measured schema"));
            }
            return Ok(Vec::new());
        }
        let (points, returned) = {
            let _entry = c.runtime().map(crate::context::RuntimeContext::enter);
            let points = DoubleArray::owned_point(
                Tensor::new(self.x, vec![rows, variables])
                    .map_err(|detail| error(format!("invalid restored point storage: {detail}")))?,
            )?;
            let fval =
                if self.objective {
                    Value::Tensor(Tensor::new(self.fval, vec![rows, 1]).map_err(|detail| {
                        error(format!("invalid restored Fval storage: {detail}"))
                    })?)
                } else {
                    super::super::public_values::empty(c)?
                };
            let ineq = Value::SparseTensor(
                SparseTensor::new(
                    rows,
                    self.inequalities,
                    self.colptr,
                    self.rowval,
                    self.values,
                )
                .map_err(|detail| error(format!("invalid restored Ineq storage: {detail}")))?,
            );
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            fields.insert("Fval", fval);
            fields.insert("Ineq", ineq);
            let sequence = ValueSequence::single(Value::Struct(fields.finish()))
                .map_err(crate::sequence::sequence_error_to_runtime)?;
            let runtime = c
                .runtime()
                .ok_or_else(|| error("trial restoration requires its original runtime"))?;
            (points, RootedValueSequence::in_context(sequence, runtime)?)
        };
        let observations = observation::decode(
            returned,
            Form::Initial(NonZeroUsize::new(rows).expect("nonempty restored rows")),
            tolerance,
            c,
        )
        .await?;
        if observations.len() != rows {
            return Err(error("restored observations changed row extent"));
        }
        let mut trials = super::super::public_values::capacity(rows)?;
        for (row, (observation, metadata)) in
            observations.into_iter().zip(self.metadata).enumerate()
        {
            trials.push(Trial {
                point: Point::batch_row(points.clone(), row, rows, variables)?,
                observation,
                phase: PhaseId(metadata.phase),
                origin: metadata.origin,
                source: metadata.source,
            });
            c.completed(1).await?;
        }
        Ok(trials)
    }
}

pub(super) struct Borrowed<'a>(pub &'a History);
impl Borrowed<'_> {
    pub(super) async fn write(
        &self,
        writer: &mut super::json::Writer,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<()> {
        let schema = self.0.schema();
        let columns = schema.map_or(0, |value| value.inequalities);
        writer.token(b"{")?;
        writer.key("variables", true)?;
        writer.metadata(&self.0.variables(), c).await?;
        writer.key("objective", false)?;
        writer
            .metadata(
                &schema.is_some_and(|value| value.objective == ObjectiveKind::Present),
                c,
            )
            .await?;
        writer.key("inequalities", false)?;
        writer.metadata(&columns, c).await?;
        writer.key("x", false)?;
        writer.token(b"[")?;
        let mut first = true;
        for column in 0..self.0.variables() {
            for trial in self.0.iter() {
                if !first {
                    writer.token(b",")?;
                }
                first = false;
                writer
                    .metadata(&trial.point.coordinate(column).to_bits(), c)
                    .await?;
            }
        }
        writer.token(b"]")?;
        writer.key("fval", false)?;
        writer.token(b"[")?;
        let mut first = true;
        for trial in self.0.iter() {
            if let Objective::Measured(value) = trial.observation.objective() {
                if !first {
                    writer.token(b",")?;
                }
                first = false;
                writer.metadata(&value.to_bits(), c).await?;
            }
        }
        writer.token(b"]")?;
        for role in [Role::Pointers, Role::Rows, Role::Inequalities] {
            writer.key(
                match role {
                    Role::Pointers => "colptr",
                    Role::Rows => "rowval",
                    _ => "values",
                },
                false,
            )?;
            writer.token(b"[")?;
            let mut count = 0usize;
            let mut first = true;
            if matches!(role, Role::Pointers) {
                writer.metadata(&count, c).await?;
                first = false;
            }
            for column in 0..columns {
                for (row, trial) in self.0.iter().enumerate() {
                    let value = trial.observation.inequality(column);
                    // Preserve -0 and all nonfinite measured payload bits.
                    if value.to_bits() == 0 {
                        c.completed(1).await?;
                        continue;
                    }
                    if !matches!(role, Role::Pointers) {
                        if !first {
                            writer.token(b",")?;
                        }
                        first = false;
                        match role {
                            Role::Rows => writer.metadata(&row, c).await?,
                            _ => writer.metadata(&value.to_bits(), c).await?,
                        }
                    }
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| error("checkpoint sparse extent overflow"))?;
                    c.completed(1).await?;
                }
                if matches!(role, Role::Pointers) {
                    writer.token(b",")?;
                    writer.metadata(&count, c).await?;
                }
            }
            writer.token(b"]")?;
        }
        writer.key("metadata", false)?;
        writer.token(b"[")?;
        for (row, trial) in self.0.iter().enumerate() {
            if row != 0 {
                writer.token(b",")?;
            }
            writer
                .metadata(
                    &Metadata {
                        phase: trial.phase.0,
                        origin: trial.origin,
                        source: trial.source,
                    },
                    c,
                )
                .await?;
        }
        writer.token(b"]}")
    }
}
enum Role {
    Pointers,
    Rows,
    Inequalities,
}

mod origin {
    use super::*;
    use serde::Deserialize;
    pub(super) fn serialize<S: Serializer>(
        value: &Origin,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value.name())
    }
    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Origin, D::Error> {
        let name = String::deserialize(deserializer)?;
        Origin::parse(&name)
            .ok_or_else(|| serde::de::Error::custom("unknown measured point origin"))
    }
}
