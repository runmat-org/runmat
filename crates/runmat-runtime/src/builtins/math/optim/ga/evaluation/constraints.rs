//! Retain actual nonlinear callback matrices independently from fitness/rank.
use super::super::{common, invalid, numeric::INPUT_POLICY, ScalarCooperation, POLICY};
use crate::BuiltinResult;
use common::constrained_numeric as numeric;
use runmat_value::{NumericDType, Tensor, Value};

pub(in super::super) struct Constraints {
    pub inequalities: Tensor,
    pub equalities: Tensor,
    pub rows: usize,
    pub inequality_count: usize,
    pub equality_count: usize,
}

impl Constraints {
    pub async fn reorder(
        &self,
        positions: &[usize],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut identity = positions.len() == self.rows;
        for (index, &row) in positions.iter().enumerate() {
            if row >= self.rows {
                return Err(invalid("constraint duplicate map exceeds measured rows"));
            }
            identity &= index == row;
            cooperation.completed(1).await?;
        }
        if identity {
            return Ok(Self {
                inequalities: self.inequalities.clone(),
                equalities: self.equalities.clone(),
                rows: self.rows,
                inequality_count: self.inequality_count,
                equality_count: self.equality_count,
            });
        }
        let count = positions.len();
        let mut inequalities = POLICY.capacity(
            count
                .checked_mul(self.inequality_count)
                .ok_or_else(|| invalid("nonlinear matrix size overflow"))?,
            cooperation.runtime(),
        )?;
        let mut equalities = POLICY.capacity(
            count
                .checked_mul(self.equality_count)
                .ok_or_else(|| invalid("nonlinear matrix size overflow"))?,
            cooperation.runtime(),
        )?;
        for column in 0..self.inequality_count {
            for &row in positions {
                inequalities.push(self.inequality(row, column));
                cooperation.completed(1).await?;
            }
        }
        for column in 0..self.equality_count {
            for &row in positions {
                equalities.push(self.equality(row, column));
                cooperation.completed(1).await?;
            }
        }
        Ok(Self {
            inequalities: Tensor::new(inequalities, vec![count, self.inequality_count])
                .map_err(invalid)?,
            equalities: Tensor::new(equalities, vec![count, self.equality_count])
                .map_err(invalid)?,
            rows: count,
            inequality_count: self.inequality_count,
            equality_count: self.equality_count,
        })
    }
    pub fn extents(&self) -> (usize, usize) {
        (self.inequality_count, self.equality_count)
    }
    pub fn inequality(&self, row: usize, column: usize) -> f64 {
        self.inequalities
            .as_f64_slice()
            .expect("admitted real constraints")[row + self.rows * column]
    }
    pub fn equality(&self, row: usize, column: usize) -> f64 {
        self.equalities
            .as_f64_slice()
            .expect("admitted real constraints")[row + self.rows * column]
    }
    pub async fn row_violation(
        &self,
        row: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<(f64, f64)>> {
        if row >= self.rows {
            return Err(invalid("nonlinear row index outside population"));
        }
        let mut maximum = 0.0_f64;
        let mut sum = 0.0;
        for column in 0..self.inequality_count {
            let value = self.inequality(row, column);
            if !value.is_finite() {
                return Ok(None);
            }
            let violation = value.max(0.0);
            maximum = maximum.max(violation);
            sum += violation;
            cooperation.completed(1).await?;
        }
        for column in 0..self.equality_count {
            let value = self.equality(row, column);
            if !value.is_finite() {
                return Ok(None);
            }
            maximum = maximum.max(value.abs());
            sum += value.abs();
            cooperation.completed(1).await?;
        }
        Ok(sum.is_finite().then_some((maximum, sum)))
    }
    pub fn empty(rows: usize) -> BuiltinResult<Self> {
        Ok(Self {
            inequalities: Tensor::new(Vec::new(), vec![rows, 0]).map_err(invalid)?,
            equalities: Tensor::new(Vec::new(), vec![rows, 0]).map_err(invalid)?,
            rows,
            inequality_count: 0,
            equality_count: 0,
        })
    }
}

fn columns(value: &Value, rows: usize) -> BuiltinResult<usize> {
    let shape = numeric::shape(value)
        .ok_or_else(|| invalid("nonlcon must return two real numeric arrays"))?;
    let length = shape
        .iter()
        .try_fold(1usize, |count, &axis| count.checked_mul(axis))
        .ok_or_else(|| invalid("nonlinear output shape overflow"))?;
    if length == 0 {
        return Ok(0);
    }
    if rows == 1 {
        if shape.len() <= 2
            && (shape.first().copied().unwrap_or(1) == 1 || shape.get(1).copied().unwrap_or(1) == 1)
        {
            return Ok(length);
        }
    } else if shape.len() <= 2 && shape.first() == Some(&rows) {
        return Ok(shape.get(1).copied().unwrap_or(1));
    }
    Err(invalid("vectorized nonlcon outputs require one row per submitted population member; scalar calls require vectors"))
}

async fn matrix(
    value: Value,
    rows: usize,
    columns: usize,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Tensor> {
    let Some(value) = numeric::prepare(
        value,
        role,
        numeric::Geometry::Array,
        &INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid("nonlcon outputs must be real numeric arrays"));
    };
    // A full vectorized double result retains its exact decoded storage. A
    // column-to-row scalar-call reshape changes metadata, not numeric backing.
    let tensor = match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => tensor,
        value => {
            let numeric::Numeric::Real { values, .. } =
                numeric::materialize_raw(value, role, &INPUT_POLICY, cooperation).await?
            else {
                return Err(invalid("nonlcon outputs must be real numeric arrays"));
            };
            Tensor::new(values, vec![rows, columns]).map_err(invalid)?
        }
    };
    if rows.checked_mul(columns) != Some(tensor.len()) {
        return Err(invalid(
            "nonlcon changed output geometry during materialization",
        ));
    }
    tensor.reshape(vec![rows, columns]).map_err(invalid)
}

pub(in super::super) async fn decode(
    returned: crate::execution::RootedValueSequence,
    rows: usize,
    expected: Option<(usize, usize)>,
    mixed_integer: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Constraints> {
    let returned = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.resolve_sequence(
            runmat_types::SequenceUse::from_requested_outputs(
                runmat_types::RequestedOutputCount::Exactly(2),
            ),
            crate::sequence::SequenceResolutionContext::default(),
        )?
    };
    if returned.as_slice().len() != 2 {
        return Err(invalid("nonlcon must return both c and ceq"));
    }
    let extents = (
        columns(&returned.as_slice()[0], rows)?,
        columns(&returned.as_slice()[1], rows)?,
    );
    if expected.is_some_and(|expected| expected != extents) {
        return Err(invalid("nonlcon changed its inequality or equality count"));
    }
    if mixed_integer && extents.1 != 0 {
        return Err(invalid(
            "integer-constrained ga does not support nonlinear equality constraints",
        ));
    }
    // Both output roles are admitted before the first provider transfer. A
    // supplied fitness/penalty score is never nonlinear feasibility evidence.
    for (value, role) in returned
        .as_slice()
        .iter()
        .zip(["nonlinear inequalities", "nonlinear equalities"])
    {
        super::super::numeric::preflight(
            value,
            role,
            numeric::Geometry::Array,
            cooperation.runtime(),
        )?;
    }
    let mut values = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_values()?
    };
    let inequalities = matrix(
        values.next().expect("validated c"),
        rows,
        extents.0,
        "nonlinear inequalities",
        cooperation,
    )
    .await?;
    let equalities = matrix(
        values.next().expect("validated ceq"),
        rows,
        extents.1,
        "nonlinear equalities",
        cooperation,
    )
    .await?;
    drop(values);
    Ok(Constraints {
        inequalities,
        equalities,
        rows,
        inequality_count: extents.0,
        equality_count: extents.1,
    })
}

/// Assemble only actually measured row results. Full vectorized waves bypass
/// this projection and retain their original matrix backing directly.
pub(in super::super) async fn assemble(
    rows: Vec<Constraints>,
    positions: Option<&[usize]>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Constraints> {
    let Some(first) = rows.first() else {
        if positions.is_some_and(|positions| !positions.is_empty()) {
            return Err(invalid(
                "nonempty duplicate map has no measured constraints",
            ));
        }
        return Constraints::empty(0);
    };
    let extents = first.extents();
    if rows
        .iter()
        .any(|row| row.rows != 1 || row.extents() != extents)
    {
        return Err(invalid(
            "nonlinear row measurements have inconsistent extents",
        ));
    }
    if let Some(positions) = positions {
        for &row in positions {
            if row >= rows.len() {
                return Err(invalid("constraint duplicate map exceeds measured rows"));
            }
            cooperation.completed(1).await?;
        }
    }
    let count = positions.map_or(rows.len(), |positions| positions.len());
    let mut inequalities = POLICY.capacity(
        count
            .checked_mul(extents.0)
            .ok_or_else(|| invalid("nonlinear matrix size overflow"))?,
        cooperation.runtime(),
    )?;
    let mut equalities = POLICY.capacity(
        count
            .checked_mul(extents.1)
            .ok_or_else(|| invalid("nonlinear matrix size overflow"))?,
        cooperation.runtime(),
    )?;
    for column in 0..extents.0 {
        for index in 0..count {
            let row = &rows[positions.map_or(index, |positions| positions[index])];
            inequalities.push(row.inequality(0, column));
            cooperation.completed(1).await?;
        }
    }
    for column in 0..extents.1 {
        for index in 0..count {
            let row = &rows[positions.map_or(index, |positions| positions[index])];
            equalities.push(row.equality(0, column));
            cooperation.completed(1).await?;
        }
    }
    Ok(Constraints {
        inequalities: Tensor::new(inequalities, vec![count, extents.0]).map_err(invalid)?,
        equalities: Tensor::new(equalities, vec![count, extents.1]).map_err(invalid)?,
        rows: count,
        inequality_count: extents.0,
        equality_count: extents.1,
    })
}
