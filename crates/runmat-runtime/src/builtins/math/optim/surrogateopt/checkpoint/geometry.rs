//! The original linear model is serialized through its checked borrowed
//! prefix view, never through the MIP distance lift or copied sparse matrix.

use crate::builtins::math::optim::intlinprog::feasibility::{Domain, Input};

pub(super) struct Geometry {
    variables: usize,
    rows: usize,
    equalities: usize,
    colptr: Vec<usize>,
    rowval: Vec<usize>,
    coefficients: Vec<f64>,
    rhs: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    integers: Vec<usize>,
}

impl Geometry {
    pub(super) async fn read(
        reader: &mut super::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> crate::BuiltinResult<Self> {
        reader.token(b'{')?;
        reader.key("variables", true)?;
        let variables = reader.metadata(20, c).await?;
        reader.key("rows", false)?;
        let rows = reader.metadata(20, c).await?;
        reader.key("equalities", false)?;
        let equalities = reader.metadata(20, c).await?;
        reader.key("colptr", false)?;
        let colptr = reader.array(20, c).await?;
        reader.key("rowval", false)?;
        let rowval = reader.array(20, c).await?;
        reader.key("coefficients", false)?;
        let coefficients = reader.bits(c).await?;
        reader.key("rhs", false)?;
        let rhs = reader.bits(c).await?;
        reader.key("lower", false)?;
        let lower = reader.bits(c).await?;
        reader.key("upper", false)?;
        let upper = reader.bits(c).await?;
        reader.key("integers", false)?;
        let integers = reader.array(20, c).await?;
        reader.token(b'}')?;
        Ok(Self {
            variables,
            rows,
            equalities,
            colptr,
            rowval,
            coefficients,
            rhs,
            lower,
            upper,
            integers,
        })
    }
    pub(super) fn into_input(self) -> crate::BuiltinResult<Input> {
        if self.variables == 0
            || self.lower.len() != self.variables
            || self.upper.len() != self.variables
            || self.rhs.len() != self.rows
            || self.equalities > self.rows
            || self.colptr.len() != self.variables.checked_add(1).unwrap_or(0)
            || self.colptr.first() != Some(&0)
            || self.colptr.last() != Some(&self.coefficients.len())
            || self.rowval.len() != self.coefficients.len()
            || self
                .lower
                .iter()
                .chain(&self.upper)
                .any(|value| !value.is_finite())
            || self
                .rhs
                .iter()
                .chain(&self.coefficients)
                .any(|value| !value.is_finite())
        {
            return Err(super::error(
                "checkpoint original geometry has an invalid extent or value",
            ));
        }
        for column in 0..self.variables {
            let first = self.colptr[column];
            let end = self.colptr[column + 1];
            if first > end
                || end > self.coefficients.len()
                || self.rowval[first..end].iter().any(|&row| row >= self.rows)
                || self.rowval[first..end]
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
            {
                return Err(super::error("checkpoint original CSC column is invalid"));
            }
        }
        if self.integers.iter().any(|&column| column >= self.variables)
            || self.integers.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(super::error("checkpoint integer columns are invalid"));
        }
        Ok(Input {
            matrix: clarabel::algebra::CscMatrix::new(
                self.rows,
                self.variables,
                self.colptr,
                self.rowval,
                self.coefficients,
            ),
            rhs: self.rhs,
            equalities: self.equalities,
            lower: self.lower,
            upper: self.upper,
            integers: self.integers,
        })
    }
}

pub(super) struct Borrowed<'a>(pub &'a Domain);
impl Borrowed<'_> {
    pub(super) async fn write(
        &self,
        writer: &mut super::json::Writer,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> crate::BuiltinResult<()> {
        let linear = self.0.original_linear();
        writer.token(b"{")?;
        writer.key("variables", true)?;
        writer.metadata(&linear.variables(), c).await?;
        writer.key("rows", false)?;
        writer.metadata(&linear.rows(), c).await?;
        writer.key("equalities", false)?;
        writer.metadata(&linear.equalities(), c).await?;
        writer.key("colptr", false)?;
        writer.token(b"[0")?;
        let mut count = 0usize;
        for column in 0..linear.variables() {
            let values = linear
                .column(column)
                .ok_or_else(|| super::error("original checkpoint column is absent"))?;
            for _ in values {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| super::error("checkpoint column extent overflow"))?;
                c.completed(1).await?;
            }
            writer.token(b",")?;
            writer.metadata(&count, c).await?;
        }
        writer.token(b"]")?;
        for (key, coefficient) in [("rowval", false), ("coefficients", true)] {
            writer.key(key, false)?;
            writer.token(b"[")?;
            let mut first = true;
            for column in 0..linear.variables() {
                for (row, value) in linear
                    .column(column)
                    .ok_or_else(|| super::error("original checkpoint column is absent"))?
                {
                    if !first {
                        writer.token(b",")?;
                    }
                    first = false;
                    if coefficient {
                        writer.metadata(&value.to_bits(), c).await?;
                    } else {
                        writer.metadata(&row, c).await?;
                    }
                }
            }
            writer.token(b"]")?;
        }
        for (key, values) in [
            ("rhs", linear.rhs()),
            ("lower", self.0.bounds().0),
            ("upper", self.0.bounds().1),
        ] {
            writer.key(key, false)?;
            writer
                .array(values.iter().map(|value| value.to_bits()), c)
                .await?;
        }
        writer.key("integers", false)?;
        writer.array(self.0.integers(), c).await?;
        writer.token(b"}")
    }
}
