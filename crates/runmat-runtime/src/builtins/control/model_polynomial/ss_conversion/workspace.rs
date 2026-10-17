//! Explicit state-polynomial workspaces shared by synchronous and cooperative drivers.
use super::{conversion_error, ConversionErrorKind, ConversionResult};
use crate::context::host_allocation::HostComplexWorkspace;
use nalgebra::DMatrix;
use num_complex::Complex64;

pub(super) trait ComplexMatrixRead {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn at(&self, row: usize, col: usize) -> Complex64;
}

impl ComplexMatrixRead for DMatrix<Complex64> {
    fn nrows(&self) -> usize {
        self.shape().0
    }
    fn ncols(&self) -> usize {
        self.shape().1
    }
    fn at(&self, row: usize, col: usize) -> Complex64 {
        self[(row, col)]
    }
}

pub(super) enum ComplexScratch {
    Ordinary(Vec<Complex64>),
    Managed(HostComplexWorkspace),
}

impl ComplexScratch {
    fn get(&self, index: usize) -> Complex64 {
        match self {
            Self::Ordinary(values) => values[index],
            Self::Managed(values) => values
                .get(index)
                .expect("validated complex workspace index"),
        }
    }

    fn set(&mut self, index: usize, value: Complex64) {
        match self {
            Self::Ordinary(values) => values[index] = value,
            Self::Managed(values) => {
                values
                    .set(index, value)
                    .expect("validated complex workspace index");
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Ordinary(values) => values.len(),
            Self::Managed(values) => values.len(),
        }
    }
}

fn reserved_zeros(length: usize) -> ConversionResult<Vec<Complex64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| {
        conversion_error(
            "state polynomial workspace exceeds available host memory",
            ConversionErrorKind::Internal,
        )
    })?;
    values.resize(length, Complex64::new(0.0, 0.0));
    Ok(values)
}

pub(super) struct CharacteristicWorkspace {
    pub(super) coefficients: Vec<Complex64>,
    current: ComplexScratch,
    next: ComplexScratch,
    order: usize,
}

impl CharacteristicWorkspace {
    pub(super) fn new(order: usize) -> ConversionResult<Self> {
        let square = order.checked_mul(order).ok_or_else(|| {
            conversion_error(
                "state polynomial matrix dimensions overflow",
                ConversionErrorKind::InvalidModel,
            )
        })?;
        Self::with_scratch(
            order,
            ComplexScratch::Ordinary(reserved_zeros(square)?),
            ComplexScratch::Ordinary(reserved_zeros(square)?),
        )
    }

    pub(super) fn with_scratch(
        order: usize,
        mut current: ComplexScratch,
        next: ComplexScratch,
    ) -> ConversionResult<Self> {
        let square = order.checked_mul(order).ok_or_else(|| {
            conversion_error(
                "state polynomial matrix dimensions overflow",
                ConversionErrorKind::InvalidModel,
            )
        })?;
        if current.len() != square || next.len() != square {
            return Err(conversion_error(
                "state polynomial workspace has the wrong length",
                ConversionErrorKind::Internal,
            ));
        }
        let coefficient_count = order.checked_add(1).ok_or_else(|| {
            conversion_error(
                "state polynomial coefficient count overflows",
                ConversionErrorKind::InvalidModel,
            )
        })?;
        for diagonal in 0..order {
            current.set(diagonal + diagonal * order, Complex64::new(1.0, 0.0));
        }
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(coefficient_count)
            .map_err(|_| {
                conversion_error(
                    "state polynomial coefficient workspace exceeds available host memory",
                    ConversionErrorKind::Internal,
                )
            })?;
        coefficients.push(Complex64::new(1.0, 0.0));
        Ok(Self {
            coefficients,
            current,
            next,
            order,
        })
    }

    /// One dot product; the driver can yield after each bounded row of work.
    pub(super) fn advance_cell<M: ComplexMatrixRead>(&mut self, a: &M, row: usize, col: usize) {
        self.advance_cell_fragment(a, row, col, 0, self.order);
    }

    pub(super) fn advance_cell_fragment<M: ComplexMatrixRead>(
        &mut self,
        a: &M,
        row: usize,
        col: usize,
        start: usize,
        end: usize,
    ) {
        let slot = row + col * self.order;
        let mut value = if start == 0 {
            Complex64::new(0.0, 0.0)
        } else {
            self.next.get(slot)
        };
        for inner in start..end {
            value += a.at(row, inner) * self.current.get(inner + col * self.order);
        }
        self.next.set(slot, value);
    }

    pub(super) fn finish_degree(&mut self, degree: usize) {
        let trace = self.trace_fragment(0, self.order, Complex64::new(0.0, 0.0));
        let coefficient = self.append_coefficient(degree, trace);
        self.diagonal_fragment(0, self.order, coefficient);
        self.swap_degree();
    }

    pub(super) fn trace_fragment(
        &self,
        start: usize,
        end: usize,
        mut trace: Complex64,
    ) -> Complex64 {
        for diagonal in start..end {
            trace += self.next.get(diagonal + diagonal * self.order);
        }
        trace
    }

    pub(super) fn append_coefficient(&mut self, degree: usize, trace: Complex64) -> Complex64 {
        let coefficient = -trace / Complex64::new(degree as f64, 0.0);
        self.coefficients.push(coefficient);
        coefficient
    }

    pub(super) fn diagonal_fragment(&mut self, start: usize, end: usize, coefficient: Complex64) {
        for diagonal in start..end {
            let slot = diagonal + diagonal * self.order;
            self.next.set(slot, self.next.get(slot) + coefficient);
        }
    }

    pub(super) fn swap_degree(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }

    pub(super) fn into_parts(self) -> (Vec<Complex64>, ComplexScratch, ComplexScratch) {
        (self.coefficients, self.current, self.next)
    }
}

pub(super) struct NumeratorWorkspace {
    pub(super) coefficients: Vec<Complex64>,
    state: ComplexScratch,
    next: ComplexScratch,
    order: usize,
}

impl NumeratorWorkspace {
    pub(super) fn new<M: ComplexMatrixRead>(
        b: &M,
        input: usize,
        direct: Complex64,
        denominator: &[Complex64],
    ) -> ConversionResult<Self> {
        Self::with_scratch(
            b,
            input,
            direct,
            denominator,
            ComplexScratch::Ordinary(reserved_zeros(b.nrows())?),
            ComplexScratch::Ordinary(reserved_zeros(b.nrows())?),
        )
    }

    pub(super) fn with_scratch<M: ComplexMatrixRead>(
        b: &M,
        input: usize,
        direct: Complex64,
        denominator: &[Complex64],
        mut state: ComplexScratch,
        next: ComplexScratch,
    ) -> ConversionResult<Self> {
        if state.len() < b.nrows() || next.len() < b.nrows() {
            return Err(conversion_error(
                "state numerator workspace has the wrong length",
                ConversionErrorKind::Internal,
            ));
        }
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(denominator.len())
            .map_err(|_| {
                conversion_error(
                    "state numerator coefficient workspace exceeds available host memory",
                    ConversionErrorKind::Internal,
                )
            })?;
        coefficients.extend(denominator.iter().map(|coefficient| direct * *coefficient));
        for row in 0..b.nrows() {
            state.set(row, b.at(row, input));
        }
        Ok(Self {
            coefficients,
            next,
            state,
            order: b.nrows(),
        })
    }

    pub(super) fn finish_markov<M: ComplexMatrixRead>(
        &mut self,
        c: &M,
        output: usize,
        denominator: &[Complex64],
        degree: usize,
    ) {
        let markov = self.markov_fragment(c, output, 0, self.order, Complex64::new(0.0, 0.0));
        self.apply_markov_fragment(denominator, degree, degree, self.coefficients.len(), markov);
    }

    pub(super) fn state_len(&self) -> usize {
        self.order
    }

    pub(super) fn coefficient_len(&self) -> usize {
        self.coefficients.len()
    }

    pub(super) fn markov_fragment<M: ComplexMatrixRead>(
        &self,
        c: &M,
        output: usize,
        start: usize,
        end: usize,
        mut sum: Complex64,
    ) -> Complex64 {
        for row in start..end {
            sum += c.at(output, row) * self.state.get(row);
        }
        sum
    }

    pub(super) fn apply_markov_fragment(
        &mut self,
        denominator: &[Complex64],
        degree: usize,
        start: usize,
        end: usize,
        markov: Complex64,
    ) {
        for index in start..end {
            self.coefficients[index] += denominator[index - degree] * markov;
        }
    }

    pub(super) fn advance_row<M: ComplexMatrixRead>(&mut self, a: &M, row: usize) {
        self.advance_row_fragment(a, row, 0, self.order);
    }

    pub(super) fn advance_row_fragment<M: ComplexMatrixRead>(
        &mut self,
        a: &M,
        row: usize,
        start: usize,
        end: usize,
    ) {
        let mut value = if start == 0 {
            Complex64::new(0.0, 0.0)
        } else {
            self.next.get(row)
        };
        for inner in start..end {
            value += a.at(row, inner) * self.state.get(inner);
        }
        self.next.set(row, value);
    }

    pub(super) fn finish_state(&mut self) {
        std::mem::swap(&mut self.state, &mut self.next);
    }

    pub(super) fn into_parts(self) -> (Vec<Complex64>, ComplexScratch, ComplexScratch) {
        (self.coefficients, self.state, self.next)
    }
}
