use super::{
    numerical_output_error, unsupported_output, NumericalAllocation, PreparedHostNumericalOutput,
};
use crate::context::RuntimeContext;
use runmat_types::NumericDomain;
use runmat_value::{ComplexElement, Value};

impl PreparedHostNumericalOutput {
    pub(crate) fn generate_promoting_f32(
        mut self,
        context: &RuntimeContext,
        mut generate: impl FnMut(usize) -> Result<(NumericDomain, (f32, f32)), crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        let mut publish_complex = false;
        let mut output = BorrowedFloatingOutput::f32(&mut self.allocation)?;
        for index in 0..self.len {
            if index % Self::COPY_CHUNK == 0 {
                self.receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            let (domain, (real, imaginary)) = generate(index)?;
            publish_complex |= domain == NumericDomain::Complex;
            output.write(index, domain, real, imaginary)?;
        }
        self.finish_promoting(context, publish_complex)
    }

    pub(crate) fn generate_promoting_f64(
        mut self,
        context: &RuntimeContext,
        mut generate: impl FnMut(usize) -> Result<(NumericDomain, (f64, f64)), crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        let mut publish_complex = false;
        let mut output = BorrowedFloatingOutput::f64(&mut self.allocation)?;
        for index in 0..self.len {
            if index % Self::COPY_CHUNK == 0 {
                self.receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            let (domain, (real, imaginary)) = generate(index)?;
            publish_complex |= domain == NumericDomain::Complex;
            output.write(index, domain, real, imaginary)?;
        }
        self.finish_promoting(context, publish_complex)
    }

    pub(crate) fn map_promoting_f32(
        mut self,
        context: &RuntimeContext,
        source: &[f32],
        publish_complex: bool,
        mut map: impl FnMut(f32) -> Result<(f32, f32), crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        self.ensure_source_len(source.len())?;
        let mut output = BorrowedFloatingOutput::f32(&mut self.allocation)?;
        for (index, source) in source.iter().copied().enumerate() {
            if index % Self::COPY_CHUNK == 0 {
                self.receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            let (real, imaginary) = map(source)?;
            let domain = if publish_complex {
                NumericDomain::Complex
            } else {
                NumericDomain::Real
            };
            output.write(index, domain, real, imaginary)?;
        }
        self.finish_promoting(context, publish_complex)
    }

    pub(crate) fn map_promoting_f64(
        mut self,
        context: &RuntimeContext,
        source: &[f64],
        publish_complex: bool,
        mut map: impl FnMut(f64) -> Result<(f64, f64), crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        self.ensure_source_len(source.len())?;
        let mut output = BorrowedFloatingOutput::f64(&mut self.allocation)?;
        for (index, source) in source.iter().copied().enumerate() {
            if index % Self::COPY_CHUNK == 0 {
                self.receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            let (real, imaginary) = map(source)?;
            let domain = if publish_complex {
                NumericDomain::Complex
            } else {
                NumericDomain::Real
            };
            output.write(index, domain, real, imaginary)?;
        }
        self.finish_promoting(context, publish_complex)
    }

    fn finish_promoting(
        self,
        context: &RuntimeContext,
        publish_complex: bool,
    ) -> Result<Value, crate::RuntimeError> {
        match (
            self.allocation.real.is_some(),
            self.allocation.complex.is_some(),
            publish_complex,
        ) {
            (true, _, false) => self
                .into_tensor(context)
                .map(Value::Tensor)
                .map_err(|error| numerical_output_error(context, error)),
            (_, true, true) | (false, true, false) => self
                .into_complex_tensor(context)
                .map(Value::ComplexTensor)
                .map_err(|error| numerical_output_error(context, error)),
            _ => Err(unsupported_output()),
        }
    }
}

struct BorrowedFloatingOutput<'a, T> {
    real: Option<&'a mut [T]>,
    complex: Option<&'a mut [ComplexElement<T>]>,
}

impl<'a> BorrowedFloatingOutput<'a, f32> {
    fn f32(allocation: &'a mut NumericalAllocation) -> Result<Self, crate::RuntimeError> {
        let real = allocation
            .real
            .as_mut()
            .map(|target| {
                target
                    .try_as_f32_slice_mut_unique()
                    .ok_or_else(unsupported_output)
            })
            .transpose()?;
        let complex = allocation
            .complex
            .as_mut()
            .map(|target| {
                target
                    .try_as_f32_slice_mut_unique()
                    .ok_or_else(unsupported_output)
            })
            .transpose()?;
        Self::new(real, complex)
    }
}

impl<'a> BorrowedFloatingOutput<'a, f64> {
    fn f64(allocation: &'a mut NumericalAllocation) -> Result<Self, crate::RuntimeError> {
        let real = allocation
            .real
            .as_mut()
            .map(|target| {
                target
                    .try_as_f64_slice_mut_unique()
                    .ok_or_else(unsupported_output)
            })
            .transpose()?;
        let complex = allocation
            .complex
            .as_mut()
            .map(|target| {
                target
                    .try_as_f64_slice_mut_unique()
                    .ok_or_else(unsupported_output)
            })
            .transpose()?;
        Self::new(real, complex)
    }
}

impl<'a, T: Copy> BorrowedFloatingOutput<'a, T> {
    fn new(
        real: Option<&'a mut [T]>,
        complex: Option<&'a mut [ComplexElement<T>]>,
    ) -> Result<Self, crate::RuntimeError> {
        if real.is_none() && complex.is_none() {
            return Err(unsupported_output());
        }
        Ok(Self { real, complex })
    }

    fn write(
        &mut self,
        index: usize,
        domain: NumericDomain,
        real: T,
        imaginary: T,
    ) -> Result<(), crate::RuntimeError> {
        if let Some(target) = self.real.as_mut() {
            if domain != NumericDomain::Real && self.complex.is_none() {
                return Err(unsupported_output());
            }
            target[index] = real;
        }
        if let Some(target) = self.complex.as_mut() {
            target[index] = ComplexElement(real, imaginary);
        }
        Ok(())
    }
}
