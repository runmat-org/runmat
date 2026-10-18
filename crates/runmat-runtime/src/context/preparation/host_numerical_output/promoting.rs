use super::{
    numerical_output_error, unsupported_output, NumericalAllocation, PreparedHostNumericalOutput,
};
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use runmat_types::NumericDomain;
use runmat_value::Value;

impl PreparedHostNumericalOutput {
    pub(crate) fn generate_promoting_f32(
        mut self,
        context: &RuntimeContext,
        mut generate: impl FnMut(usize) -> Result<(NumericDomain, (f32, f32)), crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        let mut publish_complex = false;
        for index in 0..self.len {
            if index % Self::COPY_CHUNK == 0 {
                self.receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            let (domain, (real, imaginary)) = generate(index)?;
            publish_complex |= domain == NumericDomain::Complex;
            write_f32(
                context,
                &mut self.allocation,
                index,
                domain,
                real,
                imaginary,
            )?;
        }
        self.finish_promoting(context, publish_complex)
    }

    pub(crate) fn generate_promoting_f64(
        mut self,
        context: &RuntimeContext,
        mut generate: impl FnMut(usize) -> Result<(NumericDomain, (f64, f64)), crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        let mut publish_complex = false;
        for index in 0..self.len {
            if index % Self::COPY_CHUNK == 0 {
                self.receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            let (domain, (real, imaginary)) = generate(index)?;
            publish_complex |= domain == NumericDomain::Complex;
            write_f64(
                context,
                &mut self.allocation,
                index,
                domain,
                real,
                imaginary,
            )?;
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
            write_f32(
                context,
                &mut self.allocation,
                index,
                domain,
                real,
                imaginary,
            )?;
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
            write_f64(
                context,
                &mut self.allocation,
                index,
                domain,
                real,
                imaginary,
            )?;
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

fn write_f32(
    context: &RuntimeContext,
    allocation: &mut NumericalAllocation,
    index: usize,
    domain: NumericDomain,
    real: f32,
    imaginary: f32,
) -> Result<(), crate::RuntimeError> {
    if let Some(target) = allocation.real.as_mut() {
        if domain != NumericDomain::Real && allocation.complex.is_none() {
            return Err(unsupported_output());
        }
        target.as_f32_slice_mut().ok_or_else(unsupported_output)?[index] = real;
    }
    if let Some(target) = allocation.complex.as_mut() {
        target
            .set(index, f64::from(real), f64::from(imaginary))
            .map_err(|error| {
                numerical_output_error(context, HostAllocationError::Numeric(error))
            })?;
    }
    if allocation.real.is_none() && allocation.complex.is_none() {
        return Err(unsupported_output());
    }
    Ok(())
}

fn write_f64(
    context: &RuntimeContext,
    allocation: &mut NumericalAllocation,
    index: usize,
    domain: NumericDomain,
    real: f64,
    imaginary: f64,
) -> Result<(), crate::RuntimeError> {
    if let Some(target) = allocation.real.as_mut() {
        if domain != NumericDomain::Real && allocation.complex.is_none() {
            return Err(unsupported_output());
        }
        target.as_f64_slice_mut().ok_or_else(unsupported_output)?[index] = real;
    }
    if let Some(target) = allocation.complex.as_mut() {
        target.set(index, real, imaginary).map_err(|error| {
            numerical_output_error(context, HostAllocationError::Numeric(error))
        })?;
    }
    if allocation.real.is_none() && allocation.complex.is_none() {
        return Err(unsupported_output());
    }
    Ok(())
}
