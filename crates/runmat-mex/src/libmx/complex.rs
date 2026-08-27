use runmat_value::{
    record_host_copy, ComplexElement, HostComplexBuffer, HostCopyReason, HostNumericBuffer,
    NumericDType, NumericStorage,
};

use crate::mxarray::MxArrayData;
use crate::{MxApiMode, MxArray, MxInterleaved, MxNumeric, MxSparseValues};

use super::MxApi;

impl MxApi {
    pub fn make_complex(&mut self, value: *mut MxArray) -> Result<(), String> {
        let mode = self.mode;
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        match value.data_mut() {
            MxArrayData::Numeric(numeric) if numeric.imag.is_none() => {
                if mode == MxApiMode::InterleavedComplex {
                    let real = numeric.real.clone();
                    *value.data_mut() = MxArrayData::Interleaved(MxInterleaved {
                        values: crate::mxarray::MxInterleavedStorage::from_host_components(
                            &real,
                            &HostNumericBuffer::from_numeric_storage(NumericStorage::zeros(
                                real.numeric_dtype(),
                                real.len(),
                            )),
                        )?,
                    });
                } else {
                    numeric.imag = Some(HostNumericBuffer::from_numeric_storage(
                        NumericStorage::zeros(numeric.real.numeric_dtype(), numeric.real.len()),
                    ));
                }
                Ok(())
            }
            MxArrayData::Numeric(_) | MxArrayData::Interleaved(_) => Ok(()),
            MxArrayData::Sparse(sparse) => match &mut sparse.values {
                MxSparseValues::Numeric(real) if real.numeric_dtype() == NumericDType::F64 => {
                    if mode == MxApiMode::InterleavedComplex {
                        let values = real
                            .as_f64_slice()
                            .expect("double sparse storage")
                            .iter()
                            .map(|value| ComplexElement(*value, 0.0))
                            .collect::<Vec<_>>();
                        record_host_copy(
                            HostCopyReason::SparseLayoutConversion,
                            values
                                .len()
                                .saturating_mul(std::mem::size_of::<ComplexElement<f64>>()),
                        );
                        sparse.values = MxSparseValues::InterleavedComplex(
                            HostComplexBuffer::from_elements(values),
                        );
                    } else {
                        sparse.values = MxSparseValues::SeparateComplex {
                            real: real.clone(),
                            imaginary: HostNumericBuffer::from_numeric_storage(
                                NumericStorage::F64(vec![0.0; real.len()]),
                            ),
                        };
                    }
                    Ok(())
                }
                MxSparseValues::InterleavedComplex(_) | MxSparseValues::SeparateComplex { .. } => {
                    Ok(())
                }
                MxSparseValues::Numeric(_) | MxSparseValues::Logical(_) => {
                    Err("only double sparse mxArrays can be made complex".into())
                }
            },
            _ => Err("only numeric mxArrays can be made complex".into()),
        }
    }

    pub fn make_real(&mut self, value: *mut MxArray) -> Result<(), String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        match value.data_mut() {
            MxArrayData::Numeric(numeric) => {
                numeric.imag = None;
                Ok(())
            }
            MxArrayData::Interleaved(interleaved) => {
                let (real, _) = interleaved.values.components();
                *value.data_mut() = MxArrayData::Numeric(MxNumeric {
                    real: HostNumericBuffer::from_numeric_storage(real),
                    imag: None,
                });
                Ok(())
            }
            MxArrayData::Sparse(sparse) => match &mut sparse.values {
                MxSparseValues::SeparateComplex { real, .. } => {
                    sparse.values = MxSparseValues::Numeric(real.clone());
                    Ok(())
                }
                MxSparseValues::InterleavedComplex(values) => {
                    let real = values.iter().map(|value| value.0).collect::<Vec<_>>();
                    record_host_copy(
                        HostCopyReason::SparseLayoutConversion,
                        real.len().saturating_mul(std::mem::size_of::<f64>()),
                    );
                    sparse.values = MxSparseValues::Numeric(
                        HostNumericBuffer::from_numeric_storage(NumericStorage::F64(real)),
                    );
                    Ok(())
                }
                MxSparseValues::Numeric(_) => Ok(()),
                MxSparseValues::Logical(_) => Err("only numeric mxArrays can be made real".into()),
            },
            _ => Err("only numeric mxArrays can be made real".into()),
        }
    }
}
