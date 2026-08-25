use runmat_value::NumericStorage;

use crate::mxarray::MxArrayData;
use crate::{MxApiMode, MxArray, MxInterleaved, MxNumeric};

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
                        values: crate::mxarray::MxInterleavedStorage::from_components(
                            &real,
                            &NumericStorage::zeros(real.numeric_dtype(), real.len()),
                        )?,
                    });
                } else {
                    numeric.imag = Some(NumericStorage::zeros(
                        numeric.real.numeric_dtype(),
                        numeric.real.len(),
                    ));
                }
                Ok(())
            }
            MxArrayData::Numeric(_) | MxArrayData::Interleaved(_) => Ok(()),
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
                *value.data_mut() = MxArrayData::Numeric(MxNumeric { real, imag: None });
                Ok(())
            }
            _ => Err("only numeric mxArrays can be made real".into()),
        }
    }
}
