use super::InProcessNumericData;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::{GpuTensorHandle, NumericElementType};

pub(super) fn floating_payload(
    handle: &GpuTensorHandle,
    payload: &InProcessNumericData,
) -> Result<Vec<f64>> {
    match (handle.physical().element_type(), payload) {
        (NumericElementType::F64, InProcessNumericData::F64(values)) => {
            let mut output = Vec::new();
            output
                .try_reserve_exact(values.len())
                .map_err(|source| anyhow!(source).context("iir_filter: input allocation failed"))?;
            output.extend_from_slice(values);
            Ok(output)
        }
        (NumericElementType::F32, InProcessNumericData::F32(values)) => {
            let mut output = Vec::new();
            output
                .try_reserve_exact(values.len())
                .map_err(|source| anyhow!(source).context("iir_filter: input allocation failed"))?;
            output.extend(values.iter().copied().map(f64::from));
            Ok(output)
        }
        (declared, _) => Err(anyhow!(
            "iir_filter: physical element type {:?} does not match a floating payload",
            declared
        )),
    }
}

pub(super) fn output_payload(
    values: Vec<f64>,
    physical: NumericElementType,
) -> Result<InProcessNumericData> {
    match physical {
        NumericElementType::F64 => Ok(InProcessNumericData::F64(values)),
        NumericElementType::F32 => {
            let mut output = Vec::new();
            output.try_reserve_exact(values.len()).map_err(|source| {
                anyhow!(source).context("iir_filter: output allocation failed")
            })?;
            output.extend(values.into_iter().map(|value| value as f32));
            Ok(InProcessNumericData::F32(output))
        }
        other => Err(anyhow!(
            "iir_filter: unsupported signal element type {:?}",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_preserves_signal_precision_without_f64_mirror() {
        assert_eq!(
            output_payload(vec![1.25, -2.5], NumericElementType::F32).unwrap(),
            InProcessNumericData::F32(vec![1.25, -2.5])
        );
        assert_eq!(
            output_payload(vec![1.25], NumericElementType::F64).unwrap(),
            InProcessNumericData::F64(vec![1.25])
        );
    }
}
