//! Image-class adaptation and host histogram evaluation.
use super::*;

pub(super) struct GrayscaleInput {
    storage: NumericStorage,
    class_min: f64,
    class_max: f64,
    default_bins: usize,
}

impl GrayscaleInput {
    pub(super) fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::LogicalArray(logical) => Self::from_logical(logical),
            Value::Num(value) => {
                Self::from_float_storage(NumericStorage::F64(vec![value]), &[1, 1])
            }
            Value::Int(value) => Ok(Self::from_integer_scalar(value)),
            Value::Bool(value) => Ok(Self {
                storage: NumericStorage::F64(vec![if value { 1.0 } else { 0.0 }]),
                class_min: 0.0,
                class_max: 1.0,
                default_bins: LOGICAL_BINS,
            }),
            other => Err(unsupported(format!(
                "expected grayscale numeric or logical image, got {other:?}"
            ))),
        }
    }

    pub(super) fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let shape = tensor.shape().to_vec();
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| internal(format!("grayscale tensor storage: {err}")))?;
        match storage {
            storage @ NumericStorage::I8(_) => Ok(Self::from_integer_storage(
                storage,
                i8::MIN as f64,
                i8::MAX as f64,
            )),
            storage @ NumericStorage::I16(_) => Ok(Self::from_integer_storage(
                storage,
                i16::MIN as f64,
                i16::MAX as f64,
            )),
            storage @ NumericStorage::I32(_) => Ok(Self::from_integer_storage(
                storage,
                i32::MIN as f64,
                i32::MAX as f64,
            )),
            NumericStorage::I64(_) => Err(unsupported("int64 grayscale images are not supported")),
            storage @ NumericStorage::U8(_) => Ok(Self::from_integer_storage(storage, 0.0, 255.0)),
            storage @ NumericStorage::U16(_) => {
                Ok(Self::from_integer_storage(storage, 0.0, 65535.0))
            }
            storage @ NumericStorage::U32(_) => {
                Ok(Self::from_integer_storage(storage, 0.0, u32::MAX as f64))
            }
            NumericStorage::U64(_) => Err(unsupported("uint64 grayscale images are not supported")),
            storage @ (NumericStorage::F32(_) | NumericStorage::F64(_)) => {
                Self::from_float_storage(storage, &shape)
            }
        }
    }

    fn from_integer_scalar(value: IntValue) -> Self {
        match value {
            IntValue::I8(value) => Self::from_integer_storage(
                NumericStorage::I8(vec![value]),
                i8::MIN as f64,
                i8::MAX as f64,
            ),
            IntValue::I16(value) => Self::from_integer_storage(
                NumericStorage::I16(vec![value]),
                i16::MIN as f64,
                i16::MAX as f64,
            ),
            IntValue::I32(value) => Self::from_integer_storage(
                NumericStorage::I32(vec![value]),
                i32::MIN as f64,
                i32::MAX as f64,
            ),
            IntValue::I64(value) => Self::from_integer_storage(
                NumericStorage::I64(vec![value]),
                i64::MIN as f64,
                i64::MAX as f64,
            ),
            IntValue::U8(value) => {
                Self::from_integer_storage(NumericStorage::U8(vec![value]), 0.0, 255.0)
            }
            IntValue::U16(value) => {
                Self::from_integer_storage(NumericStorage::U16(vec![value]), 0.0, 65535.0)
            }
            IntValue::U32(value) => {
                Self::from_integer_storage(NumericStorage::U32(vec![value]), 0.0, u32::MAX as f64)
            }
            IntValue::U64(value) => {
                Self::from_integer_storage(NumericStorage::U64(vec![value]), 0.0, u64::MAX as f64)
            }
        }
    }

    fn from_logical(logical: LogicalArray) -> BuiltinResult<Self> {
        Ok(Self {
            storage: NumericStorage::F64(
                logical
                    .data
                    .into_iter()
                    .map(|value| if value == 0 { 0.0 } else { 1.0 })
                    .collect(),
            ),
            class_min: 0.0,
            class_max: 1.0,
            default_bins: LOGICAL_BINS,
        })
    }

    fn from_integer_storage(storage: NumericStorage, class_min: f64, class_max: f64) -> Self {
        Self {
            storage,
            class_min,
            class_max,
            default_bins: DEFAULT_GRAYSCALE_BINS,
        }
    }

    fn from_float_storage(storage: NumericStorage, _shape: &[usize]) -> BuiltinResult<Self> {
        let valid = match &storage {
            NumericStorage::F64(values) => values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            NumericStorage::F32(values) => values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            _ => false,
        };
        if !valid {
            return Err(invalid(
                "floating-point grayscale image values must be finite and normalized to [0, 1]",
            ));
        }
        Ok(Self {
            storage,
            class_min: 0.0,
            class_max: 1.0,
            default_bins: DEFAULT_GRAYSCALE_BINS,
        })
    }

    pub(super) fn evaluate(
        &self,
        requested_bins: Option<usize>,
    ) -> BuiltinResult<ImhistEvaluation> {
        let bins = requested_bins.unwrap_or(self.default_bins);
        validate_bin_count(bins)?;
        let locations = linspace(self.class_min, self.class_max, bins);
        let counts = histogram_counts(&self.storage, self.class_min, self.class_max, bins)?;
        ImhistEvaluation::from_counts_locations(counts, locations)
    }
}

pub(super) struct IndexedInput {
    storage: NumericStorage,
    zero_based: bool,
    bins: usize,
}

impl IndexedInput {
    pub(super) fn from_value(value: Value, bins: usize) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => {
                let zero_based = matches!(
                    tensor.numeric_dtype(),
                    runmat_value::NumericDType::U8 | runmat_value::NumericDType::U16
                );
                let storage = tensor
                    .into_numeric_storage()
                    .map_err(|err| internal(format!("indexed tensor storage: {err}")))?;
                Ok(Self {
                    storage,
                    zero_based,
                    bins,
                })
            }
            Value::LogicalArray(logical) => Ok(Self {
                storage: NumericStorage::F64(
                    logical
                        .data
                        .into_iter()
                        .map(|value| if value == 0 { 0.0 } else { 1.0 })
                        .collect(),
                ),
                zero_based: true,
                bins,
            }),
            Value::Num(value) => Ok(Self {
                storage: NumericStorage::F64(vec![value]),
                zero_based: false,
                bins,
            }),
            Value::Int(value) => {
                let zero_based = matches!(value, IntValue::U8(_) | IntValue::U16(_));
                Ok(Self {
                    storage: numeric_storage_from_int(value),
                    zero_based,
                    bins,
                })
            }
            Value::Bool(value) => Ok(Self {
                storage: NumericStorage::F64(vec![if value { 1.0 } else { 0.0 }]),
                zero_based: true,
                bins,
            }),
            other => Err(unsupported(format!(
                "expected indexed numeric or logical image, got {other:?}"
            ))),
        }
    }

    pub(super) fn evaluate(&self) -> BuiltinResult<ImhistEvaluation> {
        let mut counts = vec![0.0; self.bins];
        count_indexed_values(&self.storage, self.zero_based, &mut counts)?;
        let locations: Vec<f64> = (1..=self.bins).map(|value| value as f64).collect();
        ImhistEvaluation::from_counts_locations(counts, locations)
    }
}

fn numeric_storage_from_int(value: IntValue) -> NumericStorage {
    match value {
        IntValue::I8(value) => NumericStorage::I8(vec![value]),
        IntValue::I16(value) => NumericStorage::I16(vec![value]),
        IntValue::I32(value) => NumericStorage::I32(vec![value]),
        IntValue::I64(value) => NumericStorage::I64(vec![value]),
        IntValue::U8(value) => NumericStorage::U8(vec![value]),
        IntValue::U16(value) => NumericStorage::U16(vec![value]),
        IntValue::U32(value) => NumericStorage::U32(vec![value]),
        IntValue::U64(value) => NumericStorage::U64(vec![value]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn exact_integer_grayscale_images_use_their_storage_class_range() {
        let u8 = Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 1, u8::MAX]), vec![2, 2])
            .expect("uint8 tensor");
        let u16 = Tensor::new_integer(
            IntegerStorage::U16(vec![0, 1, u16::MAX, u16::MAX]),
            vec![2, 2],
        )
        .expect("uint16 tensor");

        let u8_image = GrayscaleInput::from_tensor(u8).expect("uint8 image");
        assert_eq!(u8_image.class_max, 255.0);
        assert_eq!(u8_image.default_bins, 256);
        let u8_eval = u8_image.evaluate(None).expect("uint8 histogram");
        assert_eq!(u8_eval.counts.materialize_f64()[0], 1.0);
        assert_eq!(u8_eval.counts.materialize_f64()[1], 2.0);
        assert_eq!(u8_eval.counts.materialize_f64()[255], 1.0);

        let u16_image = GrayscaleInput::from_tensor(u16).expect("uint16 image");
        assert_eq!(u16_image.class_max, 65535.0);
        assert_eq!(u16_image.default_bins, 256);
    }

    #[test]
    fn exact_integer_indexed_images_use_zero_based_indices() {
        let image = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 1, 2]), vec![2, 2])
            .expect("uint16 tensor");
        let indexed = IndexedInput::from_value(Value::Tensor(image), 3).expect("indexed input");

        assert!(indexed.zero_based);
        assert_eq!(indexed.storage, NumericStorage::U16(vec![0, 1, 1, 2]));
    }
}
