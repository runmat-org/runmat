//! Borrowed, effect-free randn source description for a selected loop. This
//! accepts a conservative subset of the ordinary constructor grammar: a
//! rejected form is evaluated by the original source instruction instead.

use crate::builtins::common::random_args::{
    checked_constructor_float_dimension, checked_constructor_integer_dimension,
    checked_constructor_numeric_dimension,
};
use crate::context::RuntimeContext;
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_value::{IntValue, NumericDType, Value};

#[derive(Clone, Copy, Debug)]
pub enum ValueControlRef<'a> {
    Borrowed(&'a Value),
    Scalar(ScalarLiteral),
    Integer(&'a IntValue),
    Text(&'a str),
}

#[derive(Clone, Copy, Debug)]
pub struct SourceNoiseClass {
    pub dtype: NumericDType,
    pub complex: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RandnKeyword {
    Like,
    Single,
    Double,
    GpuArray,
    Unknown,
}

impl RandnKeyword {
    pub(super) fn from_text(text: &str) -> Self {
        if text.eq_ignore_ascii_case("like") {
            Self::Like
        } else if text.eq_ignore_ascii_case("single") {
            Self::Single
        } else if text.eq_ignore_ascii_case("double") {
            Self::Double
        } else if text.eq_ignore_ascii_case("gpuarray") {
            Self::GpuArray
        } else {
            Self::Unknown
        }
    }

    fn from_borrowed(control: ValueControlRef<'_>) -> Option<Self> {
        if !keyword_like(control) {
            return None;
        }
        if keyword_is(control, "like") {
            Some(Self::Like)
        } else if keyword_is(control, "single") {
            Some(Self::Single)
        } else if keyword_is(control, "double") {
            Some(Self::Double)
        } else if keyword_is(control, "gpuarray") {
            Some(Self::GpuArray)
        } else {
            Some(Self::Unknown)
        }
    }
}

pub(super) fn normalized_shape(dimensions: Vec<usize>, saw_dimensions: bool) -> Vec<usize> {
    if !saw_dimensions {
        vec![1, 1]
    } else {
        match dimensions.len() {
            0 => vec![0, 0],
            1 => vec![dimensions[0], dimensions[0]],
            _ => dimensions,
        }
    }
}

fn source_matches_normalized_shape(
    expected: &[usize],
    saw_dimensions: bool,
    dimension_count: usize,
    first_dimension: Option<usize>,
    matches_prefix: bool,
) -> bool {
    if !saw_dimensions {
        expected == [1, 1]
    } else {
        match dimension_count {
            0 => expected == [0, 0],
            1 => first_dimension.is_some_and(|first| expected == [first, first]),
            _ => matches_prefix && expected.len() == dimension_count,
        }
    }
}

fn keyword_like(control: ValueControlRef<'_>) -> bool {
    matches!(
        control,
        ValueControlRef::Text(_)
            | ValueControlRef::Borrowed(
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
            )
    )
}

fn keyword_is(control: ValueControlRef<'_>, expected: &str) -> bool {
    match control {
        ValueControlRef::Text(text) => text.eq_ignore_ascii_case(expected),
        ValueControlRef::Borrowed(Value::String(text)) => text
            .as_str()
            .is_some_and(|text| text.eq_ignore_ascii_case(expected)),
        ValueControlRef::Borrowed(Value::StringArray(array)) if array.data().len() == 1 => array
            .data()[0]
            .as_str()
            .is_some_and(|text| text.eq_ignore_ascii_case(expected)),
        ValueControlRef::Borrowed(Value::CharArray(array)) if array.rows == 1 => {
            array.data().len() == expected.len()
                && array
                    .data()
                    .iter()
                    .zip(expected.bytes())
                    .all(|(actual, expected)| {
                        u8::try_from(actual.raw())
                            .is_ok_and(|actual| actual.eq_ignore_ascii_case(&expected))
                    })
        }
        _ => false,
    }
}

fn prototype_class(control: ValueControlRef<'_>) -> Option<SourceNoiseClass> {
    let (dtype, complex) = match control {
        ValueControlRef::Scalar(ScalarLiteral::F32Bits(_)) => (NumericDType::F32, false),
        ValueControlRef::Scalar(ScalarLiteral::F64Bits(_)) => (NumericDType::F64, false),
        ValueControlRef::Borrowed(Value::Num(_)) => (NumericDType::F64, false),
        ValueControlRef::Borrowed(Value::Complex(_, _)) => (NumericDType::F64, true),
        ValueControlRef::Borrowed(Value::Tensor(tensor)) => (tensor.numeric_dtype(), false),
        ValueControlRef::Borrowed(Value::ComplexTensor(tensor)) => (tensor.numeric_dtype(), true),
        _ => return None,
    };
    matches!(dtype, NumericDType::F32 | NumericDType::F64)
        .then_some(SourceNoiseClass { dtype, complex })
}

fn dimension_count(control: ValueControlRef<'_>) -> Option<usize> {
    match control {
        ValueControlRef::Scalar(_)
        | ValueControlRef::Integer(_)
        | ValueControlRef::Borrowed(Value::Num(_) | Value::Int(_)) => Some(1),
        ValueControlRef::Borrowed(Value::Tensor(tensor)) => {
            let len = tensor.len();
            let shape = tensor.shape();
            let scalar = len == 1;
            let row = shape.len() >= 2 && shape[0] == 1;
            let column = shape.len() >= 2 && shape[1] == 1;
            // The selected source path does not issue the column-vector size
            // extension. Its ordinary randn owner will check that policy.
            (!(column && !row && len > 1)
                && (len == 0 || scalar || row || column || shape.len() == 1))
                .then_some(len)
        }
        _ => None,
    }
}

fn dimension_at(control: ValueControlRef<'_>, index: usize) -> Option<usize> {
    match control {
        ValueControlRef::Scalar(ScalarLiteral::F32Bits(bits)) if index == 0 => {
            checked_constructor_float_dimension(f64::from(f32::from_bits(bits))).ok()
        }
        ValueControlRef::Scalar(ScalarLiteral::F64Bits(bits)) if index == 0 => {
            checked_constructor_float_dimension(f64::from_bits(bits)).ok()
        }
        ValueControlRef::Integer(value) | ValueControlRef::Borrowed(Value::Int(value))
            if index == 0 =>
        {
            checked_constructor_integer_dimension(value).ok()
        }
        ValueControlRef::Borrowed(Value::Num(value)) if index == 0 => {
            checked_constructor_float_dimension(*value).ok()
        }
        ValueControlRef::Borrowed(Value::Tensor(tensor)) => {
            checked_constructor_numeric_dimension(tensor.numeric_value_at(index)?).ok()
        }
        _ => None,
    }
}

/// No temporary `Vec<Value>`, keyword `String`, dimension `Vec`, or boxed
/// recursive parser future is made before the caller's host admission. The
/// ordinary randn parser remains the authority for every form we decline.
pub fn describe_borrowed_source<'a>(
    context: &RuntimeContext,
    count: usize,
    at: &'a dyn Fn(usize) -> Option<ValueControlRef<'a>>,
    expected_shape: &[usize],
) -> Option<SourceNoiseClass> {
    let mut dtype = NumericDType::F64;
    let mut template = None;
    let mut saw_dimensions = false;
    let mut saw_size_vector = false;
    let mut dimensions = 0usize;
    let mut first_dimension = None;
    let mut matches_prefix = true;
    let mut index = 0;
    while index < count {
        if context.is_cancelled() {
            return None;
        }
        let control = at(index)?;
        if let Some(keyword) = RandnKeyword::from_borrowed(control) {
            match keyword {
                RandnKeyword::Like => {
                    template = Some(prototype_class(at(index.checked_add(1)?)?)?);
                    index = index.checked_add(2)?;
                    continue;
                }
                RandnKeyword::Single | RandnKeyword::Double => {
                    dtype = if keyword == RandnKeyword::Single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    };
                    template = None;
                    index = index.checked_add(1)?;
                    continue;
                }
                // Explicit GPU generation and unknown keywords stay in randn's
                // original source path, including its error and stream timing.
                RandnKeyword::GpuArray | RandnKeyword::Unknown => return None,
            }
        }
        let count_for_arg = dimension_count(control)?;
        if count_for_arg > 1 {
            if saw_size_vector || saw_dimensions {
                return None;
            }
            saw_size_vector = true;
        } else if saw_size_vector {
            return None;
        }
        saw_dimensions = true;
        for offset in 0..count_for_arg {
            if offset % 1024 == 0 && context.is_cancelled() {
                return None;
            }
            let dimension = dimension_at(control, offset)?;
            if dimensions == 0 {
                first_dimension = Some(dimension);
            }
            matches_prefix &= expected_shape.get(dimensions) == Some(&dimension);
            dimensions = dimensions.checked_add(1)?;
        }
        index = index.checked_add(1)?;
    }
    let shape_matches = source_matches_normalized_shape(
        expected_shape,
        saw_dimensions,
        dimensions,
        first_dimension,
        matches_prefix,
    );
    shape_matches.then_some(template.unwrap_or(SourceNoiseClass {
        dtype,
        complex: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use runmat_value::{ComplexTensor, Tensor};
    use std::rc::Rc;

    fn context() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn selected_source_preserves_exact_integer_size_and_native_class() {
        let wide = IntValue::U64(9_007_199_254_740_993);
        let args = [
            ValueControlRef::Integer(&wide),
            ValueControlRef::Scalar(ScalarLiteral::F64Bits(1.0_f64.to_bits())),
            ValueControlRef::Text("single"),
        ];
        let description = describe_borrowed_source(
            &context(),
            args.len(),
            &|index| args.get(index).copied(),
            &[9_007_199_254_740_993, 1],
        )
        .expect("exact uint64 dimension and single class");
        assert_eq!(description.dtype, NumericDType::F32);
        assert!(!description.complex);
        assert!(describe_borrowed_source(
            &context(),
            args.len(),
            &|index| args.get(index).copied(),
            &[9_007_199_254_740_992, 1]
        )
        .is_none());
    }

    #[test]
    fn selected_source_declines_extension_and_invalid_controls() {
        let column = Tensor::new(vec![2.0, 1.0], vec![2, 1]).expect("column sizes");
        let column_value = Value::Tensor(column);
        let args = [ValueControlRef::Borrowed(&column_value)];
        assert!(describe_borrowed_source(
            &context(),
            1,
            &|index| args.get(index).copied(),
            &[2, 1]
        )
        .is_none());
        let invalid = [ValueControlRef::Scalar(ScalarLiteral::F64Bits(
            1.5_f64.to_bits(),
        ))];
        assert!(describe_borrowed_source(
            &context(),
            1,
            &|index| invalid.get(index).copied(),
            &[1, 1]
        )
        .is_none());
    }

    #[test]
    fn borrowed_description_keeps_randn_shape_class_and_option_precedence() {
        let context = context();
        let default = [Value::Num(2.0)];
        let source = describe_borrowed_source(
            &context,
            default.len(),
            &|index| default.get(index).map(ValueControlRef::Borrowed),
            &[2, 2],
        )
        .expect("default double source");
        assert_eq!(source.dtype, NumericDType::F64);
        assert!(!source.complex);

        let single = [Value::Num(2.0), Value::Num(1.0), Value::from("single")];
        let source = describe_borrowed_source(
            &context,
            single.len(),
            &|index| single.get(index).map(ValueControlRef::Borrowed),
            &[2, 1],
        )
        .expect("explicit single source");
        assert_eq!(source.dtype, NumericDType::F32);
        assert!(!source.complex);

        let prototype = ComplexTensor::from_f32(vec![(0.0, 0.0); 2], vec![2, 1])
            .expect("complex single prototype");
        let complex = [
            Value::Num(2.0),
            Value::Num(1.0),
            Value::from("like"),
            Value::ComplexTensor(prototype),
        ];
        let source = describe_borrowed_source(
            &context,
            complex.len(),
            &|index| complex.get(index).map(ValueControlRef::Borrowed),
            &[2, 1],
        )
        .expect("complex class is described before execution declines it");
        assert_eq!(source.dtype, NumericDType::F32);
        assert!(source.complex);

        let explicit_gpu = [Value::from("gpuArray")];
        assert!(describe_borrowed_source(
            &context,
            explicit_gpu.len(),
            &|index| explicit_gpu.get(index).map(ValueControlRef::Borrowed),
            &[1, 1]
        )
        .is_none());
    }
}
