//! Sample-point host controls; admitted only after the shared host-scratch fence.
use super::*;

pub(super) async fn parse_sample_window(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<WindowSpec> {
    let values = numeric_vector(name, value).await?;
    match values.as_slice() {
        [raw] => {
            let width = positive_span(name, *raw, "sample-point window length")?;
            Ok(WindowSpec::Span {
                before: width / 2.0,
                after: width / 2.0,
            })
        }
        [before, after] => Ok(WindowSpec::Span {
            before: nonnegative_span(name, *before, "sample-point backward window")?,
            after: nonnegative_span(name, *after, "sample-point forward window")?,
        }),
        _ => Err(invalid_argument(
            name,
            format!("{name}: sample-point window must be a scalar or two-element vector"),
        )),
    }
}

pub(super) async fn parse_sample_points(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<f64>> {
    let points = numeric_vector(name, value).await?;
    if points.is_empty() {
        return Err(invalid_argument(
            name,
            format!("{name}: SamplePoints must not be empty"),
        ));
    }
    let mut previous = None;
    for point in &points {
        if !point.is_finite() {
            return Err(invalid_argument(
                name,
                format!("{name}: SamplePoints must be finite"),
            ));
        }
        if let Some(prev) = previous {
            if *point <= prev {
                return Err(invalid_argument(
                    name,
                    format!("{name}: SamplePoints must be strictly increasing"),
                ));
            }
        }
        previous = Some(*point);
    }
    Ok(points)
}

async fn numeric_vector(name: &'static str, value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(t) => {
            super::counts::validate_vector_shape(name, t.shape(), t.len())?;
            Ok(tensor::tensor_values_f64(t))
        }
        Value::LogicalArray(l) => {
            super::counts::validate_vector_shape(name, &l.shape, l.data.len())?;
            Ok(l.data
                .iter()
                .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                .collect())
        }
        _ => Err(invalid_argument(
            name,
            format!("{name}: window must be numeric"),
        )),
    }
}
