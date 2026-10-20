//! Public series interconnection, with numerical work at the model owner.
use crate::builtins::common::tensor;
use crate::builtins::control::model_polynomial::SeriesModel;
use crate::builtins::control::tf_model::{control_error, SS_CLASS, TF_CLASS};
use crate::{dispatcher, BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectArray, Value};

const NAME: &str = "series";

fn argument_error(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:series:InvalidArgument", message)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ModelArrayClass {
    Tf,
    Ss,
}

impl ModelArrayClass {
    fn of(array: &ObjectArray) -> BuiltinResult<Self> {
        if array.class_name().is(TF_CLASS) {
            Ok(Self::Tf)
        } else if array.class_name().is(SS_CLASS) {
            Ok(Self::Ss)
        } else {
            Err(argument_error(
                "series expects transfer-function or state-space model arrays",
            ))
        }
    }

    fn identity(self) -> runmat_types::StaticClassIdentity {
        match self {
            Self::Tf => TF_CLASS,
            Self::Ss => SS_CLASS,
        }
    }

    fn compose(first: Self, second: Self) -> Self {
        if first == Self::Ss || second == Self::Ss {
            Self::Ss
        } else {
            Self::Tf
        }
    }
}

fn admitted_model_kind(value: &Value) -> BuiltinResult<Option<ModelArrayClass>> {
    match value {
        Value::ObjectArray(array) => ModelArrayClass::of(array).map(Some),
        Value::Object(object) if object.is_class(TF_CLASS) => Ok(Some(ModelArrayClass::Tf)),
        Value::Object(object) if object.is_class(SS_CLASS) => Ok(Some(ModelArrayClass::Ss)),
        Value::Num(_) | Value::Complex(_, _) | Value::Int(_) => Ok(None),
        _ => Err(argument_error(
            "series expects tf or ss models or a scalar gain",
        )),
    }
}

fn empty_model_array(shape: Vec<usize>, class: ModelArrayClass) -> BuiltinResult<Value> {
    ObjectArray::empty(class.identity(), shape)
        .map(ObjectArray::with_unselected_provenance)
        .map(Value::ObjectArray)
        .map_err(argument_error)
}

async fn empty_array_with_scalar(
    array: &ObjectArray,
    scalar: Value,
    named: bool,
    channels: Option<&(Vec<usize>, Vec<usize>)>,
) -> BuiltinResult<Value> {
    if named || channels.is_some() {
        return Err(argument_error(
            "empty model arrays do not retain channel names or counts",
        ));
    }
    let array_class = ModelArrayClass::of(array)?;
    let scalar_class = match SeriesModel::from_value(scalar).await? {
        SeriesModel::Tf(_) => ModelArrayClass::Tf,
        SeriesModel::Ss(_) => ModelArrayClass::Ss,
        SeriesModel::Gain(gain) => {
            if !gain.re.is_finite() || !gain.im.is_finite() {
                return Err(argument_error("series scalar gain must be finite"));
            }
            if array_class == ModelArrayClass::Ss && gain.im != 0.0 {
                return Err(argument_error(
                    "complex scalar gain cannot be represented by a real ss model",
                ));
            }
            array_class
        }
    };
    empty_model_array(
        array.shape().to_vec(),
        ModelArrayClass::compose(array_class, scalar_class),
    )
}

#[runtime_builtin(
    name = "series",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::series",
    context = true
)]
async fn series_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    second: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let runtime = runtime.ok_or_else(|| argument_error("series requires a runtime context"))?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(argument_error("series returns at most one model"));
    }
    admitted_model_kind(&first)?;
    admitted_model_kind(&second)?;
    if let (Value::ObjectArray(first), Value::ObjectArray(second)) = (&first, &second) {
        if first.shape() != second.shape() {
            return Err(argument_error(
                "series model arrays must have the same shape",
            ));
        }
    }
    let view = rest.view();
    let (named, channels) = match view.len() {
        0 => (false, None),
        1 if tensor::value_to_string(view.first().expect("one argument"))
            .is_some_and(|text| text.eq_ignore_ascii_case("name")) =>
        {
            (true, None)
        }
        2 => (
            false,
            Some((
                channel_indices(view.get(0).expect("two arguments")).await?,
                channel_indices(view.get(1).expect("two arguments")).await?,
            )),
        ),
        1 => {
            return Err(argument_error(
                "series one-argument selector must be 'name'",
            ))
        }
        _ => {
            return Err(argument_error(
                "series accepts two models and optional channel lists",
            ))
        }
    };
    let model = match (first, second) {
        (Value::ObjectArray(first), Value::ObjectArray(second)) => {
            if first.is_empty() {
                if named || channels.is_some() {
                    return Err(argument_error(
                        "empty model arrays do not retain channel names or counts",
                    ));
                }
                let class = ModelArrayClass::compose(
                    ModelArrayClass::of(&first)?,
                    ModelArrayClass::of(&second)?,
                );
                let empty = empty_model_array(first.shape().to_vec(), class)?;
                return if crate::output_count::current_output_count() == Some(0) {
                    Ok(Value::output_list(Vec::new()))
                } else {
                    Ok(empty)
                };
            }
            let shape = first.shape().to_vec();
            let values = first
                .data()
                .iter()
                .cloned()
                .zip(second.data().iter().cloned());
            compose_array(&runtime, shape, values, named, channels.as_ref()).await?
        }
        (Value::ObjectArray(first), second) => {
            if first.is_empty() {
                empty_array_with_scalar(&first, second, named, channels.as_ref()).await?
            } else {
                let shape = first.shape().to_vec();
                let values = first
                    .data()
                    .iter()
                    .cloned()
                    .map(|entry| (entry, second.clone()));
                compose_array(&runtime, shape, values, named, channels.as_ref()).await?
            }
        }
        (first, Value::ObjectArray(second)) => {
            if second.is_empty() {
                empty_array_with_scalar(&second, first, named, channels.as_ref()).await?
            } else {
                let shape = second.shape().to_vec();
                let values = second
                    .data()
                    .iter()
                    .cloned()
                    .map(|entry| (first.clone(), entry));
                compose_array(&runtime, shape, values, named, channels.as_ref()).await?
            }
        }
        (first, second) => compose_pair(&runtime, first, second, named, channels.as_ref()).await?,
    };
    if crate::output_count::current_output_count() == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(model)
    }
}

async fn compose_pair(
    runtime: &crate::context::RuntimeContext,
    first: Value,
    second: Value,
    named: bool,
    channels: Option<&(Vec<usize>, Vec<usize>)>,
) -> BuiltinResult<Value> {
    let first = SeriesModel::from_value(first).await?;
    let second = SeriesModel::from_value(second).await?;
    if named {
        SeriesModel::compose_by_name(runtime, first, second).await
    } else {
        SeriesModel::compose(
            runtime,
            first,
            second,
            channels.map(|(outputs, inputs)| (outputs.as_slice(), inputs.as_slice())),
        )
        .await
    }
}

async fn compose_array<I>(
    runtime: &crate::context::RuntimeContext,
    shape: Vec<usize>,
    values: I,
    named: bool,
    channels: Option<&(Vec<usize>, Vec<usize>)>,
) -> BuiltinResult<Value>
where
    I: IntoIterator<Item = (Value, Value)>,
{
    let mut outputs = Vec::new();
    for (first, second) in values {
        match compose_pair(runtime, first, second, named, channels).await? {
            Value::Object(object) => outputs.push(object),
            _ => return Err(argument_error("series produced a non-model array element")),
        }
    }
    let class = outputs
        .first()
        .map(|object| object.class_name().clone())
        .ok_or_else(|| argument_error("series cannot infer the class of an empty model array"))?;
    ObjectArray::from_objects(class, outputs, shape)
        .map(Value::ObjectArray)
        .map_err(argument_error)
}

async fn channel_indices(value: &Value) -> BuiltinResult<Vec<usize>> {
    let host = dispatcher::gather_if_needed_async(value).await?;
    let values = match &host {
        Value::Tensor(tensor)
            if tensor.shape().len() <= 2 && (tensor.rows() == 1 || tensor.cols() == 1) =>
        {
            if let Some(storage) = tensor.integer_storage() {
                (0..tensor.len())
                    .map(|index| Value::Int(storage.value_at(index).expect("index in tensor")))
                    .collect::<Vec<_>>()
            } else {
                tensor::tensor_values_f64_cow(tensor)
                    .iter()
                    .copied()
                    .map(Value::Num)
                    .collect::<Vec<_>>()
            }
        }
        Value::Num(_) | Value::Int(_) => vec![host],
        _ => return Err(argument_error("channel lists must be numeric vectors")),
    };
    let mut indices = Vec::with_capacity(values.len());
    for value in &values {
        let one_based = tensor::dimension_from_value_async(value, "channel", false)
            .await
            .map_err(argument_error)?
            .ok_or_else(|| argument_error("channel indices must be positive integers"))?;
        let zero_based = one_based - 1;
        if indices.contains(&zero_based) {
            return Err(argument_error("channel indices must be distinct"));
        }
        indices.push(zero_based);
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn runtime() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
    }

    async fn series_builtin(first: Value, second: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let runtime = runtime();
        let rest = crate::call::BuiltinArguments::from_borrowed(&rest, Some(&runtime))?;
        super::series_builtin(Some(runtime), first, second, rest).await
    }

    #[test]
    fn array_cascade_preserves_column_major_model_shape() {
        let result = block_on(compose_array(
            &runtime(),
            vec![1, 2],
            [
                (Value::Num(2.0), Value::Num(3.0)),
                (Value::Num(4.0), Value::Num(5.0)),
            ],
            false,
            None,
        ))
        .unwrap();
        let Value::ObjectArray(models) = result else {
            panic!("series should retain an object array")
        };
        assert_eq!(models.shape(), &[1, 2]);
        assert_eq!(models.len(), 2);
        assert!(models
            .data()
            .iter()
            .all(|model| matches!(model, Value::Object(_))));
    }

    #[test]
    fn empty_model_arrays_preserve_typed_shape_without_guessing_channels() {
        let first =
            Value::ObjectArray(ObjectArray::empty(TF_CLASS.to_owned(), vec![0, 2]).unwrap());
        let second =
            Value::ObjectArray(ObjectArray::empty(SS_CLASS.to_owned(), vec![0, 2]).unwrap());
        let Value::ObjectArray(result) =
            block_on(series_builtin(first, second, Vec::new())).unwrap()
        else {
            panic!("series must preserve the empty model array")
        };
        assert_eq!(result.shape(), &[0, 2]);
        assert!(result.class_name().is(SS_CLASS));
        assert!(result.has_unselected_provenance());
    }

    #[test]
    fn empty_array_scalar_composition_preserves_class_in_either_order() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tf = crate::builtins::control::tf_model::TfModel::scalar(
            num_complex::Complex64::new(2.0, 0.0),
            crate::builtins::control::tf_model::TfOptions::default(),
        )
        .unwrap()
        .to_value(NAME)
        .unwrap();
        let ss_runtime = runtime();
        let empty_arguments: [Value; 0] = [];
        let ss_arguments =
            crate::call::BuiltinArguments::from_borrowed(&empty_arguments, Some(&ss_runtime))
                .expect("empty arguments");
        let ss = block_on(crate::builtins::control::ss::ss_builtin(
            Some(ss_runtime),
            Value::Num(-1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
            ss_arguments,
        ))
        .unwrap();
        for (array_class, scalar, result_class) in [
            (TF_CLASS, Value::Num(3.0), TF_CLASS),
            (SS_CLASS, Value::Num(3.0), SS_CLASS),
            (
                TF_CLASS,
                Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
                TF_CLASS,
            ),
            (TF_CLASS, ss, SS_CLASS),
            (SS_CLASS, tf, SS_CLASS),
        ] {
            for reverse in [false, true] {
                let empty = Value::ObjectArray(
                    ObjectArray::empty(array_class.to_owned(), vec![0, 2]).unwrap(),
                );
                let (first, second) = if reverse {
                    (scalar.clone(), empty)
                } else {
                    (empty, scalar.clone())
                };
                let Value::ObjectArray(result) =
                    block_on(series_builtin(first, second, Vec::new())).unwrap()
                else {
                    panic!("series should return a typed empty array")
                };
                assert_eq!(result.shape(), &[0, 2]);
                assert!(result.class_name().is(result_class));
                assert!(result.has_unselected_provenance());
            }
        }
    }

    #[test]
    fn empty_array_does_not_ignore_invalid_scalar_or_channel_selector() {
        let empty =
            || Value::ObjectArray(ObjectArray::empty(TF_CLASS.to_owned(), vec![0, 2]).unwrap());
        assert!(block_on(series_builtin(empty(), Value::Num(f64::NAN), Vec::new())).is_err());
        assert!(block_on(series_builtin(Value::Bool(true), empty(), Vec::new())).is_err());
        assert!(block_on(series_builtin(
            empty(),
            Value::Num(2.0),
            vec![Value::Num(1.0), Value::Num(1.0)],
        ))
        .is_err());
    }
}
