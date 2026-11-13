//! Real state-space realization and series composition for control models.
use crate::builtins::common::tensor;
use crate::builtins::control::ss::ss_builtin;
use crate::builtins::control::tf_model::{
    control_error, gather_ss_property_values, scalar_complex, scalar_f64, TfModel, TfOptions,
    SS_CLASS, TF_CLASS,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use nalgebra::DMatrix;
use runmat_value::{CellArray, CharArray, ObjectInstance, Tensor, Value};

const NAME: &str = "series";
const REQUIRED: [&str; 9] = [
    "A",
    "B",
    "C",
    "D",
    "Ts",
    "InputDelay",
    "OutputDelay",
    "InputName",
    "OutputName",
];

fn invalid(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:series:InvalidModel", message)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:series:UnsupportedModel", message)
}

pub(crate) enum SeriesModel {
    Tf(TfModel),
    Ss(RealStateSpace),
    Gain(num_complex::Complex64),
}

pub(crate) struct RealStateSpace {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    d: DMatrix<f64>,
    sample_time: f64,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl SeriesModel {
    pub(crate) async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Object(object) if object.is_class(TF_CLASS) => {
                TfModel::from_value_async(Value::Object(object), NAME)
                    .await
                    .map(Self::Tf)
            }
            Value::Object(object) if object.is_class(SS_CLASS) => {
                RealStateSpace::from_object(object).await.map(Self::Ss)
            }
            Value::Num(value) => Ok(Self::Gain(num_complex::Complex64::new(value, 0.0))),
            Value::Complex(re, im) => Ok(Self::Gain(num_complex::Complex64::new(re, im))),
            Value::Int(value) => {
                let source = Value::Int(value);
                crate::compatibility::ensure_builtin_extension_enabled(
                    &runmat_builtins::INTEGER_COEFFICIENT_EXTENSION,
                    NAME,
                )?;
                // Arithmetic and feedback use this same scalar boundary: a
                // wide integer gain is rounded to the model's binary64 class.
                // Coefficient constructors have a separate exactness policy.
                Ok(Self::Gain(scalar_complex(&source, NAME)?))
            }
            _ => Err(invalid("series expects tf or ss models or a scalar gain")),
        }
    }

    pub(crate) async fn compose(
        runtime: &RuntimeContext,
        first: Self,
        second: Self,
        channels: Option<(&[usize], &[usize])>,
    ) -> BuiltinResult<Value> {
        // A scalar real gain is a channelwise scale of a state-space system,
        // including MIMO systems. It must not be materialized as a 1-by-1 tf
        // and then rejected by the ordinary port-count equality check.
        match (first, second, channels) {
            (Self::Gain(gain), Self::Ss(system), None) => {
                system.scale_gain(gain, true)?.into_value(runtime).await
            }
            (Self::Ss(system), Self::Gain(gain), None) => {
                system.scale_gain(gain, false)?.into_value(runtime).await
            }
            (first, second, channels) => {
                Self::compose_models(runtime, first, second, channels).await
            }
        }
    }

    async fn compose_models(
        runtime: &RuntimeContext,
        first: Self,
        second: Self,
        channels: Option<(&[usize], &[usize])>,
    ) -> BuiltinResult<Value> {
        let (first, second) = Self::materialize_gains(first, second)?;
        match (first, second) {
            (Self::Tf(first), Self::Tf(second))
                if channels.is_none_or(|(outputs, inputs)| outputs == [0] && inputs == [0]) =>
            {
                // The first system is upstream: series(sys1,sys2) = sys2*sys1.
                second.mul(&first)?.to_value(NAME)
            }
            (first, second) => {
                let first = first.into_state_space()?;
                let second = second.into_state_space()?;
                first.compose(&second, channels)?.into_value(runtime).await
            }
        }
    }

    pub(crate) async fn compose_by_name(
        runtime: &RuntimeContext,
        first: Self,
        second: Self,
    ) -> BuiltinResult<Value> {
        let (first, second) = Self::materialize_gains(first, second)?;
        let first = first.into_state_space()?;
        let second = second.into_state_space()?;
        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        for (output, name) in first.output_names.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            if first
                .output_names
                .iter()
                .filter(|candidate| *candidate == name)
                .count()
                != 1
                || second
                    .input_names
                    .iter()
                    .filter(|candidate| *candidate == name)
                    .count()
                    > 1
            {
                return Err(invalid("series named channels must be unique"));
            }
            if let Some(input) = second
                .input_names
                .iter()
                .position(|candidate| candidate == name)
            {
                outputs.push(output);
                inputs.push(input);
            }
        }
        if outputs.is_empty() {
            return Err(invalid("series models have no matching named channels"));
        }
        first
            .compose(&second, Some((&outputs, &inputs)))?
            .into_value(runtime)
            .await
    }

    fn into_state_space(self) -> BuiltinResult<RealStateSpace> {
        match self {
            Self::Ss(model) => Ok(model),
            Self::Tf(model) => RealStateSpace::from_tf(model),
            Self::Gain(_) => unreachable!("gains materialized at series boundary"),
        }
    }

    fn materialize_gains(first: Self, second: Self) -> BuiltinResult<(Self, Self)> {
        let first = match first {
            Self::Gain(gain) => Self::gain_for(gain, &second)?,
            other => other,
        };
        let second = match second {
            Self::Gain(gain) => Self::gain_for(gain, &first)?,
            other => other,
        };
        Ok((first, second))
    }

    fn gain_for(gain: num_complex::Complex64, peer: &Self) -> BuiltinResult<Self> {
        let (variable, sample_time) = match peer {
            Self::Tf(model) => (model.variable.as_str().to_owned(), model.sample_time()),
            Self::Ss(model) if model.sample_time > 0.0 => ("z".to_owned(), model.sample_time),
            Self::Ss(_) | Self::Gain(_) => ("s".to_owned(), 0.0),
        };
        TfModel::scalar(
            gain,
            TfOptions {
                variable,
                sample_time,
            },
        )
        .map(Self::Tf)
    }
}

impl RealStateSpace {
    fn scale_gain(mut self, gain: num_complex::Complex64, upstream: bool) -> BuiltinResult<Self> {
        if gain.im != 0.0 || !gain.re.is_finite() {
            return Err(unsupported(
                "complex or nonfinite scalar gain cannot be represented by a real ss model",
            ));
        }
        if upstream {
            self.b *= gain.re;
        } else {
            self.c *= gain.re;
        }
        self.d *= gain.re;
        if self
            .b
            .iter()
            .chain(self.c.iter())
            .chain(self.d.iter())
            .any(|value| !value.is_finite())
        {
            return Err(invalid("series scalar gain produced a nonfinite matrix"));
        }
        Ok(self)
    }

    async fn from_object(object: ObjectInstance) -> BuiltinResult<Self> {
        for name in ["InputDelay", "OutputDelay"] {
            if let Some(value) = object.properties().get(name) {
                if !matches!(value, Value::GpuTensor(_)) && !zero_delay(value, name)? {
                    return Err(unsupported("series does not support nonzero model delays"));
                }
            }
        }
        let [a, b, c, d, sample_time, input_delay, output_delay, input_names, output_names] =
            gather_ss_property_values(&object, REQUIRED, NAME).await?;
        for (name, value) in [("InputDelay", &input_delay), ("OutputDelay", &output_delay)] {
            if !zero_delay(value, name)? {
                return Err(unsupported("series does not support nonzero model delays"));
            }
        }
        let a = matrix_value(&a, "A")?;
        let b = matrix_value(&b, "B")?;
        let c = matrix_value(&c, "C")?;
        let d = matrix_value(&d, "D")?;
        let input_names = names_value(&input_names, "InputName", b.ncols())?;
        let output_names = names_value(&output_names, "OutputName", c.nrows())?;
        let model = Self {
            a,
            b,
            c,
            d,
            sample_time: scalar_f64(&sample_time, "Ts", NAME)?,
            input_names,
            output_names,
        };
        if !model.sample_time.is_finite() || model.sample_time < 0.0 {
            return Err(invalid("sample time must be finite and nonnegative"));
        }
        model.validate()?;
        Ok(model)
    }

    fn from_tf(model: TfModel) -> BuiltinResult<Self> {
        if model.input_delay != 0.0 || model.output_delay != 0.0 {
            return Err(unsupported("series does not support nonzero model delays"));
        }
        let real = model.to_real(NAME)?;
        let (mut num, den) = real.normalized(NAME)?;
        let n = den.len().saturating_sub(1);
        if num.len() > n + 1 {
            return Err(unsupported(
                "an improper tf model has no finite state-space realization",
            ));
        }
        while num.len() < n + 1 {
            num.insert(0, 0.0);
        }
        let direct = num[0];
        let mut a = DMatrix::zeros(n, n);
        for col in 0..n {
            a[(0, col)] = -den[col + 1];
        }
        for row in 1..n {
            a[(row, row - 1)] = 1.0;
        }
        let mut b = DMatrix::zeros(n, 1);
        if n > 0 {
            b[(0, 0)] = 1.0;
        }
        let c = DMatrix::from_row_slice(
            1,
            n,
            &(0..n)
                .map(|i| num[i + 1] - direct * den[i + 1])
                .collect::<Vec<_>>(),
        );
        Ok(Self {
            a,
            b,
            c,
            d: DMatrix::from_element(1, 1, direct),
            sample_time: model.sample_time(),
            input_names: vec![String::new()],
            output_names: vec![String::new()],
        })
    }

    fn validate(&self) -> BuiltinResult<()> {
        let n = self.a.nrows();
        if self.a.ncols() != n
            || self.b.nrows() != n
            || self.c.ncols() != n
            || self.d.nrows() != self.c.nrows()
            || self.d.ncols() != self.b.ncols()
        {
            return Err(invalid("inconsistent state-space matrix dimensions"));
        }
        Ok(())
    }

    fn compose(
        self,
        downstream: &Self,
        channels: Option<(&[usize], &[usize])>,
    ) -> BuiltinResult<Self> {
        if self.sample_time != downstream.sample_time {
            return Err(invalid("series models must have the same sample time"));
        }
        let (outputs, inputs) = match channels {
            Some((outputs, inputs)) if outputs.len() == inputs.len() => {
                (outputs.to_vec(), inputs.to_vec())
            }
            Some(_) => {
                return Err(invalid(
                    "connected output and input index vectors must have the same length",
                ))
            }
            None if self.c.nrows() == downstream.b.ncols() => {
                let all = (0..self.c.nrows()).collect::<Vec<_>>();
                (all.clone(), all)
            }
            None => return Err(invalid("upstream outputs must match downstream inputs")),
        };
        if outputs.iter().any(|&index| index >= self.c.nrows())
            || inputs.iter().any(|&index| index >= downstream.b.ncols())
        {
            return Err(invalid("selected series channel index is out of range"));
        }
        let n1 = self.a.nrows();
        let n2 = downstream.a.nrows();
        let inputs1 = self.b.ncols();
        let outputs2 = downstream.c.nrows();
        let mut c1 = DMatrix::zeros(outputs.len(), n1);
        let mut d1 = DMatrix::zeros(outputs.len(), inputs1);
        let mut b2 = DMatrix::zeros(n2, inputs.len());
        let mut d2 = DMatrix::zeros(outputs2, inputs.len());
        for (slot, (&out, &input)) in outputs.iter().zip(&inputs).enumerate() {
            c1.row_mut(slot).copy_from(&self.c.row(out));
            d1.row_mut(slot).copy_from(&self.d.row(out));
            b2.column_mut(slot).copy_from(&downstream.b.column(input));
            d2.column_mut(slot).copy_from(&downstream.d.column(input));
        }
        let mut a = DMatrix::zeros(n1 + n2, n1 + n2);
        a.view_mut((0, 0), (n1, n1)).copy_from(&self.a);
        a.view_mut((n1, 0), (n2, n1)).copy_from(&(&b2 * &c1));
        a.view_mut((n1, n1), (n2, n2)).copy_from(&downstream.a);
        let mut b = DMatrix::zeros(n1 + n2, inputs1);
        b.rows_mut(0, n1).copy_from(&self.b);
        b.rows_mut(n1, n2).copy_from(&(&b2 * &d1));
        let mut c = DMatrix::zeros(outputs2, n1 + n2);
        c.columns_mut(0, n1).copy_from(&(&d2 * &c1));
        c.columns_mut(n1, n2).copy_from(&downstream.c);
        let d = &d2 * &d1;
        Ok(Self {
            a,
            b,
            c,
            d,
            sample_time: self.sample_time,
            input_names: self.input_names,
            output_names: downstream.output_names.clone(),
        })
    }

    async fn into_value(self, runtime: &RuntimeContext) -> BuiltinResult<Value> {
        let sample_time = [Value::Num(self.sample_time)];
        let arguments = crate::call::BuiltinArguments::from_borrowed(&sample_time, Some(runtime))?;
        let mut model = ss_builtin(
            Some(runtime.clone()),
            matrix_output_value(self.a)?,
            matrix_output_value(self.b)?,
            matrix_output_value(self.c)?,
            matrix_output_value(self.d)?,
            arguments,
        )
        .await?;
        if let Value::Object(object) = &mut model {
            let input_names = name_cells(self.input_names)?;
            let output_names = name_cells(self.output_names)?;
            crate::execution::object::replacement::replace_properties(
                Some(runtime),
                object,
                [("InputName", &input_names), ("OutputName", &output_names)].into_iter(),
                |message, source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        Some(runtime),
                        "RunMat:series:Allocation",
                        message,
                        source,
                    )
                },
                invalid,
            )?;
        }
        Ok(model)
    }
}

fn zero_delay(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Tensor(tensor) => Ok(tensor::tensor_values_f64_cow(tensor)
            .iter()
            .all(|value| *value == 0.0)),
        _ => Ok(scalar_f64(value, name, NAME)? == 0.0),
    }
}

fn matrix_value(value: &Value, name: &str) -> BuiltinResult<DMatrix<f64>> {
    match value {
        Value::Tensor(tensor)
            if tensor.shape().len() <= 2 && tensor.integer_storage().is_none() =>
        {
            let (rows, cols) = if tensor.shape().len() == 2 {
                (tensor.shape()[0], tensor.shape()[1])
            } else {
                (1, tensor.len())
            };
            let values = tensor::tensor_values_f64_cow(tensor);
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid(format!("{name} must be finite")));
            }
            Ok(DMatrix::from_column_slice(rows, cols, &values))
        }
        Value::Num(value) if value.is_finite() => Ok(DMatrix::from_element(1, 1, *value)),
        _ => Err(invalid(format!("{name} must be a finite real matrix"))),
    }
}

fn matrix_output_value(matrix: DMatrix<f64>) -> BuiltinResult<Value> {
    Tensor::new(
        matrix.as_slice().to_vec(),
        vec![matrix.nrows(), matrix.ncols()],
    )
    .map(Value::Tensor)
    .map_err(|error| control_error(NAME, "RunMat:series:Internal", error))
}

#[cfg(test)]
fn names(object: &ObjectInstance, property: &str, expected: usize) -> BuiltinResult<Vec<String>> {
    names_value(
        object
            .properties()
            .get(property)
            .ok_or_else(|| invalid(format!("missing {property}")))?,
        property,
        expected,
    )
}

fn names_value(value: &Value, property: &str, expected: usize) -> BuiltinResult<Vec<String>> {
    let names = match value {
        Value::Cell(cells) if cells.shape.len() <= 2 && (cells.rows == 1 || cells.cols == 1) => {
            cells
                .data()
                .iter()
                .map(|value| {
                    tensor::value_to_string(value).ok_or_else(|| {
                        invalid(format!("{property} contains a non-text channel name"))
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?
        }
        Value::StringArray(strings)
            if strings.shape.len() <= 2 && (strings.rows == 1 || strings.cols == 1) =>
        {
            strings
                .data()
                .iter()
                .map(|text| {
                    text.as_str().map(str::to_owned).ok_or_else(|| {
                        invalid(format!("{property} contains a non-text channel name"))
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?
        }
        _ => {
            return Err(invalid(format!(
                "{property} must be a cell or string vector of channel names"
            )))
        }
    };
    if names.len() != expected {
        return Err(invalid(format!(
            "{property} must contain {expected} channel names"
        )));
    }
    Ok(names)
}

fn name_cells(names: Vec<String>) -> BuiltinResult<Value> {
    let count = names.len();
    let values = names
        .into_iter()
        .map(|name| Value::CharArray(CharArray::new_row(&name)))
        .collect();
    CellArray::new(values, count, 1)
        .map(Value::Cell)
        .map_err(|error| control_error(NAME, "RunMat:series:Internal", error))
}

#[cfg(test)]
mod tests;
