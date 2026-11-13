//! State-space to SISO transfer-function conversion for control analysis.
use crate::builtins::common::tensor;
use crate::builtins::control::tf_model::{
    control_error, gather_ss_property_values, internal_identifier, invalid_model_identifier,
    scalar_f64, unsupported_model_identifier, TfModel, TfOptions,
};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::{BuiltinResult, RuntimeError};
use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{ObjectInstance, Value};

#[path = "ss_conversion/workspace.rs"]
mod workspace;
#[path = "ss_conversion/allocation.rs"]
mod allocation;
pub(crate) use allocation::SsPolynomialGeometry;
use workspace::{CharacteristicWorkspace, ComplexMatrixRead, ComplexScratch, NumeratorWorkspace};

struct HostMatrixView<'a> {
    value: &'a Value,
    rows: usize,
    cols: usize,
}

impl HostMatrixView<'_> {
    fn new<'a>(value: &'a Value, name: &str) -> ConversionResult<HostMatrixView<'a>> {
        let shape = match value {
            Value::Tensor(tensor) => tensor.shape(),
            Value::ComplexTensor(tensor) => tensor.shape(),
            Value::LogicalArray(array) => &array.shape,
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => &[1, 1],
            _ => {
                return Err(conversion_error(
                    format!("ss {name} must be a numeric host matrix"),
                    ConversionErrorKind::InvalidModel,
                ));
            }
        };
        ensure_matrix_shape(name, shape)?;
        let (rows, cols) = rows_cols(shape);
        Ok(HostMatrixView { value, rows, cols })
    }
}

impl ComplexMatrixRead for HostMatrixView<'_> {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
    fn at(&self, row: usize, col: usize) -> Complex64 {
        let index = row + col * self.rows;
        match self.value {
            Value::Tensor(tensor) => Complex64::new(
                tensor
                    .numeric_value_at(index)
                    .expect("matrix index")
                    .materialize_f64(),
                0.0,
            ),
            Value::ComplexTensor(tensor) => {
                let (re, im) = tensor.numeric_value_at(index).expect("matrix index");
                Complex64::new(re.materialize_f64(), im.materialize_f64())
            }
            Value::LogicalArray(array) => {
                Complex64::new(if array.data[index] == 0 { 0.0 } else { 1.0 }, 0.0)
            }
            Value::Num(value) => Complex64::new(*value, 0.0),
            Value::Int(value) => Complex64::new(value.to_f64(), 0.0),
            Value::Bool(value) => Complex64::new(if *value { 1.0 } else { 0.0 }, 0.0),
            Value::Complex(re, im) => Complex64::new(*re, *im),
            _ => unreachable!("validated matrix carrier"),
        }
    }
}

pub(crate) fn allocation_error(
    builtin: &'static str,
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), builtin, source)
}

pub(crate) struct SsPolynomialBuffers {
    first: Option<ComplexScratch>,
    second: Option<ComplexScratch>,
}

pub(crate) fn preallocate_polynomial_buffers(
    receipt: Option<&HostAllocationReceipt>,
    runtime: Option<&RuntimeContext>,
    states: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<SsPolynomialBuffers>> {
    let Some((receipt, runtime)) = receipt.zip(runtime) else {
        return Ok(None);
    };
    if states == 0 {
        return Ok(None);
    }
    let square = states
        .checked_mul(states)
        .ok_or_else(|| allocation_error(builtin, Some(runtime), HostAllocationError::Overflow))?;
    let first = receipt
        .temporary_complex(runtime, square)
        .map_err(|error| allocation_error(builtin, Some(runtime), error))?;
    let second = receipt
        .temporary_complex(runtime, square)
        .map_err(|error| allocation_error(builtin, Some(runtime), error))?;
    Ok(Some(SsPolynomialBuffers {
        first: Some(ComplexScratch::Managed(first)),
        second: Some(ComplexScratch::Managed(second)),
    }))
}

#[derive(Debug, Clone, Copy)]
enum ConversionErrorKind {
    InvalidModel,
    UnsupportedModel,
    Internal,
}
#[derive(Debug)]
pub(crate) struct ConversionError {
    kind: ConversionErrorKind,
    message: String,
}
type ConversionResult<T> = Result<T, ConversionError>;

fn conversion_error(message: impl Into<String>, kind: ConversionErrorKind) -> ConversionError {
    ConversionError {
        kind,
        message: message.into(),
    }
}
impl ConversionError {
    fn into_runtime(self, builtin: &'static str) -> RuntimeError {
        let identifier = match self.kind {
            ConversionErrorKind::InvalidModel => invalid_model_identifier(builtin),
            ConversionErrorKind::UnsupportedModel => unsupported_model_identifier(builtin),
            ConversionErrorKind::Internal => internal_identifier(builtin),
        };
        control_error(builtin, identifier, format!("{builtin}: {}", self.message))
    }
}

pub(crate) async fn ss_object_to_tf_async(
    object: ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<TfModel> {
    const REQUIRED: [&str; 7] = ["A", "B", "C", "D", "Ts", "InputDelay", "OutputDelay"];
    for name in REQUIRED {
        if !object.properties().contains_key(name) {
            return Err(conversion_error(
                format!("model object is missing {name} property"),
                ConversionErrorKind::InvalidModel,
            )
            .into_runtime(builtin));
        }
    }
    // An unsupported host delay is a model-admission error, not a reason to
    // download otherwise valid resident state matrices.
    for name in ["InputDelay", "OutputDelay"] {
        if let Some(value) = object.properties().get(name) {
            if !matches!(value, Value::GpuTensor(_)) {
                ensure_zero_delay_value(value, name)
                    .map_err(|error| error.into_runtime(builtin))?;
            }
        }
    }
    let gathered = gather_ss_property_values(&object, REQUIRED, builtin).await?;
    ss_values_to_tf(&gathered, builtin).map_err(|error| error.into_runtime(builtin))
}

fn ss_values_to_tf(values: &[Value; 7], builtin: &'static str) -> ConversionResult<TfModel> {
    let [a_value, b_value, c_value, d_value, sample_value, input_delay, output_delay] = values;
    let a = matrix_value(a_value, "A")?;
    let b = matrix_value(b_value, "B")?;
    let c = matrix_value(c_value, "C")?;
    let d = matrix_value(d_value, "D")?;
    let sample_time = scalar_value(sample_value, "Ts", builtin)?;
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(conversion_error(
            "ss sample time must be a finite nonnegative scalar",
            ConversionErrorKind::InvalidModel,
        ));
    }
    ensure_zero_delay_value(input_delay, "InputDelay")?;
    ensure_zero_delay_value(output_delay, "OutputDelay")?;
    validate_ss_dimensions(&a, &b, &c, &d)?;

    let denominator = characteristic_polynomial_from_matrix(&a)?;
    let numerator = state_space_numerator(&a, &b, &c, d[(0, 0)], &denominator)?;
    let numerator = trim_leading_complex_zeros(numerator);
    let denominator = trim_leading_complex_zeros(denominator);
    if denominator.is_empty() || denominator[0] == Complex64::new(0.0, 0.0) {
        return Err(conversion_error(
            "ss model produced an invalid denominator polynomial",
            ConversionErrorKind::Internal,
        ));
    }
    ensure_finite_coefficients("Numerator", &numerator)?;
    ensure_finite_coefficients("Denominator", &denominator)?;

    TfModel::new(
        numerator,
        denominator,
        TfOptions {
            variable: if sample_time != 0.0 { "z" } else { "s" }.to_string(),
            sample_time,
        },
    )
    .map_err(|error| {
        conversion_error(
            format!("ss model produced an invalid transfer function: {error}"),
            ConversionErrorKind::Internal,
        )
    })
}

#[cfg(test)]
fn ss_object_to_tf(object: &ObjectInstance, builtin: &'static str) -> ConversionResult<TfModel> {
    let values = [
        property(object, "A")?.clone(),
        property(object, "B")?.clone(),
        property(object, "C")?.clone(),
        property(object, "D")?.clone(),
        property(object, "Ts")?.clone(),
        property(object, "InputDelay")?.clone(),
        property(object, "OutputDelay")?.clone(),
    ];
    ss_values_to_tf(&values, builtin)
}

/// MIMO coefficient grid in column-major (output, input) order. This uses
/// the same characteristic and Markov coefficient algorithm as the SISO path.
pub(crate) async fn ss_object_coefficient_grid_async(
    object: ObjectInstance,
    builtin: &'static str,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<(Vec<Vec<Complex64>>, Vec<Complex64>, usize, usize, f64)> {
    const REQUIRED: [&str; 7] = ["A", "B", "C", "D", "Ts", "InputDelay", "OutputDelay"];
    for name in REQUIRED {
        if !object.properties().contains_key(name) {
            return Err(conversion_error(
                format!("model object is missing {name} property"),
                ConversionErrorKind::InvalidModel,
            )
            .into_runtime(builtin));
        }
    }
    for name in ["InputDelay", "OutputDelay"] {
        if let Some(value) = object.properties().get(name) {
            if !matches!(value, Value::GpuTensor(_)) {
                ensure_zero_delay_value(value, name)
                    .map_err(|error| error.into_runtime(builtin))?;
            }
        }
    }
    let geometry = SsPolynomialGeometry::from_values(
        [
            property(&object, "A").map_err(|error| error.into_runtime(builtin))?,
            property(&object, "B").map_err(|error| error.into_runtime(builtin))?,
            property(&object, "C").map_err(|error| error.into_runtime(builtin))?,
            property(&object, "D").map_err(|error| error.into_runtime(builtin))?,
        ],
        builtin,
    )?;
    let plan = geometry
        .coefficient_grid_plan()
        .map_err(|error| allocation_error(builtin, runtime, error))?;
    let receipt = HostAllocationReceipt::admit(runtime, plan)
        .map_err(|error| allocation_error(builtin, runtime, error))?;
    let mut buffers =
        preallocate_polynomial_buffers(receipt.as_ref(), runtime, geometry.states, builtin)?;
    let values = gather_ss_property_values(&object, REQUIRED, builtin).await?;
    let result = (|| -> ConversionResult<_> {
        let [a, b, c, d, sample, input_delay, output_delay] = &values;
        ensure_zero_delay_value(input_delay, "InputDelay")?;
        ensure_zero_delay_value(output_delay, "OutputDelay")?;
        let a = HostMatrixView::new(a, "A")?;
        let b = HostMatrixView::new(b, "B")?;
        let c = HostMatrixView::new(c, "C")?;
        let d = HostMatrixView::new(d, "D")?;
        validate_general_dimensions(&a, &b, &c, &d)?;
        let sample_time = scalar_value(sample, "Ts", builtin)?;
        if sample_time < 0.0 {
            return Err(conversion_error(
                "ss sample time must be nonnegative",
                ConversionErrorKind::InvalidModel,
            ));
        }
        Ok((a, b, c, d, sample_time))
    })();
    let (a, b, c, d, sample_time) = result.map_err(|error| error.into_runtime(builtin))?;
    let mut cooperation = PolynomialCooperation::new(runtime);
    validate_host_matrices(
        [("A", &a), ("B", &b), ("C", &c), ("D", &d)],
        &mut cooperation,
        builtin,
    )
    .await?;
    let denominator =
        characteristic_polynomial_cooperative(&a, &mut cooperation, builtin, buffers.as_mut())
            .await?;
    let count = c.nrows().checked_mul(b.ncols()).ok_or_else(|| {
        conversion_error(
            "coefficient grid size overflows",
            ConversionErrorKind::InvalidModel,
        )
        .into_runtime(builtin)
    })?;
    let mut numerators = Vec::new();
    numerators.try_reserve_exact(count).map_err(|_| {
        conversion_error(
            "coefficient grid exceeds available host memory",
            ConversionErrorKind::Internal,
        )
        .into_runtime(builtin)
    })?;
    for input in 0..b.ncols() {
        for output in 0..c.nrows() {
            let coefficients = state_space_numerator_cooperative(
                &a,
                &b,
                input,
                &c,
                output,
                d.at(output, input),
                &denominator,
                &mut cooperation,
                builtin,
                buffers.as_mut(),
            )
            .await?;
            numerators.push(
                if coefficients
                    .iter()
                    .all(|value| *value == Complex64::new(0.0, 0.0))
                {
                    vec![Complex64::new(0.0, 0.0)]
                } else {
                    trim_leading_complex_zeros(coefficients)
                },
            );
        }
    }
    Ok((
        numerators,
        trim_leading_complex_zeros(denominator),
        c.nrows(),
        b.ncols(),
        sample_time,
    ))
}

fn property<'a>(object: &'a ObjectInstance, name: &str) -> ConversionResult<&'a Value> {
    object.properties().get(name).ok_or_else(|| {
        conversion_error(
            format!("model object is missing {name} property"),
            ConversionErrorKind::InvalidModel,
        )
    })
}

#[cfg(test)]
fn matrix_property(object: &ObjectInstance, name: &str) -> ConversionResult<DMatrix<Complex64>> {
    matrix_value(property(object, name)?, name)
}

fn matrix_value(value: &Value, name: &str) -> ConversionResult<DMatrix<Complex64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_matrix_shape(name, tensor.shape())?;
            let data = tensor::tensor_values_f64(tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect::<Vec<_>>();
            ensure_finite_coefficients(name, &data)?;
            let (rows, cols) = rows_cols(tensor.shape());
            Ok(DMatrix::from_vec(rows, cols, data))
        }
        Value::ComplexTensor(tensor) => {
            ensure_matrix_shape(name, tensor.shape())?;
            let data = tensor::complex_tensor_values_complex64(tensor);
            ensure_finite_coefficients(name, &data)?;
            Ok(DMatrix::from_vec(tensor.rows(), tensor.cols(), data))
        }
        Value::LogicalArray(logical) => {
            ensure_matrix_shape(name, &logical.shape)?;
            let (rows, cols) = rows_cols(&logical.shape);
            let data = logical
                .data
                .iter()
                .map(|&value| Complex64::new(if value == 0 { 0.0 } else { 1.0 }, 0.0))
                .collect::<Vec<_>>();
            Ok(DMatrix::from_vec(rows, cols, data))
        }
        Value::Num(n) => scalar_matrix(*n, 0.0, name),
        Value::Int(i) => scalar_matrix(i.to_f64(), 0.0, name),
        Value::Bool(b) => scalar_matrix(if *b { 1.0 } else { 0.0 }, 0.0, name),
        Value::Complex(re, im) => scalar_matrix(*re, *im, name),
        other => Err(conversion_error(
            format!("ss {name} must be a finite numeric matrix, got {other:?}"),
            ConversionErrorKind::InvalidModel,
        )),
    }
}

fn scalar_matrix(re: f64, im: f64, name: &str) -> ConversionResult<DMatrix<Complex64>> {
    let value = Complex64::new(re, im);
    ensure_finite_coefficients(name, &[value])?;
    Ok(DMatrix::from_element(1, 1, value))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> ConversionResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(conversion_error(
            format!("ss {name} must be a 2-D matrix, got shape {shape:?}"),
            ConversionErrorKind::InvalidModel,
        ))
    }
}

fn rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.len() >= 2 {
        (shape[0], shape[1])
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (0, 0)
    }
}

fn scalar_value(value: &Value, name: &str, builtin: &'static str) -> ConversionResult<f64> {
    let value = scalar_f64(value, name, builtin)
        .map_err(|error| conversion_error(error.message(), ConversionErrorKind::InvalidModel))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(conversion_error(
            format!("ss {name} must be finite"),
            ConversionErrorKind::InvalidModel,
        ))
    }
}

fn ensure_zero_delay_value(value: &Value, name: &str) -> ConversionResult<()> {
    let values = numeric_values(value, name)?;
    if values
        .iter()
        .any(|value| *value != Complex64::new(0.0, 0.0))
    {
        return Err(conversion_error(
            "ss models with input or output delays are not supported",
            ConversionErrorKind::UnsupportedModel,
        ));
    }
    Ok(())
}

fn numeric_values(value: &Value, name: &str) -> ConversionResult<Vec<Complex64>> {
    let values = match value {
        Value::Num(n) => vec![Complex64::new(*n, 0.0)],
        Value::Int(i) => vec![Complex64::new(i.to_f64(), 0.0)],
        Value::Bool(b) => vec![Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0)],
        Value::Complex(re, im) => vec![Complex64::new(*re, *im)],
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor)
            .into_iter()
            .map(|re| Complex64::new(re, 0.0))
            .collect(),
        Value::ComplexTensor(tensor) => tensor::complex_tensor_values_complex64(tensor),
        Value::LogicalArray(logical) => logical
            .data
            .iter()
            .map(|&value| Complex64::new(if value == 0 { 0.0 } else { 1.0 }, 0.0))
            .collect(),
        other => {
            return Err(conversion_error(
                format!("ss {name} must be numeric, got {other:?}"),
                ConversionErrorKind::InvalidModel,
            ));
        }
    };
    ensure_finite_coefficients(name, &values)?;
    Ok(values)
}

fn validate_ss_dimensions(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: &DMatrix<Complex64>,
) -> ConversionResult<()> {
    validate_general_dimensions(a, b, c, d)?;
    if b.ncols() != 1 || c.nrows() != 1 || d.nrows() != 1 || d.ncols() != 1 {
        return Err(conversion_error(
            "only SISO ss models are supported",
            ConversionErrorKind::UnsupportedModel,
        ));
    }
    Ok(())
}

fn validate_general_dimensions<M: ComplexMatrixRead>(
    a: &M,
    b: &M,
    c: &M,
    d: &M,
) -> ConversionResult<()> {
    if a.nrows() != a.ncols() {
        return Err(conversion_error(
            format!("ss A must be square, got {}x{}", a.nrows(), a.ncols()),
            ConversionErrorKind::InvalidModel,
        ));
    }
    let states = a.nrows();
    if b.nrows() != states {
        return Err(conversion_error(
            format!(
                "ss B must have {states} rows to match A, got {}x{}",
                b.nrows(),
                b.ncols()
            ),
            ConversionErrorKind::InvalidModel,
        ));
    }
    if c.ncols() != states {
        return Err(conversion_error(
            format!(
                "ss C must have {states} columns to match A, got {}x{}",
                c.nrows(),
                c.ncols()
            ),
            ConversionErrorKind::InvalidModel,
        ));
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(conversion_error(
            format!(
                "ss D must have shape {}x{} to match C outputs and B inputs, got {}x{}",
                c.nrows(),
                b.ncols(),
                d.nrows(),
                d.ncols()
            ),
            ConversionErrorKind::InvalidModel,
        ));
    }
    Ok(())
}

/// One selected input and every output, retaining the full denominator order.
/// `ss2tf` uses this same characteristic/Markov owner as SISO model analysis.
pub(crate) async fn selected_input_polynomials(
    matrices: [&Value; 4],
    input: usize,
    builtin: &'static str,
    runtime: Option<&RuntimeContext>,
    mut buffers: Option<SsPolynomialBuffers>,
) -> BuiltinResult<(Vec<Vec<Complex64>>, Vec<Complex64>)> {
    let [a, b, c, d] = matrices;
    let a = HostMatrixView::new(a, "A").map_err(|error| error.into_runtime(builtin))?;
    let b = HostMatrixView::new(b, "B").map_err(|error| error.into_runtime(builtin))?;
    let c = HostMatrixView::new(c, "C").map_err(|error| error.into_runtime(builtin))?;
    let d = HostMatrixView::new(d, "D").map_err(|error| error.into_runtime(builtin))?;
    validate_general_dimensions(&a, &b, &c, &d).map_err(|error| error.into_runtime(builtin))?;
    if input >= b.ncols() {
        return Err(conversion_error(
            "selected input index is outside the system input range",
            ConversionErrorKind::InvalidModel,
        )
        .into_runtime(builtin));
    }
    let mut cooperation = PolynomialCooperation::new(runtime);
    validate_host_matrices(
        [("A", &a), ("B", &b), ("C", &c), ("D", &d)],
        &mut cooperation,
        builtin,
    )
    .await?;
    let denominator =
        characteristic_polynomial_cooperative(&a, &mut cooperation, builtin, buffers.as_mut())
            .await?;
    let mut numerators = Vec::new();
    numerators.try_reserve_exact(c.nrows()).map_err(|_| {
        conversion_error(
            "numerator grid exceeds available host memory",
            ConversionErrorKind::Internal,
        )
        .into_runtime(builtin)
    })?;
    for output in 0..c.nrows() {
        numerators.push(
            state_space_numerator_cooperative(
                &a,
                &b,
                input,
                &c,
                output,
                d.at(output, input),
                &denominator,
                &mut cooperation,
                builtin,
                buffers.as_mut(),
            )
            .await?,
        );
    }
    Ok((numerators, denominator))
}

struct PolynomialCooperation<'runtime> {
    runtime: Option<&'runtime RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
}

impl<'runtime> PolynomialCooperation<'runtime> {
    fn new(runtime: Option<&'runtime RuntimeContext>) -> Self {
        Self {
            runtime,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
        }
    }
}

async fn completed_polynomial_work(
    cooperation: &mut PolynomialCooperation<'_>,
    units: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let Some(budget) = cooperation.budget.as_mut() else {
        return Ok(());
    };
    let units = u64::try_from(units).map_err(|_| {
        conversion_error(
            "state polynomial work count overflows",
            ConversionErrorKind::InvalidModel,
        )
        .into_runtime(builtin)
    })?;
    budget.checkpoint(units).await.map_err(|source| {
        let mut error = source.into_runtime_error_in_context(cooperation.runtime);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        crate::builtins::common::map_control_flow_with_builtin(error, builtin)
    })
}

fn count_polynomial_work(
    pending: usize,
    units: usize,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    pending.checked_add(units).ok_or_else(|| {
        conversion_error(
            "state polynomial work count overflows",
            ConversionErrorKind::InvalidModel,
        )
        .into_runtime(builtin)
    })
}

async fn validate_host_matrices(
    matrices: [(&'static str, &HostMatrixView<'_>); 4],
    cooperation: &mut PolynomialCooperation<'_>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    for (name, matrix) in matrices {
        let count = matrix.nrows().checked_mul(matrix.ncols()).ok_or_else(|| {
            conversion_error(
                "state matrix size overflows",
                ConversionErrorKind::InvalidModel,
            )
            .into_runtime(builtin)
        })?;
        for start in (0..count).step_by(4096) {
            let end = start.saturating_add(4096).min(count);
            for index in start..end {
                let value = matrix.at(index % matrix.nrows(), index / matrix.nrows());
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(conversion_error(
                        format!("ss {name} contains non-finite values"),
                        ConversionErrorKind::InvalidModel,
                    )
                    .into_runtime(builtin));
                }
            }
            completed_polynomial_work(cooperation, end - start, builtin).await?;
        }
    }
    Ok(())
}

async fn characteristic_polynomial_cooperative<M: ComplexMatrixRead>(
    a: &M,
    cooperation: &mut PolynomialCooperation<'_>,
    builtin: &'static str,
    mut buffers: Option<&mut SsPolynomialBuffers>,
) -> BuiltinResult<Vec<Complex64>> {
    let order = a.nrows();
    let mut work = if let Some(holder) = buffers.as_deref_mut() {
        let current = holder
            .first
            .take()
            .expect("preallocated characteristic buffer");
        let next = holder
            .second
            .take()
            .expect("preallocated characteristic buffer");
        CharacteristicWorkspace::with_scratch(order, current, next)
    } else {
        CharacteristicWorkspace::new(order)
    }
    .map_err(|error| error.into_runtime(builtin))?;
    let mut pending = 0usize;
    for degree in 1..=order {
        for column in 0..order {
            for row in 0..order {
                for start in (0..order).step_by(4096) {
                    let end = start.saturating_add(4096).min(order);
                    work.advance_cell_fragment(a, row, column, start, end);
                    pending = count_polynomial_work(pending, end - start, builtin)?;
                    if pending >= 4096 {
                        completed_polynomial_work(cooperation, pending, builtin).await?;
                        pending = 0;
                    }
                }
            }
        }
        let mut trace = Complex64::new(0.0, 0.0);
        for start in (0..order).step_by(4096) {
            let end = start.saturating_add(4096).min(order);
            trace = work.trace_fragment(start, end, trace);
            pending = count_polynomial_work(pending, end - start, builtin)?;
            if pending >= 4096 {
                completed_polynomial_work(cooperation, pending, builtin).await?;
                pending = 0;
            }
        }
        let coefficient = work.append_coefficient(degree, trace);
        for start in (0..order).step_by(4096) {
            let end = start.saturating_add(4096).min(order);
            work.diagonal_fragment(start, end, coefficient);
            pending = count_polynomial_work(pending, end - start, builtin)?;
            if pending >= 4096 {
                completed_polynomial_work(cooperation, pending, builtin).await?;
                pending = 0;
            }
        }
        work.swap_degree();
    }
    completed_polynomial_work(cooperation, pending, builtin).await?;
    ensure_finite_coefficients("ss characteristic polynomial", &work.coefficients)
        .map_err(|error| error.into_runtime(builtin))?;
    if let Some(holder) = buffers {
        let (coefficients, first, second) = work.into_parts();
        holder.first = Some(first);
        holder.second = Some(second);
        Ok(coefficients)
    } else {
        Ok(work.coefficients)
    }
}

async fn state_space_numerator_cooperative<M: ComplexMatrixRead>(
    a: &M,
    b: &M,
    input: usize,
    c: &M,
    output: usize,
    direct: Complex64,
    denominator: &[Complex64],
    cooperation: &mut PolynomialCooperation<'_>,
    builtin: &'static str,
    mut buffers: Option<&mut SsPolynomialBuffers>,
) -> BuiltinResult<Vec<Complex64>> {
    let order = a.nrows();
    if order == 0 {
        return Ok(vec![direct]);
    }
    let mut work = if let Some(holder) = buffers.as_deref_mut() {
        let state = holder.first.take().expect("preallocated numerator buffer");
        let next = holder.second.take().expect("preallocated numerator buffer");
        NumeratorWorkspace::with_scratch(b, input, direct, denominator, state, next)
    } else {
        NumeratorWorkspace::new(b, input, direct, denominator)
    }
    .map_err(|error| error.into_runtime(builtin))?;
    let mut pending = 0usize;
    for degree in 1..=order {
        let mut markov = Complex64::new(0.0, 0.0);
        for start in (0..work.state_len()).step_by(4096) {
            let end = start.saturating_add(4096).min(work.state_len());
            markov = work.markov_fragment(c, output, start, end, markov);
            pending = count_polynomial_work(pending, end - start, builtin)?;
            if pending >= 4096 {
                completed_polynomial_work(cooperation, pending, builtin).await?;
                pending = 0;
            }
        }
        for start in (degree..work.coefficient_len()).step_by(4096) {
            let end = start.saturating_add(4096).min(work.coefficient_len());
            work.apply_markov_fragment(denominator, degree, start, end, markov);
            pending = count_polynomial_work(pending, end - start, builtin)?;
            if pending >= 4096 {
                completed_polynomial_work(cooperation, pending, builtin).await?;
                pending = 0;
            }
        }
        if degree < order {
            for row in 0..order {
                for start in (0..order).step_by(4096) {
                    let end = start.saturating_add(4096).min(order);
                    work.advance_row_fragment(a, row, start, end);
                    pending = count_polynomial_work(pending, end - start, builtin)?;
                    if pending >= 4096 {
                        completed_polynomial_work(cooperation, pending, builtin).await?;
                        pending = 0;
                    }
                }
            }
            work.finish_state();
        }
    }
    completed_polynomial_work(cooperation, pending, builtin).await?;
    ensure_finite_coefficients("ss numerator polynomial", &work.coefficients)
        .map_err(|error| error.into_runtime(builtin))?;
    if let Some(holder) = buffers {
        let (coefficients, first, second) = work.into_parts();
        holder.first = Some(first);
        holder.second = Some(second);
        Ok(coefficients)
    } else {
        Ok(work.coefficients)
    }
}

fn characteristic_polynomial_from_matrix(
    a: &DMatrix<Complex64>,
) -> ConversionResult<Vec<Complex64>> {
    let n = a.nrows();
    let mut work = CharacteristicWorkspace::new(n)?;
    for degree in 1..=n {
        for column in 0..n {
            for row in 0..n {
                work.advance_cell(a, row, column);
            }
        }
        work.finish_degree(degree);
    }
    ensure_finite_coefficients("ss characteristic polynomial", &work.coefficients)?;
    Ok(work.coefficients)
}

fn state_space_numerator(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: Complex64,
    denominator: &[Complex64],
) -> ConversionResult<Vec<Complex64>> {
    let degree = a.nrows();
    if degree == 0 {
        return Ok(vec![d]);
    }

    // Let H(s) = D + C(sI-A)^-1B. The denominator is
    // s^n+d1*s^(n-1)+...+dn. Its numerator coefficients are D*di plus
    // the convolution of d0..d(i-1) with the Markov parameters C*A^(k-1)*B.
    // This avoids interpolating at arbitrary points and, crucially, does not
    // erase a small but genuine B/C gain with an absolute cleanup threshold.
    let mut work = NumeratorWorkspace::new(b, 0, d, denominator)?;
    for k in 1..=degree {
        work.finish_markov(c, 0, denominator, k);
        if k < degree {
            for row in 0..degree {
                work.advance_row(a, row);
            }
            work.finish_state();
        }
    }
    ensure_finite_coefficients("ss numerator polynomial", &work.coefficients)?;
    Ok(work.coefficients)
}
fn trim_leading_complex_zeros(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| *value != Complex64::new(0.0, 0.0))
        .unwrap_or(coeffs.len());
    coeffs.drain(..first_nonzero);
    coeffs
}

fn ensure_finite_coefficients(label: &str, coeffs: &[Complex64]) -> ConversionResult<()> {
    if coeffs
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(conversion_error(
            format!("{label} values must be finite"),
            ConversionErrorKind::InvalidModel,
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "ss_conversion/tests.rs"]
mod tests;
