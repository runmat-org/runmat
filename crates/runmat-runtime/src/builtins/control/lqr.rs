//! Standard real `lqr` state-feedback gain synthesis.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_builtins::{
    BuiltinErrorDescriptor, LQR_ERROR_INTERNAL, LQR_ERROR_INVALID_ARGUMENT,
    LQR_ERROR_INVALID_MODEL, LQR_ERROR_UNSUPPORTED_MODEL, LQR_INTEGER_MATRIX_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, Tensor, Value};

use crate::builtins::common::{
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::control::tf_model::{
    output_complex_column, state_matrix_dimensions, validate_sample_time, SS_CLASS,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::{build_runtime_error, dispatcher, BuiltinResult};

const BUILTIN_NAME: &str = "lqr";

mod error;
mod gain;
mod managed_input;
mod managed_output;
mod managed_request;
mod managed_validation;
mod managed_weights;
mod standard_matrix;
mod validation;
mod weights;

pub(in crate::builtins::control) use error::RiccatiApi;
use error::RiccatiIssue;
pub(in crate::builtins::control) use managed_request::ManagedRiccatiRequest;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::lqr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("control-lqr"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "lqr gathers GPU-resident matrices and solves the dense Riccati equation on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::lqr")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "LQR synthesis is a control solver and does not participate in elementwise fusion.",
};

#[runtime_builtin(
    name = "lqr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::lqr"
)]
async fn lqr_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();
    if let Some(out_count) = out_count {
        if out_count > 3 {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "lqr supports at most three outputs",
            ));
        }
    }
    // Preserve call-shape errors ahead of compatibility admission, without
    // touching any resident operand or state-space property.
    let valid_arity = if matches!(&first, Value::Object(object) if object.is_class(SS_CLASS)) {
        (2..=3).contains(&rest.len())
    } else {
        (3..=4).contains(&rest.len())
    };
    if !valid_arity {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_ARGUMENT,
            "lqr expects (A,B,Q,R[,N]) or (sys,Q,R[,N])",
        ));
    }
    if integer_matrix_extension_required(&first, rest.view()) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LQR_INTEGER_MATRIX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if HostAllocationReceipt::policy(runtime.as_ref())
        .map_err(|error| RiccatiApi::Lqr.admission(runtime.as_ref(), error))?
        == HostAllocationPolicy::Managed
    {
        let runtime = runtime.as_ref().ok_or_else(|| {
            RiccatiApi::Lqr.admission(None, HostAllocationError::MissingManagedOwner)
        })?;
        return ManagedRiccatiRequest::from_lqr(first, rest, runtime)?
            .execute(out_count, runtime)
            .await;
    }
    let eval = LqrRequest::parse(first, rest, runtime.as_ref())
        .await?
        .solve(runtime.as_ref())
        .await?;
    if let Some(out_count) = out_count {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.into_outputs(out_count)?,
        ));
    }
    eval.into_outputs(1)?
        .into_iter()
        .next()
        .ok_or_else(|| lqr_error(&LQR_ERROR_INTERNAL, "LQR produced no gain output"))
}

fn integer_matrix_extension_required(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> bool {
    match first {
        Value::Object(object) if object.is_class(SS_CLASS) => ["A", "B"]
            .into_iter()
            .filter_map(|name| object.properties().get(name))
            .chain(rest.iter())
            .any(is_typed_real_integer_matrix),
        Value::Object(_) => false,
        _ => std::iter::once(first)
            .chain(rest.iter())
            .any(is_typed_real_integer_matrix),
    }
}

fn is_typed_real_integer_matrix(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
        }
        _ => false,
    }
}

#[derive(Clone)]
struct RealMatrix {
    matrix: DMatrix<f64>,
}

impl RealMatrix {
    fn preflight(
        label: &'static str,
        value: &Value,
        api: RiccatiApi,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        HostAllocationReceipt::policy(runtime).map_err(|error| api.admission(runtime, error))?;
        match value {
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved =>
            {
                Err(api.error(
                    RiccatiIssue::UnsupportedInput,
                    format!("{label} must contain real numeric values"),
                ))
            }
            Value::GpuTensor(_) => Ok(()),
            Value::Tensor(tensor) => Self::validate_tensor(label, tensor, api, runtime),
            Value::Num(value) if value.is_finite() => Ok(()),
            Value::Int(_) => Ok(()),
            Value::Num(_) => Err(api.error(
                RiccatiIssue::UnsupportedInput,
                format!("{label} must contain finite real values"),
            )),
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(api.error(
                RiccatiIssue::UnsupportedInput,
                format!("{label} must contain real numeric values"),
            )),
            other => Err(api.error(
                RiccatiIssue::UnsupportedInput,
                format!("{label} must be a real numeric matrix, got {other:?}"),
            )),
        }
    }

    async fn parse(
        label: &'static str,
        value: Value,
        api: RiccatiApi,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
            || matches!(&value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
        {
            return Err(api.error(
                RiccatiIssue::UnsupportedInput,
                format!("{label} must contain real numeric values"),
            ));
        }
        // A provider/cancellation failure is operational, not a malformed
        // Riccati matrix. Preserve its typed source across host gathering.
        let gathered = dispatcher::gather_if_needed_async(&value).await?;
        Self::parse_host(label, gathered, api, runtime)
    }

    fn parse_host(
        label: &'static str,
        gathered: Value,
        api: RiccatiApi,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let tensor = match gathered {
            Value::Tensor(tensor) => tensor::integer_tensor_to_f64(tensor).map_err(|err| {
                api.error(
                    RiccatiIssue::Internal,
                    format!("failed to normalize {label}: {err}"),
                )
            })?,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|err| {
                api.error(
                    RiccatiIssue::Internal,
                    format!("failed to build scalar tensor: {err}"),
                )
            })?,
            Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
                api.error(
                    RiccatiIssue::Internal,
                    format!("failed to build scalar tensor: {err}"),
                )
            })?,
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                return Err(api.error(
                    RiccatiIssue::UnsupportedInput,
                    format!("{label} must contain real numeric values"),
                ));
            }
            other => {
                return Err(api.error(
                    RiccatiIssue::UnsupportedInput,
                    format!("{label} must be a real numeric matrix, got {other:?}"),
                ));
            }
        };
        Self::from_tensor(label, tensor, api, runtime)
    }

    fn from_tensor(
        label: &'static str,
        tensor: Tensor,
        api: RiccatiApi,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        Self::validate_tensor(label, &tensor, api, runtime)?;
        HostAllocationReceipt::policy(runtime).map_err(|error| api.admission(runtime, error))?;
        let values = tensor::tensor_values_f64_cow(&tensor);
        let (rows, cols) = state_matrix_dimensions(&tensor);
        Ok(Self {
            matrix: DMatrix::from_column_slice(rows, cols, &values),
        })
    }

    fn validate_tensor(
        label: &'static str,
        tensor: &Tensor,
        api: RiccatiApi,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if tensor.shape().len() > 2 {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                format!(
                    "{label} must be a 2-D matrix, got shape {:?}",
                    tensor.shape()
                ),
            ));
        }
        let values = tensor::tensor_values_f64_cow(tensor);
        for chunk in values.chunks(4096) {
            HostAllocationReceipt::policy(runtime)
                .map_err(|error| api.admission(runtime, error))?;
            if chunk.iter().any(|value| !value.is_finite()) {
                return Err(api.error(
                    RiccatiIssue::UnsupportedInput,
                    format!("{label} must contain finite real values"),
                ));
            }
        }
        Ok(())
    }

    fn rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn cols(&self) -> usize {
        self.matrix.ncols()
    }
}

pub(in crate::builtins::control) struct LqrRequest {
    a: RealMatrix,
    b: RealMatrix,
    q: RealMatrix,
    r: RealMatrix,
    n: Option<RealMatrix>,
    domain: LqrDomain,
    api: RiccatiApi,
}

#[derive(Clone, Copy)]
pub(in crate::builtins::control) enum LqrDomain {
    Continuous,
    Discrete,
}

impl LqrRequest {
    async fn parse(
        first: Value,
        rest: crate::call::BuiltinArguments,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        match first {
            Value::Object(object) if object.is_class(SS_CLASS) => {
                Self::from_state_space_object(object, rest, runtime).await
            }
            Value::Object(_) => Err(lqr_error(
                &LQR_ERROR_UNSUPPORTED_MODEL,
                "lqr(sys,Q,R) requires a state-space ss model",
            )),
            a => Self::from_matrices(a, rest, runtime).await,
        }
    }

    async fn from_matrices(
        a: Value,
        mut rest: crate::call::BuiltinArguments,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if !(3..=4).contains(&rest.len()) {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "expected lqr(A,B,Q,R) or lqr(A,B,Q,R,N)",
            ));
        }
        let b = rest.pop_front().expect("validated B");
        let q = rest.pop_front().expect("validated Q");
        let r = rest.pop_front().expect("validated R");
        Self::prepare_standard_matrix(
            a,
            b,
            q,
            Some(r),
            rest.pop_front(),
            LqrDomain::Continuous,
            RiccatiApi::Lqr,
            runtime,
        )
        .await
    }

    async fn from_state_space_object(
        object: ObjectInstance,
        mut rest: crate::call::BuiltinArguments,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if !(2..=3).contains(&rest.len()) {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "expected lqr(sys,Q,R) or lqr(sys,Q,R,N)",
            ));
        }
        let q = rest.get(0).expect("validated Q");
        let r = rest.get(1).expect("validated R");
        let sample_time = scalar_property(&object, "Ts")?;
        validate_sample_time(sample_time, BUILTIN_NAME)?;
        let a_value = object
            .properties()
            .get("A")
            .ok_or_else(|| lqr_error(&LQR_ERROR_INVALID_MODEL, "ss object missing A property"))?;
        let b_value = object
            .properties()
            .get("B")
            .ok_or_else(|| lqr_error(&LQR_ERROR_INVALID_MODEL, "ss object missing B property"))?;
        for (label, value) in [("A", a_value), ("B", b_value), ("Q", q), ("R", r)] {
            RealMatrix::preflight(label, value, RiccatiApi::Lqr, runtime)?;
        }
        if let Some(n) = rest.get(2) {
            RealMatrix::preflight("N", n, RiccatiApi::Lqr, runtime)?;
        }
        let a = object.property("A").cloned().expect("preflighted A");
        let b = object.property("B").cloned().expect("preflighted B");
        let q = rest.pop_front().expect("validated Q");
        let r = rest.pop_front().expect("validated R");
        Self::prepare_standard_matrix(
            a,
            b,
            q,
            Some(r),
            rest.pop_front(),
            if sample_time > 0.0 {
                LqrDomain::Discrete
            } else {
                LqrDomain::Continuous
            },
            RiccatiApi::Lqr,
            runtime,
        )
        .await
    }

    pub(in crate::builtins::control) async fn solve(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<LqrEval> {
        validate_dimensions(&self)?;
        let api = self.api;
        let a = self.a.matrix;
        let b = self.b.matrix;
        let q = symmetric_part(self.q.matrix);
        let r = symmetric_part(self.r.matrix);
        let n = self
            .n
            .map(|matrix| matrix.matrix)
            .unwrap_or_else(|| DMatrix::<f64>::zeros(a.nrows(), b.ncols()));
        weights::validate(&q, &r, &n, api)?;
        let domain = match self.domain {
            LqrDomain::Continuous => {
                crate::builtins::math::linalg::factor::qz::riccati::RiccatiTimeDomain::Continuous
            }
            LqrDomain::Discrete => {
                crate::builtins::math::linalg::factor::qz::riccati::RiccatiTimeDomain::Discrete
            }
        };
        let factor = crate::builtins::math::linalg::factor::qz::riccati::stabilizing_subspace(
            a,
            b,
            q,
            r,
            n,
            domain,
            runtime,
            api.name(),
            |message| api.error(RiccatiIssue::SolverFailed, message),
        )
        .await?;
        validation::validate_solution(factor, self.domain, api)
    }
}

pub(in crate::builtins::control) struct LqrEval {
    k: DMatrix<f64>,
    s: DMatrix<f64>,
    poles: Option<Vec<Complex64>>,
    api: RiccatiApi,
}

impl LqrEval {
    pub(in crate::builtins::control) fn into_outputs(
        self,
        count: usize,
    ) -> BuiltinResult<Vec<Value>> {
        let mut outputs = Vec::with_capacity(count.min(3));
        if count >= 1 {
            outputs.push(matrix_value_owned("K", self.k, self.api)?);
        }
        if count >= 2 {
            outputs.push(matrix_value_owned("S", self.s, self.api)?);
        }
        if count >= 3 {
            outputs.push(output_complex_column(
                self.poles.ok_or_else(|| {
                    self.api
                        .error(RiccatiIssue::Internal, "closed-loop poles missing")
                })?,
                self.api.name(),
            )?);
        }
        Ok(outputs)
    }

    pub(in crate::builtins::control) fn into_care_outputs(
        self,
        count: usize,
    ) -> BuiltinResult<Vec<Value>> {
        let mut outputs = Vec::with_capacity(count.min(3));
        if count >= 1 {
            outputs.push(matrix_value_owned("X", self.s, self.api)?);
        }
        if count >= 2 {
            outputs.push(output_complex_column(
                self.poles.ok_or_else(|| {
                    self.api
                        .error(RiccatiIssue::Internal, "closed-loop poles missing")
                })?,
                self.api.name(),
            )?);
        }
        if count >= 3 {
            outputs.push(matrix_value_owned("G", self.k, self.api)?);
        }
        Ok(outputs)
    }
}
fn validate_dimensions(request: &LqrRequest) -> BuiltinResult<()> {
    let api = request.api;
    let n = request.a.rows();
    let m = request.b.cols();
    if n == 0 {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            "A must have at least one state",
        ));
    }
    if request.a.cols() != n {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            format!(
                "A must be square, got {}x{}",
                request.a.rows(),
                request.a.cols()
            ),
        ));
    }
    if request.b.rows() != n || m == 0 {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            format!(
                "B must have {n} rows and at least one input, got {}x{}",
                request.b.rows(),
                request.b.cols()
            ),
        ));
    }
    if request.q.rows() != n || request.q.cols() != n {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            format!(
                "Q must be {n}x{n}, got {}x{}",
                request.q.rows(),
                request.q.cols()
            ),
        ));
    }
    if request.r.rows() != m || request.r.cols() != m {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            format!(
                "R must be {m}x{m}, got {}x{}",
                request.r.rows(),
                request.r.cols()
            ),
        ));
    }
    if let Some(cross) = &request.n {
        if cross.rows() != n || cross.cols() != m {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                format!("N must be {n}x{m}, got {}x{}", cross.rows(), cross.cols()),
            ));
        }
    }
    if !is_symmetric(&request.q.matrix, 1.0e-8) {
        return Err(api.error(RiccatiIssue::InvalidDimensions, "Q must be symmetric"));
    }
    if !is_symmetric(&request.r.matrix, 1.0e-8) {
        return Err(api.error(RiccatiIssue::InvalidDimensions, "R must be symmetric"));
    }
    Ok(())
}

fn closed_loop_poles(closed_loop: &DMatrix<f64>, api: RiccatiApi) -> BuiltinResult<Vec<Complex64>> {
    let complex = closed_loop.map(|value| Complex64::new(value, 0.0));
    let eigenvalues = complex.eigenvalues().ok_or_else(|| {
        api.error(
            RiccatiIssue::SolverFailed,
            "failed to compute closed-loop poles",
        )
    })?;
    Ok(eigenvalues.iter().copied().collect())
}

fn scalar_property(object: &ObjectInstance, name: &'static str) -> BuiltinResult<f64> {
    match object.properties().get(name) {
        Some(Value::Num(value)) => Ok(*value),
        Some(Value::Int(value)) => Ok(value.to_f64()),
        Some(Value::Tensor(tensor)) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        Some(other) => Err(lqr_error(
            &LQR_ERROR_INVALID_MODEL,
            format!("ss object property {name} must be a scalar, got {other:?}"),
        )),
        None => Err(lqr_error(
            &LQR_ERROR_INVALID_MODEL,
            format!("ss object missing {name} property"),
        )),
    }
}

fn matrix_value_owned(
    label: &'static str,
    matrix: DMatrix<f64>,
    api: RiccatiApi,
) -> BuiltinResult<Value> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new(Vec::from(matrix.data), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| {
            api.error(
                RiccatiIssue::Internal,
                format!("failed to build {label} output tensor: {err}"),
            )
        })
}

fn symmetric_part(matrix: DMatrix<f64>) -> DMatrix<f64> {
    (&matrix + matrix.transpose()) * 0.5
}

fn is_symmetric(matrix: &DMatrix<f64>, tol: f64) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    for row in 0..matrix.nrows() {
        for col in (row + 1)..matrix.ncols() {
            let scale = matrix[(row, col)]
                .abs()
                .max(matrix[(col, row)].abs())
                .max(1.0);
            if (matrix[(row, col)] - matrix[(col, row)]).abs() > tol * scale {
                return false;
            }
        }
    }
    true
}

fn max_abs_matrix(matrix: &DMatrix<f64>) -> f64 {
    if matrix.iter().any(|value| !value.is_finite()) {
        return f64::INFINITY;
    }
    matrix
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn lqr_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    match error.identifier {
        Some(identifier) => builder.with_identifier(identifier).build(),
        None => builder.build(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        LQR_DESCRIPTOR, LQR_ERROR_INVALID_DIMENSIONS, LQR_ERROR_UNSUPPORTED_INPUT,
    };
    use runmat_value::{IntValue, IntegerStorage};
    use std::rc::Rc;

    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };

    struct LqrManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for LqrManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => {
                crate::sequence::into_unbounded_output_values(values).expect("ordinary LQR outputs")
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn matrix_from_value(value: &Value) -> DMatrix<f64> {
        match value {
            Value::Tensor(tensor) => {
                let (rows, cols) = state_matrix_dimensions(tensor);
                DMatrix::from_column_slice(rows, cols, &tensor.materialize_f64())
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lqr_double_integrator_matches_known_solution() {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = block_on(lqr_builtin(
            None,
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
            ]),
        ))
        .expect("lqr");
        let outputs = output_list(result);
        let k = matrix_from_value(&outputs[0]);
        let s = matrix_from_value(&outputs[1]);

        assert!((k[(0, 0)] - 1.0).abs() < 1.0e-8);
        assert!((k[(0, 1)] - 3.0_f64.sqrt()).abs() < 1.0e-8);
        assert!((s[(0, 0)] - 3.0_f64.sqrt()).abs() < 1.0e-8);
        assert!((s[(0, 1)] - 1.0).abs() < 1.0e-8);
        assert!((s[(1, 1)] - 3.0_f64.sqrt()).abs() < 1.0e-8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn pendulum_lqr_runs_through_ordinary_and_managed_riccati_workers() {
        let p = 0.0132;
        let a = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.0024 / p,
            0.0,
            -0.006 / p,
            0.0,
            0.035316 / p,
            0.0,
            0.41202 / p,
            0.0,
            0.0,
            1.0,
            0.0,
        ];
        let b = vec![0.0, 0.024 / p, 0.0, 0.06 / p];
        let q = vec![
            5000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let run = |runtime| {
            block_on(lqr_builtin(
                runtime,
                tensor(a.clone(), 4, 4),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    tensor(b.clone(), 4, 1),
                    tensor(q.clone(), 4, 4),
                    Value::Num(1.0),
                ]),
            ))
            .expect("pendulum LQR gain")
        };
        let ordinary = matrix_from_value(&run(None));
        let ledger = ManagedHostEffectLedger::new(16 * 1024 * 1024);
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(LqrManagedParallel(ledger.clone()))),
                );
        let managed = matrix_from_value(&run(Some(runtime)));
        for (index, expected) in [-70.711, -37.82, 105.56, 20.919].into_iter().enumerate() {
            assert!((ordinary[(0, index)] - expected).abs() < 0.02);
            assert!((managed[(0, index)] - ordinary[(0, index)]).abs() < 1.0e-7);
        }
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn lqr_preserves_rank_one_dense_matrix_projection() {
        let error = block_on(lqr_builtin(
            None,
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("rank-one A is a one-by-two matrix");
        assert!(error.message().contains("A must be square, got 1x2"));
    }

    #[test]
    fn lqr_accepts_typed_integer_matrices_at_double_boundary() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = block_on(lqr_builtin(
            None,
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1, 0]), 2, 2),
            crate::builtins::common::test_support::builtin_arguments(vec![
                integer_tensor(IntegerStorage::I16(vec![0, 1]), 2, 1),
                integer_tensor(IntegerStorage::U16(vec![1, 0, 0, 1]), 2, 2),
                integer_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            ]),
        ))
        .expect("lqr");
        let outputs = output_list(result);
        let k = matrix_from_value(&outputs[0]);
        let s = matrix_from_value(&outputs[1]);
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
        assert!(k.iter().all(|value| value.is_finite()));
        assert!(s.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn strict_mode_rejects_integer_matrix_before_provider_access() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(false);
        let host_error = block_on(lqr_builtin(
            None,
            integer_tensor(IntegerStorage::I16(vec![-1]), 1, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("strict host integer extension gate");
        assert_eq!(
            host_error.identifier(),
            LQR_INTEGER_MATRIX_EXTENSION.error_identifier
        );
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::U64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("allocate integer handle");
        let error = block_on(lqr_builtin(
            None,
            Value::GpuTensor(allocation.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("strict integer extension gate");
        assert_eq!(
            error.identifier(),
            LQR_INTEGER_MATRIX_EXTENSION.error_identifier
        );
        assert_eq!(allocation.probe.retirements(), 0);
        allocation.handle.release().expect("release handle");
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn invalid_arity_precedes_integer_extension_admission() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(lqr_builtin(
            None,
            Value::Int(IntValue::I16(1)),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("invalid call shape");
        assert_eq!(error.identifier(), LQR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn zero_output_request_still_validates_the_riccati_problem() {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let error = block_on(lqr_builtin(
            None,
            Value::Num(-1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(-1.0),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("negative Q must not succeed when no output is requested");
        assert_eq!(error.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn all_integer_scalar_classes_enter_double_solver_in_runmat_mode() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for gain in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            let result = block_on(lqr_builtin(
                None,
                Value::Num(-1.0),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::Num(1.0),
                    Value::Int(gain),
                ]),
            ))
            .expect("integer R accepted");
            let gain_matrix = matrix_from_value(&result);
            assert_eq!(gain_matrix.shape(), (1, 1));
            assert!(gain_matrix[(0, 0)].is_finite());
        }
    }

    #[test]
    fn resident_float_matrices_gather_for_host_solver() {
        let a = Tensor::new(vec![0.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a)
                .expect("upload A");
            provider.reset_telemetry();
            let result = block_on(lqr_builtin(
                None,
                Value::GpuTensor(resident),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    tensor(vec![0.0, 1.0], 2, 1),
                    tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                    Value::Num(1.0),
                ]),
            ))
            .expect("resident LQR");
            assert_eq!(matrix_from_value(&result).shape(), (1, 2));
            assert!(provider.telemetry_snapshot().download_bytes > 0);
        });
    }

    #[test]
    fn state_space_required_properties_preflight_before_resident_gather() {
        let a = Tensor::new(vec![0.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a)
                .expect("upload A");
            let mut model = ObjectInstance::new(SS_CLASS.to_string());
            model.replace_property_unbounded("Ts".into(), Value::Num(0.0));
            model.replace_property_unbounded("A".into(), Value::GpuTensor(resident.clone()));
            model.replace_property_unbounded("B".into(), tensor(vec![0.0, 1.0], 2, 1));
            provider.reset_telemetry();
            let result = block_on(lqr_builtin(
                None,
                Value::Object(model.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                    Value::Num(1.0),
                ]),
            ))
            .expect("resident ss LQR");
            assert_eq!(matrix_from_value(&result).shape(), (1, 2));
            assert!(provider.telemetry_snapshot().download_bytes > 0);

            crate::builtins::common::test_support::remove_object_property_unbounded(
                &mut model, "B",
            );
            provider.reset_telemetry();
            let error = block_on(lqr_builtin(
                None,
                Value::Object(model),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                    Value::Num(1.0),
                ]),
            ))
            .expect_err("missing B");
            assert_eq!(error.identifier(), LQR_ERROR_INVALID_MODEL.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn invalid_host_weight_precedes_resident_state_matrix_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![0.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &a).unwrap();
            provider.reset_telemetry();
            let error = block_on(lqr_builtin(
                None,
                Value::GpuTensor(resident),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    tensor(vec![0.0, 1.0], 2, 1),
                    Value::String("bad".into()),
                    Value::Num(1.0),
                ]),
            ))
            .expect_err("invalid Q");
            assert_eq!(error.identifier(), LQR_ERROR_UNSUPPORTED_INPUT.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn lqr_cross_weight_changes_gain_and_satisfies_care() {
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = block_on(lqr_builtin(
            None,
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![2.0, 0.0, 0.0, 3.0], 2, 2),
                Value::Num(2.0),
                tensor(vec![0.25, 0.0], 2, 1),
            ]),
        ))
        .expect("lqr with N");
        let outputs = output_list(result);
        let k = matrix_from_value(&outputs[0]);
        let s = matrix_from_value(&outputs[1]);
        assert!(k[(0, 0)].is_finite());
        assert!(k[(0, 1)].is_finite());

        let a = DMatrix::from_column_slice(2, 2, &[0.0, 0.0, 1.0, 0.0]);
        let b = DMatrix::from_column_slice(2, 1, &[0.0, 1.0]);
        let q = DMatrix::from_column_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let r_inv = DMatrix::from_element(1, 1, 0.5);
        let n = DMatrix::from_column_slice(2, 1, &[0.25, 0.0]);
        let residual = a.transpose() * &s + &s * &a
            - (&s * &b + &n) * r_inv * (b.transpose() * &s + n.transpose())
            + q;
        assert!(max_abs_matrix(&residual) < 1.0e-7);
    }

    #[test]
    fn lqr_accepts_continuous_state_space_object() {
        let sys = block_on(crate::builtins::control::ss::ss_builtin(
            None,
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            tensor(vec![0.0, 1.0], 2, 1),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("ss");
        let k = block_on(lqr_builtin(
            None,
            sys,
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
            ]),
        ))
        .expect("lqr(sys)");
        let k = matrix_from_value(&k);
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn lqr_accepts_discrete_state_space_object() {
        let sys = block_on(crate::builtins::control::ss::ss_builtin(
            None,
            tensor(vec![1.0, 0.0, 0.1, 1.0], 2, 2),
            tensor(vec![0.005, 0.1], 2, 1),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.1)]),
        ))
        .expect("discrete ss");
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = block_on(lqr_builtin(
            None,
            sys,
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
            ]),
        ))
        .expect("lqr(discrete sys)");
        let outputs = output_list(result);
        let k = matrix_from_value(&outputs[0]);
        let s = matrix_from_value(&outputs[1]);

        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
        assert!(s[(0, 0)].is_finite());
        assert!(s[(1, 1)].is_finite());
        match &outputs[2] {
            Value::ComplexTensor(poles) => {
                assert_eq!(poles.shape(), &[2, 1]);
                assert!(poles
                    .materialize_f64()
                    .iter()
                    .all(|(re, im)| (re * re + im * im).sqrt() < 1.0));
            }
            other => panic!("expected complex pole tensor, got {other:?}"),
        }
    }

    #[test]
    fn lqr_accepts_typed_integer_sample_time_property_on_state_space_object() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let sample_time = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();
        let sys = block_on(crate::builtins::control::ss::ss_builtin(
            None,
            tensor(vec![1.0, 0.0, 1.0, 1.0], 2, 2),
            tensor(vec![0.0, 1.0], 2, 1),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                sample_time,
            )]),
        ))
        .expect("discrete ss");
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = block_on(lqr_builtin(
            None,
            sys,
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
            ]),
        ))
        .expect("lqr(discrete sys)");
        let outputs = output_list(result);
        let k = matrix_from_value(&outputs[0]);
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn lqr_rejects_bad_dimensions() {
        let err = block_on(lqr_builtin(
            None,
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0], 1, 1),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("dimension mismatch");
        assert_eq!(err.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn lqr_rejects_non_positive_definite_r() {
        let err = block_on(lqr_builtin(
            None,
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(-1.0),
            ]),
        ))
        .expect_err("invalid R");
        assert_eq!(err.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
        assert!(err.to_string().contains("positive definite"));
    }

    #[test]
    fn lqr_rejects_indefinite_block_cost() {
        let err = block_on(lqr_builtin(
            None,
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            crate::builtins::common::test_support::builtin_arguments(vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
                tensor(vec![2.0, 0.0], 2, 1),
            ]),
        ))
        .expect_err("indefinite block cost");
        assert_eq!(err.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
        assert!(err.to_string().contains("positive semidefinite"));
    }

    #[test]
    fn lqr_zero_requested_outputs_still_reject_invalid_call_shape() {
        let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let error = block_on(lqr_builtin(
            None,
            Value::String("not matrices".to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("zero-output LQR still validates the call");
        assert_eq!(error.identifier(), LQR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn lqr_catalog_descriptor_covers_matrix_and_model_forms() {
        let labels: Vec<&str> = LQR_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"K = lqr(A, B, Q, R)"));
        assert!(labels.contains(&"[K, S, e] = lqr(sys, Q, R, N)"));
    }
}
