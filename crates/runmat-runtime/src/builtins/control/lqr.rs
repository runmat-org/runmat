//! MATLAB-compatible `lqr` state-feedback gain synthesis.

use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ObjectInstance, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::control::tf_model::{
    control_error, output_complex_column, validate_sample_time, SS_CLASS,
};
use crate::builtins::control::type_resolvers::lqr_type;
use crate::{dispatcher, BuiltinResult};

const BUILTIN_NAME: &str = "lqr";
const EPS: f64 = 1.0e-10;
const SIGN_MAX_ITERS: usize = 100;
const DARE_MAX_ITERS: usize = 1000;

const LQR_PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "K",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Optimal state-feedback gain matrix.",
};
const LQR_PARAM_S: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stabilizing algebraic Riccati solution.",
};
const LQR_PARAM_E: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "e",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closed-loop poles of A-B*K.",
};
const LQR_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State matrix.",
};
const LQR_PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
};
const LQR_PARAM_Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State weighting matrix.",
};
const LQR_PARAM_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input weighting matrix.",
};
const LQR_PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("zeros(size(B))"),
    description: "Optional state-input cross-weight matrix.",
};
const LQR_PARAM_SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State-space model object.",
};

const LQR_OUTPUT_K: [BuiltinParamDescriptor; 1] = [LQR_PARAM_K];
const LQR_OUTPUT_KS: [BuiltinParamDescriptor; 2] = [LQR_PARAM_K, LQR_PARAM_S];
const LQR_OUTPUT_KSE: [BuiltinParamDescriptor; 3] = [LQR_PARAM_K, LQR_PARAM_S, LQR_PARAM_E];
const LQR_INPUTS_ABQR: [BuiltinParamDescriptor; 4] =
    [LQR_PARAM_A, LQR_PARAM_B, LQR_PARAM_Q, LQR_PARAM_R];
const LQR_INPUTS_ABQRN: [BuiltinParamDescriptor; 5] = [
    LQR_PARAM_A,
    LQR_PARAM_B,
    LQR_PARAM_Q,
    LQR_PARAM_R,
    LQR_PARAM_N,
];
const LQR_INPUTS_SYS_QR: [BuiltinParamDescriptor; 3] = [LQR_PARAM_SYS, LQR_PARAM_Q, LQR_PARAM_R];
const LQR_INPUTS_SYS_QRN: [BuiltinParamDescriptor; 4] =
    [LQR_PARAM_SYS, LQR_PARAM_Q, LQR_PARAM_R, LQR_PARAM_N];

const LQR_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "K = lqr(A, B, Q, R)",
        inputs: &LQR_INPUTS_ABQR,
        outputs: &LQR_OUTPUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "K = lqr(A, B, Q, R, N)",
        inputs: &LQR_INPUTS_ABQRN,
        outputs: &LQR_OUTPUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(A, B, Q, R)",
        inputs: &LQR_INPUTS_ABQR,
        outputs: &LQR_OUTPUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(A, B, Q, R, N)",
        inputs: &LQR_INPUTS_ABQRN,
        outputs: &LQR_OUTPUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(A, B, Q, R)",
        inputs: &LQR_INPUTS_ABQR,
        outputs: &LQR_OUTPUT_KSE,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(A, B, Q, R, N)",
        inputs: &LQR_INPUTS_ABQRN,
        outputs: &LQR_OUTPUT_KSE,
    },
    BuiltinSignatureDescriptor {
        label: "K = lqr(sys, Q, R)",
        inputs: &LQR_INPUTS_SYS_QR,
        outputs: &LQR_OUTPUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "K = lqr(sys, Q, R, N)",
        inputs: &LQR_INPUTS_SYS_QRN,
        outputs: &LQR_OUTPUT_K,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(sys, Q, R)",
        inputs: &LQR_INPUTS_SYS_QR,
        outputs: &LQR_OUTPUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S] = lqr(sys, Q, R, N)",
        inputs: &LQR_INPUTS_SYS_QRN,
        outputs: &LQR_OUTPUT_KS,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(sys, Q, R)",
        inputs: &LQR_INPUTS_SYS_QR,
        outputs: &LQR_OUTPUT_KSE,
    },
    BuiltinSignatureDescriptor {
        label: "[K, S, e] = lqr(sys, Q, R, N)",
        inputs: &LQR_INPUTS_SYS_QRN,
        outputs: &LQR_OUTPUT_KSE,
    },
];

const LQR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INVALID_ARGUMENT",
    identifier: Some("RunMat:lqr:InvalidArgument"),
    when: "Call shape, requested output count, or optional arguments are invalid.",
    message: "lqr: invalid argument",
};
const LQR_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INVALID_MODEL",
    identifier: Some("RunMat:lqr:InvalidModel"),
    when: "State-space model metadata is malformed or unsupported.",
    message: "lqr: invalid model",
};
const LQR_ERROR_INVALID_DIMENSIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INVALID_DIMENSIONS",
    identifier: Some("RunMat:lqr:InvalidDimensions"),
    when: "System and weighting matrix dimensions are incompatible.",
    message: "lqr: invalid dimensions",
};
const LQR_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:lqr:UnsupportedInput"),
    when: "Input contains unsupported types, complex values, or non-finite data.",
    message: "lqr: unsupported input",
};
const LQR_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:lqr:UnsupportedModel"),
    when: "The requested model form is outside state-space LQR.",
    message: "lqr: unsupported model",
};
const LQR_ERROR_SOLVER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.SOLVER_FAILED",
    identifier: Some("RunMat:lqr:SolverFailed"),
    when: "The algebraic Riccati solver cannot produce a stabilizing finite solution.",
    message: "lqr: Riccati solver failed",
};
const LQR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LQR.INTERNAL",
    identifier: Some("RunMat:lqr:Internal"),
    when: "Runtime fails while constructing outputs.",
    message: "lqr: internal error",
};
const LQR_ERRORS: [BuiltinErrorDescriptor; 7] = [
    LQR_ERROR_INVALID_ARGUMENT,
    LQR_ERROR_INVALID_MODEL,
    LQR_ERROR_INVALID_DIMENSIONS,
    LQR_ERROR_UNSUPPORTED_INPUT,
    LQR_ERROR_UNSUPPORTED_MODEL,
    LQR_ERROR_SOLVER_FAILED,
    LQR_ERROR_INTERNAL,
];

pub const LQR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LQR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LQR_ERRORS,
};

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
    category = "control",
    summary = "Compute linear quadratic regulator gains.",
    keywords = "lqr,care,riccati,state feedback,control system",
    accel = "sink",
    sink = true,
    type_resolver(lqr_type),
    descriptor(crate::builtins::control::lqr::LQR_DESCRIPTOR),
    builtin_path = "crate::builtins::control::lqr"
)]
async fn lqr_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();
    if let Some(out_count) = out_count {
        if out_count > 3 {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "lqr supports at most three outputs",
            ));
        }
        if out_count == 0 {
            return Ok(crate::output_count::output_list_with_padding(0, Vec::new()));
        }
    }
    let requested_outputs = out_count.unwrap_or(1);
    let eval = LqrRequest::parse(first, rest)
        .await?
        .solve(requested_outputs >= 3)?;
    if let Some(out_count) = out_count {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(out_count)?,
        ));
    }
    eval.k_value()
}

#[derive(Clone)]
struct RealMatrix {
    matrix: DMatrix<f64>,
}

impl RealMatrix {
    async fn parse(label: &'static str, value: Value) -> BuiltinResult<Self> {
        let gathered = dispatcher::gather_if_needed_async(&value)
            .await
            .map_err(|err| {
                control_error(
                    BUILTIN_NAME,
                    "RunMat:lqr:UnsupportedInput",
                    format!(
                        "{BUILTIN_NAME}: failed to gather {label}: {}",
                        err.message()
                    ),
                )
            })?;
        let tensor = match gathered {
            Value::Tensor(tensor) => tensor::integer_tensor_to_f64(tensor).map_err(|err| {
                lqr_error(
                    &LQR_ERROR_INTERNAL,
                    format!("failed to normalize {label}: {err}"),
                )
            })?,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|err| {
                lqr_error(
                    &LQR_ERROR_INTERNAL,
                    format!("failed to build scalar tensor: {err}"),
                )
            })?,
            Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
                lqr_error(
                    &LQR_ERROR_INTERNAL,
                    format!("failed to build scalar tensor: {err}"),
                )
            })?,
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                return Err(lqr_error(
                    &LQR_ERROR_UNSUPPORTED_INPUT,
                    format!("{label} must contain real numeric values"),
                ));
            }
            other => {
                return Err(lqr_error(
                    &LQR_ERROR_UNSUPPORTED_INPUT,
                    format!("{label} must be a real numeric matrix, got {other:?}"),
                ));
            }
        };
        Self::from_tensor(label, tensor)
    }

    fn from_tensor(label: &'static str, tensor: Tensor) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_DIMENSIONS,
                format!("{label} must be a 2-D matrix, got shape {:?}", tensor.shape),
            ));
        }
        let values = tensor::tensor_values_f64_cow(&tensor);
        if values.iter().any(|value| !value.is_finite()) {
            return Err(lqr_error(
                &LQR_ERROR_UNSUPPORTED_INPUT,
                format!("{label} must contain finite real values"),
            ));
        }
        Ok(Self {
            matrix: DMatrix::from_column_slice(tensor.rows, tensor.cols, &values),
        })
    }

    fn rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn cols(&self) -> usize {
        self.matrix.ncols()
    }
}

struct LqrRequest {
    a: RealMatrix,
    b: RealMatrix,
    q: RealMatrix,
    r: RealMatrix,
    n: Option<RealMatrix>,
    domain: LqrDomain,
}

#[derive(Clone, Copy)]
enum LqrDomain {
    Continuous,
    Discrete,
}

impl LqrRequest {
    async fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        match first {
            Value::Object(object) if object.is_class(SS_CLASS) => {
                Self::from_state_space_object(object, rest).await
            }
            Value::Object(_) => Err(lqr_error(
                &LQR_ERROR_UNSUPPORTED_MODEL,
                "lqr(sys,Q,R) requires a state-space ss model",
            )),
            a => Self::from_matrices(a, rest).await,
        }
    }

    async fn from_matrices(a: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let [b, q, r, tail @ ..] = rest.as_slice() else {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "expected lqr(A,B,Q,R) or lqr(A,B,Q,R,N)",
            ));
        };
        if tail.len() > 1 {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "lqr(A,B,Q,R,N) accepts at most five inputs",
            ));
        }
        Ok(Self {
            a: RealMatrix::parse("A", a).await?,
            b: RealMatrix::parse("B", b.clone()).await?,
            q: RealMatrix::parse("Q", q.clone()).await?,
            r: RealMatrix::parse("R", r.clone()).await?,
            n: match tail {
                [] => None,
                [n] => Some(RealMatrix::parse("N", n.clone()).await?),
                _ => unreachable!(),
            },
            domain: LqrDomain::Continuous,
        })
    }

    async fn from_state_space_object(
        object: ObjectInstance,
        rest: Vec<Value>,
    ) -> BuiltinResult<Self> {
        let [q, r, tail @ ..] = rest.as_slice() else {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "expected lqr(sys,Q,R) or lqr(sys,Q,R,N)",
            ));
        };
        if tail.len() > 1 {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_ARGUMENT,
                "lqr(sys,Q,R,N) accepts at most four inputs",
            ));
        }
        let sample_time = scalar_property(&object, "Ts")?;
        validate_sample_time(sample_time, BUILTIN_NAME)?;
        let a_value = object
            .properties
            .get("A")
            .ok_or_else(|| lqr_error(&LQR_ERROR_INVALID_MODEL, "ss object missing A property"))?;
        let b_value = object
            .properties
            .get("B")
            .ok_or_else(|| lqr_error(&LQR_ERROR_INVALID_MODEL, "ss object missing B property"))?;
        Ok(Self {
            a: RealMatrix::parse("A", a_value.clone()).await?,
            b: RealMatrix::parse("B", b_value.clone()).await?,
            q: RealMatrix::parse("Q", q.clone()).await?,
            r: RealMatrix::parse("R", r.clone()).await?,
            n: match tail {
                [] => None,
                [n] => Some(RealMatrix::parse("N", n.clone()).await?),
                _ => unreachable!(),
            },
            domain: if sample_time > 0.0 {
                LqrDomain::Discrete
            } else {
                LqrDomain::Continuous
            },
        })
    }

    fn solve(self, include_poles: bool) -> BuiltinResult<LqrEval> {
        validate_dimensions(&self)?;
        let a = self.a.matrix;
        let b = self.b.matrix;
        let q = symmetric_part(self.q.matrix);
        let r = symmetric_part(self.r.matrix);
        let n = self
            .n
            .map(|matrix| matrix.matrix)
            .unwrap_or_else(|| DMatrix::<f64>::zeros(a.nrows(), b.ncols()));
        validate_cost_weights(&q, &r, &n)?;
        let (s, k) = match self.domain {
            LqrDomain::Continuous => {
                let r_inv = r.clone().try_inverse().ok_or_else(|| {
                    lqr_error(
                        &LQR_ERROR_INVALID_DIMENSIONS,
                        "R must be nonsingular for continuous-time LQR",
                    )
                })?;
                let s = solve_care(&a, &b, &q, &r_inv, &n)?;
                let k = &r_inv * (b.transpose() * &s + n.transpose());
                (s, k)
            }
            LqrDomain::Discrete => {
                let s = solve_dare(&a, &b, &q, &r, &n)?;
                let k = discrete_gain(&a, &b, &r, &n, &s)?;
                (s, k)
            }
        };
        let poles = if include_poles {
            let closed_loop = &a - &b * &k;
            Some(closed_loop_poles(&closed_loop)?)
        } else {
            None
        };
        Ok(LqrEval { k, s, poles })
    }
}

struct LqrEval {
    k: DMatrix<f64>,
    s: DMatrix<f64>,
    poles: Option<Vec<Complex64>>,
}

impl LqrEval {
    fn outputs(&self, count: usize) -> BuiltinResult<Vec<Value>> {
        let mut outputs = Vec::with_capacity(count.min(3));
        if count >= 1 {
            outputs.push(self.k_value()?);
        }
        if count >= 2 {
            outputs.push(self.s_value()?);
        }
        if count >= 3 {
            outputs.push(self.poles_value()?);
        }
        Ok(outputs)
    }

    fn k_value(&self) -> BuiltinResult<Value> {
        matrix_value("K", &self.k)
    }

    fn s_value(&self) -> BuiltinResult<Value> {
        matrix_value("S", &self.s)
    }

    fn poles_value(&self) -> BuiltinResult<Value> {
        let poles = self.poles.as_ref().ok_or_else(|| {
            lqr_error(
                &LQR_ERROR_INTERNAL,
                "closed-loop poles were not computed for this output request",
            )
        })?;
        output_complex_column(poles.clone(), BUILTIN_NAME)
    }
}

fn validate_dimensions(request: &LqrRequest) -> BuiltinResult<()> {
    let n = request.a.rows();
    let m = request.b.cols();
    if n == 0 {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            "A must have at least one state",
        ));
    }
    if request.a.cols() != n {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            format!(
                "A must be square, got {}x{}",
                request.a.rows(),
                request.a.cols()
            ),
        ));
    }
    if request.b.rows() != n || m == 0 {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            format!(
                "B must have {n} rows and at least one input, got {}x{}",
                request.b.rows(),
                request.b.cols()
            ),
        ));
    }
    if request.q.rows() != n || request.q.cols() != n {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            format!(
                "Q must be {n}x{n}, got {}x{}",
                request.q.rows(),
                request.q.cols()
            ),
        ));
    }
    if request.r.rows() != m || request.r.cols() != m {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            format!(
                "R must be {m}x{m}, got {}x{}",
                request.r.rows(),
                request.r.cols()
            ),
        ));
    }
    if let Some(cross) = &request.n {
        if cross.rows() != n || cross.cols() != m {
            return Err(lqr_error(
                &LQR_ERROR_INVALID_DIMENSIONS,
                format!("N must be {n}x{m}, got {}x{}", cross.rows(), cross.cols()),
            ));
        }
    }
    if !is_symmetric(&request.q.matrix, 1.0e-8) {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            "Q must be symmetric",
        ));
    }
    if !is_symmetric(&request.r.matrix, 1.0e-8) {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            "R must be symmetric",
        ));
    }
    Ok(())
}

fn validate_cost_weights(
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    n: &DMatrix<f64>,
) -> BuiltinResult<()> {
    if r.clone().cholesky().is_none() {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            "R must be symmetric positive definite",
        ));
    }
    let r_inv = r.clone().try_inverse().ok_or_else(|| {
        lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            "R must be nonsingular for LQR",
        )
    })?;
    let schur = symmetric_part(q - n * r_inv * n.transpose());
    if !is_positive_semidefinite(&schur, 1.0e-8) {
        return Err(lqr_error(
            &LQR_ERROR_INVALID_DIMENSIONS,
            "[Q N; N' R] must be positive semidefinite",
        ));
    }
    Ok(())
}

fn is_positive_semidefinite(matrix: &DMatrix<f64>, tol: f64) -> bool {
    let scale = max_abs_matrix(matrix).max(1.0);
    let eigen = SymmetricEigen::new(matrix.clone());
    eigen.eigenvalues.iter().all(|value| *value >= -tol * scale)
}

fn solve_care(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r_inv: &DMatrix<f64>,
    n: &DMatrix<f64>,
) -> BuiltinResult<DMatrix<f64>> {
    let state_count = a.nrows();
    let a_bar = a - b * r_inv * n.transpose();
    let q_bar = q - n * r_inv * n.transpose();
    let g = b * r_inv * b.transpose();
    let mut h = DMatrix::<f64>::zeros(state_count * 2, state_count * 2);
    h.view_mut((0, 0), (state_count, state_count))
        .copy_from(&a_bar);
    h.view_mut((0, state_count), (state_count, state_count))
        .copy_from(&(-g));
    h.view_mut((state_count, 0), (state_count, state_count))
        .copy_from(&(-q_bar));
    h.view_mut((state_count, state_count), (state_count, state_count))
        .copy_from(&(-a_bar.transpose()));

    let sign_h = matrix_sign(h)?;
    let identity = DMatrix::<f64>::identity(state_count * 2, state_count * 2);
    let projector = (identity - sign_h) * 0.5;
    let p11 = projector
        .view((0, 0), (state_count, state_count))
        .into_owned();
    let p21 = projector
        .view((state_count, 0), (state_count, state_count))
        .into_owned();
    let p11_inv = p11.try_inverse().ok_or_else(|| {
        lqr_error(
            &LQR_ERROR_SOLVER_FAILED,
            "stable Hamiltonian invariant subspace is singular",
        )
    })?;
    let solution = symmetric_part(p21 * p11_inv);
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(lqr_error(
            &LQR_ERROR_SOLVER_FAILED,
            "Riccati solution contains non-finite values",
        ));
    }
    let residual = a.transpose() * &solution + &solution * a
        - (&solution * b + n) * r_inv * (b.transpose() * &solution + n.transpose())
        + q;
    let residual_norm = max_abs_matrix(&residual);
    let scale = 1.0
        + max_abs_matrix(q)
        + max_abs_matrix(a)
        + max_abs_matrix(b)
        + max_abs_matrix(r_inv)
        + max_abs_matrix(&solution);
    if residual_norm > 1.0e-6 * scale {
        return Err(lqr_error(
            &LQR_ERROR_SOLVER_FAILED,
            format!("Riccati residual did not converge: {residual_norm:e}"),
        ));
    }
    Ok(solution)
}

fn matrix_sign(mut matrix: DMatrix<f64>) -> BuiltinResult<DMatrix<f64>> {
    for _ in 0..SIGN_MAX_ITERS {
        let inverse = matrix.clone().try_inverse().ok_or_else(|| {
            lqr_error(
                &LQR_ERROR_SOLVER_FAILED,
                "Hamiltonian matrix sign iteration encountered a singular iterate",
            )
        })?;
        let next = (matrix.clone() + inverse) * 0.5;
        let delta = max_abs_matrix(&(next.clone() - matrix));
        let next_norm = max_abs_matrix(&next).max(1.0);
        matrix = next;
        if delta <= EPS * next_norm {
            return Ok(matrix);
        }
    }
    Err(lqr_error(
        &LQR_ERROR_SOLVER_FAILED,
        "Hamiltonian matrix sign iteration did not converge",
    ))
}

fn solve_dare(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    n: &DMatrix<f64>,
) -> BuiltinResult<DMatrix<f64>> {
    let mut solution = q.clone();
    for _ in 0..DARE_MAX_ITERS {
        let gain_inv = discrete_gain_denominator(b, r, &solution)?;
        let next = symmetric_part(
            a.transpose() * &solution * a
                - (a.transpose() * &solution * b + n)
                    * gain_inv
                    * (b.transpose() * &solution * a + n.transpose())
                + q,
        );
        if next.iter().any(|value| !value.is_finite()) {
            return Err(lqr_error(
                &LQR_ERROR_SOLVER_FAILED,
                "discrete Riccati solution contains non-finite values",
            ));
        }
        let delta = max_abs_matrix(&(next.clone() - &solution));
        let scale = 1.0 + max_abs_matrix(&next);
        solution = next;
        if delta <= EPS * scale {
            validate_dare_residual(a, b, q, r, n, &solution)?;
            return Ok(solution);
        }
    }
    Err(lqr_error(
        &LQR_ERROR_SOLVER_FAILED,
        "discrete Riccati iteration did not converge",
    ))
}

fn discrete_gain(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    r: &DMatrix<f64>,
    n: &DMatrix<f64>,
    s: &DMatrix<f64>,
) -> BuiltinResult<DMatrix<f64>> {
    Ok(discrete_gain_denominator(b, r, s)? * (b.transpose() * s * a + n.transpose()))
}

fn discrete_gain_denominator(
    b: &DMatrix<f64>,
    r: &DMatrix<f64>,
    s: &DMatrix<f64>,
) -> BuiltinResult<DMatrix<f64>> {
    (r + b.transpose() * s * b).try_inverse().ok_or_else(|| {
        lqr_error(
            &LQR_ERROR_SOLVER_FAILED,
            "R + B'*S*B is singular for discrete-time LQR",
        )
    })
}

fn validate_dare_residual(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    n: &DMatrix<f64>,
    s: &DMatrix<f64>,
) -> BuiltinResult<()> {
    let gain_inv = discrete_gain_denominator(b, r, s)?;
    let residual = a.transpose() * s * a
        - s
        - (a.transpose() * s * b + n) * gain_inv * (b.transpose() * s * a + n.transpose())
        + q;
    let residual_norm = max_abs_matrix(&residual);
    let scale = 1.0
        + max_abs_matrix(q)
        + max_abs_matrix(a)
        + max_abs_matrix(b)
        + max_abs_matrix(r)
        + max_abs_matrix(s);
    if residual_norm > 1.0e-6 * scale {
        return Err(lqr_error(
            &LQR_ERROR_SOLVER_FAILED,
            format!("discrete Riccati residual did not converge: {residual_norm:e}"),
        ));
    }
    Ok(())
}

fn closed_loop_poles(closed_loop: &DMatrix<f64>) -> BuiltinResult<Vec<Complex64>> {
    let complex = closed_loop.map(|value| Complex64::new(value, 0.0));
    let eigenvalues = complex.eigenvalues().ok_or_else(|| {
        lqr_error(
            &LQR_ERROR_SOLVER_FAILED,
            "failed to compute closed-loop poles",
        )
    })?;
    Ok(eigenvalues.iter().copied().collect())
}

fn scalar_property(object: &ObjectInstance, name: &'static str) -> BuiltinResult<f64> {
    match object.properties.get(name) {
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

fn matrix_value(label: &'static str, matrix: &DMatrix<f64>) -> BuiltinResult<Value> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new(matrix.as_slice().to_vec(), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| {
            lqr_error(
                &LQR_ERROR_INTERNAL,
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
    matrix
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn lqr_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    control_error(
        BUILTIN_NAME,
        error.identifier.unwrap_or("RunMat:lqr:Error"),
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, ResolveContext};

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("tensor"))
    }

    fn integer_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn matrix_from_value(value: &Value) -> DMatrix<f64> {
        match value {
            Value::Tensor(tensor) => {
                DMatrix::from_column_slice(tensor.rows, tensor.cols, &tensor.materialize_f64())
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lqr_double_integrator_matches_known_solution() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = block_on(lqr_builtin(
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
            ],
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

    #[test]
    fn lqr_accepts_typed_integer_matrices_at_double_boundary() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(lqr_builtin(
            integer_tensor(IntegerStorage::I16(vec![0, 0, 1, 0]), 2, 2),
            vec![
                integer_tensor(IntegerStorage::I16(vec![0, 1]), 2, 1),
                integer_tensor(IntegerStorage::U16(vec![1, 0, 0, 1]), 2, 2),
                integer_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            ],
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
    fn lqr_cross_weight_changes_gain_and_satisfies_care() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(lqr_builtin(
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![2.0, 0.0, 0.0, 3.0], 2, 2),
                Value::Num(2.0),
                tensor(vec![0.25, 0.0], 2, 1),
            ],
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
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            tensor(vec![0.0, 1.0], 2, 1),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.0),
            Vec::new(),
        ))
        .expect("ss");
        let k = block_on(lqr_builtin(
            sys,
            vec![tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2), Value::Num(1.0)],
        ))
        .expect("lqr(sys)");
        let k = matrix_from_value(&k);
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn lqr_accepts_discrete_state_space_object() {
        let sys = block_on(crate::builtins::control::ss::ss_builtin(
            tensor(vec![1.0, 0.0, 0.1, 1.0], 2, 2),
            tensor(vec![0.005, 0.1], 2, 1),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.0),
            vec![Value::Num(0.1)],
        ))
        .expect("discrete ss");
        let _guard = crate::output_count::push_output_count(Some(3));
        let result = block_on(lqr_builtin(
            sys,
            vec![tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2), Value::Num(1.0)],
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
                assert_eq!(poles.shape, vec![2, 1]);
                assert!(poles
                    .data
                    .iter()
                    .all(|(re, im)| (re * re + im * im).sqrt() < 1.0));
            }
            other => panic!("expected complex pole tensor, got {other:?}"),
        }
    }

    #[test]
    fn lqr_accepts_typed_integer_sample_time_property_on_state_space_object() {
        let sample_time = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();
        let sys = block_on(crate::builtins::control::ss::ss_builtin(
            tensor(vec![1.0, 0.0, 1.0, 1.0], 2, 2),
            tensor(vec![0.0, 1.0], 2, 1),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::Num(0.0),
            vec![Value::Tensor(sample_time)],
        ))
        .expect("discrete ss");
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(lqr_builtin(
            sys,
            vec![tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2), Value::Num(1.0)],
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
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0], 1, 1),
                Value::Num(1.0),
            ],
        ))
        .expect_err("dimension mismatch");
        assert_eq!(err.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn lqr_rejects_non_positive_definite_r() {
        let err = block_on(lqr_builtin(
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(-1.0),
            ],
        ))
        .expect_err("invalid R");
        assert_eq!(err.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
        assert!(err.to_string().contains("positive definite"));
    }

    #[test]
    fn lqr_rejects_indefinite_block_cost() {
        let err = block_on(lqr_builtin(
            tensor(vec![0.0, 0.0, 1.0, 0.0], 2, 2),
            vec![
                tensor(vec![0.0, 1.0], 2, 1),
                tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
                Value::Num(1.0),
                tensor(vec![2.0, 0.0], 2, 1),
            ],
        ))
        .expect_err("indefinite block cost");
        assert_eq!(err.identifier(), LQR_ERROR_INVALID_DIMENSIONS.identifier);
        assert!(err.to_string().contains("positive semidefinite"));
    }

    #[test]
    fn lqr_zero_requested_outputs_returns_empty_without_solving() {
        let _guard = crate::output_count::push_output_count(Some(0));
        let result = block_on(lqr_builtin(
            Value::String("not matrices".to_string()),
            Vec::new(),
        ))
        .expect("zero-output lqr skips computation");
        assert!(matches!(result, Value::OutputList(values) if values.is_empty()));
    }

    #[test]
    fn lqr_descriptor_and_type_resolver_are_registered() {
        let labels: Vec<&str> = LQR_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"K = lqr(A, B, Q, R)"));
        assert!(labels.contains(&"[K, S, e] = lqr(sys, Q, R, N)"));
        assert_eq!(
            lqr_type(&[], &ResolveContext::new(Vec::new())),
            runmat_builtins::Type::tensor()
        );
    }
}
