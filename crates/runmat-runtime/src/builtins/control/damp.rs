//! Damping ratio and natural-frequency analysis for control models.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ObjectInstance, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    control_error, output_complex_column, polynomial_roots, scalar_f64, validate_sample_time,
    TfModel, EPS, TF_CLASS,
};
use crate::builtins::control::type_resolvers::damp_type;
use crate::{dispatcher, BuiltinResult};

const BUILTIN_NAME: &str = "damp";
const SS_CLASS: &str = "ss";

const DAMP_PARAM_WN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "wn",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Natural frequencies as an N-by-1 column vector.",
};
const DAMP_PARAM_ZETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "zeta",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Damping ratios as an N-by-1 column vector.",
};
const DAMP_PARAM_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Model poles as an N-by-1 real or complex column vector.",
};
const DAMP_OUTPUT_WN: [BuiltinParamDescriptor; 1] = [DAMP_PARAM_WN];
const DAMP_OUTPUT_WNZETA: [BuiltinParamDescriptor; 2] = [DAMP_PARAM_WN, DAMP_PARAM_ZETA];
const DAMP_OUTPUT_WNZETAP: [BuiltinParamDescriptor; 3] =
    [DAMP_PARAM_WN, DAMP_PARAM_ZETA, DAMP_PARAM_P];
const DAMP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model or ss state-space model.",
}];
const DAMP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "wn = damp(sys)",
        inputs: &DAMP_INPUTS,
        outputs: &DAMP_OUTPUT_WN,
    },
    BuiltinSignatureDescriptor {
        label: "[wn, zeta] = damp(sys)",
        inputs: &DAMP_INPUTS,
        outputs: &DAMP_OUTPUT_WNZETA,
    },
    BuiltinSignatureDescriptor {
        label: "[wn, zeta, p] = damp(sys)",
        inputs: &DAMP_INPUTS,
        outputs: &DAMP_OUTPUT_WNZETAP,
    },
];
const DAMP_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.DAMP.INVALID_MODEL",
        identifier: Some("RunMat:damp:InvalidModel"),
        when: "Input system is malformed or missing model metadata.",
        message: "damp: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.DAMP.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:damp:UnsupportedModel"),
        when: "Model class is unsupported.",
        message: "damp: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.DAMP.OUTPUT_COUNT",
        identifier: Some("RunMat:damp:OutputCount"),
        when: "More than three outputs are requested.",
        message: "damp: too many output arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.DAMP.INTERNAL",
        identifier: Some("RunMat:damp:Internal"),
        when: "Pole extraction or output construction failed.",
        message: "damp: internal error",
    },
];
pub const DAMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DAMP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DAMP_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::damp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "damp",
    op_kind: GpuOpKind::Custom("control-damping"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "damp computes pole-derived control metadata on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::damp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "damp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "damp is model analysis and is not fused.",
};

#[runtime_builtin(
    name = "damp",
    category = "control",
    summary = "Return natural frequencies, damping ratios, and poles for control models.",
    keywords = "damp,damping ratio,natural frequency,poles,control system,tf,ss",
    type_resolver(damp_type),
    descriptor(crate::builtins::control::damp::DAMP_DESCRIPTOR),
    builtin_path = "crate::builtins::control::damp"
)]
async fn damp_builtin(sys: Value) -> BuiltinResult<Value> {
    let eval = DampingEval::from_value(sys).await?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![eval.wn_value()?])),
        Some(2) => Ok(Value::OutputList(vec![
            eval.wn_value()?,
            eval.zeta_value()?,
        ])),
        Some(3) => Ok(Value::OutputList(vec![
            eval.wn_value()?,
            eval.zeta_value()?,
            eval.poles_value()?,
        ])),
        Some(_) => Err(damp_error(
            "RunMat:damp:OutputCount",
            "damp: at most three outputs are supported",
        )),
        None => eval.wn_value(),
    }
}

#[derive(Clone, Debug)]
struct DampingEval {
    modes: Vec<DampingMode>,
}

#[derive(Clone, Copy, Debug)]
struct DampingMode {
    pole: Complex64,
    wn: f64,
    zeta: f64,
}

impl DampingEval {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        let gathered = dispatcher::gather_if_needed_async(&value).await?;
        match gathered {
            Value::Object(object) if object.is_class(TF_CLASS) => {
                let model = TfModel::from_value(Value::Object(object), BUILTIN_NAME)?;
                let poles = polynomial_roots(&model.denominator, BUILTIN_NAME)?;
                Ok(Self::from_poles(poles, model.sample_time))
            }
            Value::Object(object) if object.is_class(SS_CLASS) => {
                let (poles, sample_time) = ss_poles(&object)?;
                Ok(Self::from_poles(poles, sample_time))
            }
            Value::Object(object) => Err(damp_error(
                "RunMat:damp:UnsupportedModel",
                format!(
                    "damp: unsupported model class '{}'; supported classes are tf and ss",
                    object.class_name
                ),
            )),
            other => Err(damp_error(
                "RunMat:damp:InvalidModel",
                format!("damp: expected a tf or ss object, got {other:?}"),
            )),
        }
    }

    fn from_poles(poles: Vec<Complex64>, sample_time: f64) -> Self {
        let mut modes = poles
            .into_iter()
            .map(|pole| {
                if sample_time > 0.0 && pole.norm() <= EPS {
                    return DampingMode {
                        pole,
                        wn: f64::INFINITY,
                        zeta: 1.0,
                    };
                }
                let equivalent = if sample_time > 0.0 {
                    pole.ln() / sample_time
                } else {
                    pole
                };
                let wn = equivalent.norm();
                let zeta = if wn <= EPS {
                    f64::NAN
                } else {
                    -equivalent.re / wn
                };
                DampingMode { pole, wn, zeta }
            })
            .collect::<Vec<_>>();
        modes.sort_by(compare_modes);
        Self { modes }
    }

    fn wn_value(&self) -> BuiltinResult<Value> {
        real_column(self.modes.iter().map(|mode| mode.wn).collect())
    }

    fn zeta_value(&self) -> BuiltinResult<Value> {
        real_column(self.modes.iter().map(|mode| mode.zeta).collect())
    }

    fn poles_value(&self) -> BuiltinResult<Value> {
        output_complex_column(
            self.modes.iter().map(|mode| mode.pole).collect(),
            BUILTIN_NAME,
        )
    }
}

fn compare_modes(lhs: &DampingMode, rhs: &DampingMode) -> std::cmp::Ordering {
    lhs.wn
        .total_cmp(&rhs.wn)
        .then_with(|| lhs.zeta.total_cmp(&rhs.zeta))
        .then_with(|| lhs.pole.re.total_cmp(&rhs.pole.re))
        .then_with(|| lhs.pole.im.total_cmp(&rhs.pole.im))
}

fn ss_poles(object: &ObjectInstance) -> BuiltinResult<(Vec<Complex64>, f64)> {
    let a = matrix_property(object, "A")?;
    let sample_time = scalar_property(object, "Ts")?;
    validate_sample_time(sample_time, BUILTIN_NAME)?;
    let eigenvalues = a.eigenvalues().ok_or_else(|| {
        damp_error(
            "RunMat:damp:Internal",
            "damp: failed to compute state matrix eigenvalues",
        )
    })?;
    Ok((eigenvalues.iter().copied().collect(), sample_time))
}

fn matrix_property(
    object: &ObjectInstance,
    name: &'static str,
) -> BuiltinResult<DMatrix<Complex64>> {
    let value = object.properties.get(name).ok_or_else(|| {
        damp_error(
            "RunMat:damp:InvalidModel",
            format!("damp: ss object is missing {name}"),
        )
    })?;
    let tensor = match value {
        Value::Tensor(tensor) => tensor.clone(),
        Value::Num(n) => Tensor::new(vec![*n], vec![1, 1]).map_err(|err| {
            damp_error(
                "RunMat:damp:Internal",
                format!("damp: failed to build scalar matrix: {err}"),
            )
        })?,
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
            damp_error(
                "RunMat:damp:Internal",
                format!("damp: failed to build scalar matrix: {err}"),
            )
        })?,
        other => {
            return Err(damp_error(
                "RunMat:damp:UnsupportedModel",
                format!("damp: ss {name} must be a finite real matrix, got {other:?}"),
            ));
        }
    };
    if tensor.shape.len() != 2 || tensor.rows != tensor.cols {
        return Err(damp_error(
            "RunMat:damp:InvalidModel",
            format!("damp: ss {name} must be square, got {:?}", tensor.shape),
        ));
    }
    if tensor.data.iter().any(|value| !value.is_finite()) {
        return Err(damp_error(
            "RunMat:damp:UnsupportedModel",
            format!("damp: ss {name} must contain only finite real values"),
        ));
    }
    let mut matrix = DMatrix::<Complex64>::zeros(tensor.rows, tensor.cols);
    for col in 0..tensor.cols {
        for row in 0..tensor.rows {
            matrix[(row, col)] = Complex64::new(tensor.data[row + col * tensor.rows], 0.0);
        }
    }
    Ok(matrix)
}

fn scalar_property(object: &ObjectInstance, name: &'static str) -> BuiltinResult<f64> {
    let value = object.properties.get(name).ok_or_else(|| {
        damp_error(
            "RunMat:damp:InvalidModel",
            format!("damp: ss object is missing {name}"),
        )
    })?;
    scalar_f64(value, name, BUILTIN_NAME)
}

fn real_column(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    Tensor::new(data, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(|err| {
            damp_error(
                "RunMat:damp:Internal",
                format!("damp: failed to build output tensor: {err}"),
            )
        })
}

fn damp_error(identifier: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    control_error(BUILTIN_NAME, identifier, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::ComplexTensor;

    fn tf(numerator: Value, denominator: Value, rest: &[Value]) -> Value {
        let mut args = vec![numerator, denominator];
        args.extend_from_slice(rest);
        block_on(crate::call_builtin_async("tf", &args)).expect("tf")
    }

    fn ss(a: Value, b: Value, c: Value, d: Value, rest: &[Value]) -> Value {
        let mut args = vec![a, b, c, d];
        args.extend_from_slice(rest);
        block_on(crate::call_builtin_async("ss", &args)).expect("ss")
    }

    fn damp_outputs(sys: Value, count: usize) -> Vec<Value> {
        let _guard = crate::output_count::push_output_count(Some(count));
        let Value::OutputList(outputs) = block_on(damp_builtin(sys)).expect("damp") else {
            panic!("expected output list");
        };
        outputs
    }

    fn tensor_data(value: &Value) -> &[f64] {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor, got {value:?}");
        };
        &tensor.data
    }

    #[test]
    fn damp_descriptor_signatures_cover_multi_output_forms() {
        let labels: Vec<&str> = DAMP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"wn = damp(sys)"));
        assert!(labels.contains(&"[wn, zeta] = damp(sys)"));
        assert!(labels.contains(&"[wn, zeta, p] = damp(sys)"));
    }

    #[test]
    fn damp_tf_second_order_returns_frequency_ratio_and_poles() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![100.0, 50.0, 400.0], vec![1, 3]).unwrap()),
            &[],
        );

        let outputs = damp_outputs(sys, 3);
        let wn = tensor_data(&outputs[0]);
        let zeta = tensor_data(&outputs[1]);
        assert_eq!(wn.len(), 2);
        assert_eq!(zeta.len(), 2);
        for value in wn {
            assert!((*value - 2.0).abs() < 1.0e-8);
        }
        for value in zeta {
            assert!((*value - 0.125).abs() < 1.0e-8);
        }
        match &outputs[2] {
            Value::ComplexTensor(poles) => {
                assert_eq!(poles.shape, vec![2, 1]);
                assert!(poles.data.iter().all(|(re, _)| (*re + 0.25).abs() < 1.0e-8));
                assert!(poles
                    .data
                    .iter()
                    .any(|(_, im)| (*im - 1.984313483298443).abs() < 1.0e-8));
                assert!(poles
                    .data
                    .iter()
                    .any(|(_, im)| (*im + 1.984313483298443).abs() < 1.0e-8));
            }
            other => panic!("expected complex poles, got {other:?}"),
        }
    }

    #[test]
    fn damp_tf_first_order_returns_real_pole() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            &[],
        );

        let outputs = damp_outputs(sys, 3);
        assert!((tensor_data(&outputs[0])[0] - 2.0).abs() < 1.0e-12);
        assert!((tensor_data(&outputs[1])[0] - 1.0).abs() < 1.0e-12);
        assert_eq!(tensor_data(&outputs[2]), &[-2.0]);
    }

    #[test]
    fn damp_discrete_tf_uses_log_equivalent_poles() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap()),
            &[Value::Num(0.1)],
        );

        let outputs = damp_outputs(sys, 3);
        assert!((tensor_data(&outputs[0])[0] - 6.931471805599453).abs() < 1.0e-10);
        assert!((tensor_data(&outputs[1])[0] - 1.0).abs() < 1.0e-12);
        assert!((tensor_data(&outputs[2])[0] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn damp_discrete_zero_pole_reports_deadbeat_damping() {
        let report = DampingEval::from_poles(vec![Complex64::new(0.0, 0.0)], 0.1);
        assert_eq!(report.modes.len(), 1);
        assert!(report.modes[0].wn.is_infinite());
        assert_eq!(report.modes[0].zeta, 1.0);
    }

    #[test]
    fn damp_ss_uses_state_matrix_eigenvalues() {
        let sys = ss(
            Value::Tensor(Tensor::new(vec![0.0, -4.0, 1.0, -0.5], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            &[],
        );

        let outputs = damp_outputs(sys, 3);
        for value in tensor_data(&outputs[0]) {
            assert!((*value - 2.0).abs() < 1.0e-8);
        }
        for value in tensor_data(&outputs[1]) {
            assert!((*value - 0.125).abs() < 1.0e-8);
        }
        assert!(
            matches!(&outputs[2], Value::ComplexTensor(ComplexTensor { shape, .. }) if shape == &vec![2, 1])
        );
    }

    #[test]
    fn damp_rejects_more_than_three_outputs() {
        let sys = tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            &[],
        );

        let _guard = crate::output_count::push_output_count(Some(4));
        let err = block_on(damp_builtin(sys)).expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), Some("RunMat:damp:OutputCount"));
    }
}
