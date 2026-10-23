//! MATLAB-compatible `cheb2ord` Chebyshev Type II order selection.

use runmat_builtins::{
    BuiltinErrorDescriptor, CHEB2ORD_ERROR_ARG_COUNT, CHEB2ORD_ERROR_INTERNAL,
    CHEB2ORD_ERROR_INVALID_ATTENUATION, CHEB2ORD_ERROR_INVALID_FREQUENCY,
    CHEB2ORD_ERROR_INVALID_OPTION, CHEB2ORD_ERROR_TOO_MANY_OUTPUTS,
    CHEB2ORD_INTEGER_ATTENUATION_EXTENSION, CHEB2ORD_INTEGER_FREQUENCY_EXTENSION,
    CHEB2ORD_LOGICAL_INPUT_EXTENSION, CHEB2ORD_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{real_vector_to_row_value, SignalCooperation};
use crate::builtins::math::signal::order_selection::{
    chebyshev_v_pass_stop, checked_order, classify_kind, db_excess_log_ratio,
    find_natural_frequency, option_text, parse_positive_db, real_edges, transform_edges,
    unwarp_frequency, validate_critical_frequencies, FilterKind, OrderEdgeError, OrderFamily,
};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, GpuGatherRetry, RuntimeError,
};

const BUILTIN_NAME: &str = "cheb2ord";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::cheb2ord")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cheb2ord",
    op_kind: GpuOpKind::Custom("chebyshev2-order-selection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Chebyshev Type II order selection is host-side analysis. Interactive resident input is an explicit RunMat-only gather extension; outputs remain host values.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::cheb2ord")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cheb2ord",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "cheb2ord materialises scalar/vector design parameters and is not fused.",
};

#[derive(Clone, Debug)]
struct Cheb2ordResult {
    order: usize,
    wn: Vec<f64>,
}

fn cheb2ord_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cheb2ord_error_with_message(error.message, error)
}

fn cheb2ord_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cheb2ord_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cheb2ord_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cheb2ord_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let mut owner = build_runtime_error(format!("{}: {}", error.message, source.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        owner = owner.with_identifier(identifier);
    }
    owner.build()
}

fn frequency_error(error: OrderEdgeError) -> RuntimeError {
    match error {
        OrderEdgeError::Invalid(detail) => {
            cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail)
        }
        OrderEdgeError::Operational(source) => {
            cheb2ord_error_with_source(&CHEB2ORD_ERROR_INVALID_FREQUENCY, *source)
        }
    }
}

#[runtime_builtin(
    name = "cheb2ord",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::cheb2ord"
)]
async fn cheb2ord_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate_in_context(wp, ws, rp, rs, rest.view(), runtime.as_ref()).await
}

pub async fn evaluate(
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: &[Value],
) -> BuiltinResult<Value> {
    evaluate_in_context(
        wp,
        ws,
        rp,
        rs,
        crate::call::FunctionArgumentView::from_slice(rest),
        None,
    )
    .await
}

async fn evaluate_in_context(
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    if rest.len() > 1 {
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_ARG_COUNT));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_TOO_MANY_OUTPUTS));
    }
    let analog = if let Some(option) = rest.first() {
        match option_text(option)
            .map_err(|source| {
                let mut error = cheb2ord_error(&CHEB2ORD_ERROR_INVALID_OPTION);
                error.source = Some(Box::new(source));
                error
            })?
            .as_deref()
        {
            Some("s") => true,
            Some(_) => return Err(cheb2ord_error(&CHEB2ORD_ERROR_INVALID_OPTION)),
            None => return Err(cheb2ord_error(&CHEB2ORD_ERROR_INVALID_OPTION)),
        }
    } else {
        false
    };
    ensure_compatibility(&wp, &ws, &rp, &rs)?;
    let wp = prepare_numeric_value(wp, OrderInputRole::Wp, &mut cooperation).await?;
    cooperation.completed(1).await?;
    let ws = prepare_numeric_value(ws, OrderInputRole::Ws, &mut cooperation).await?;
    cooperation.completed(1).await?;
    let rp = prepare_numeric_value(rp, OrderInputRole::Rp, &mut cooperation).await?;
    cooperation.completed(1).await?;
    let rs = prepare_numeric_value(rs, OrderInputRole::Rs, &mut cooperation).await?;
    cooperation.completed(1).await?;
    let wp = real_edges(BUILTIN_NAME, "Wp", wp)
        .await
        .map_err(frequency_error)?;
    let ws = real_edges(BUILTIN_NAME, "Ws", ws)
        .await
        .map_err(frequency_error)?;
    let rp = parse_positive_db(BUILTIN_NAME, "Rp", &rp).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    let rs = parse_positive_db(BUILTIN_NAME, "Rs", &rs).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    if rs <= rp {
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_INVALID_ATTENUATION));
    }

    let result = compute_cheb2ord(&wp, &ws, rp, rs, analog)?;
    cooperation.completed(1).await?;
    output_result(result)
}

fn output_result(result: Cheb2ordResult) -> BuiltinResult<Value> {
    let order = Value::Num(result.order as f64);
    let wn = if result.wn.len() == 1 {
        Value::Num(result.wn[0])
    } else {
        real_vector_to_row_value(result.wn)
            .map_err(|err| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INTERNAL, err.message()))?
    };
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![order]));
        }
        if out_count == 2 {
            return Ok(Value::output_list(vec![order, wn]));
        }
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_TOO_MANY_OUTPUTS));
    }
    Ok(order)
}

fn ensure_compatibility(wp: &Value, ws: &Value, rp: &Value, rs: &Value) -> BuiltinResult<()> {
    if is_typed_integer(wp) || is_typed_integer(ws) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHEB2ORD_INTEGER_FREQUENCY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_typed_integer(rp) || is_typed_integer(rs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHEB2ORD_INTEGER_ATTENUATION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if [wp, ws, rp, rs].into_iter().any(|value| is_logical(value)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHEB2ORD_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if [wp, ws, rp, rs]
        .into_iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHEB2ORD_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

#[derive(Clone, Copy)]
enum OrderInputRole {
    Wp,
    Ws,
    Rp,
    Rs,
}

impl OrderInputRole {
    fn label(self) -> &'static str {
        match self {
            Self::Wp => "Wp",
            Self::Ws => "Ws",
            Self::Rp => "Rp",
            Self::Rs => "Rs",
        }
    }

    fn extension(self) -> &'static runmat_builtins::BuiltinExtensionDescriptor {
        match self {
            Self::Wp | Self::Ws => &CHEB2ORD_INTEGER_FREQUENCY_EXTENSION,
            Self::Rp | Self::Rs => &CHEB2ORD_INTEGER_ATTENUATION_EXTENSION,
        }
    }

    fn error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Wp | Self::Ws => &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            Self::Rp | Self::Rs => &CHEB2ORD_ERROR_INVALID_ATTENUATION,
        }
    }
}

async fn prepare_numeric_value(
    value: Value,
    role: OrderInputRole,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let value = if is_typed_integer(&value) {
        let prepared = cooperation
            .prepare_integer_receipt(&value, role.extension())
            .await?;
        let descriptor = role.error();
        if !prepared.is_exact() {
            return Err(cheb2ord_error_with_detail(
                descriptor,
                format!(
                    "integer {} values must be exactly representable as double",
                    role.label()
                ),
            ));
        }
        if let Some(payload) = prepared.into_resident_payload() {
            payload
                .into_value()
                .map_err(|detail| cheb2ord_error_with_detail(descriptor, detail))?
        } else {
            value
        }
    } else {
        value
    };
    let host = gather_if_needed_async(&value)
        .await
        .map_err(|source| cheb2ord_error_with_source(&CHEB2ORD_ERROR_INTERNAL, source))?;
    Ok(host)
}

fn compute_cheb2ord(
    wp: &[f64],
    ws: &[f64],
    rp: f64,
    rs: f64,
    analog: bool,
) -> BuiltinResult<Cheb2ordResult> {
    if wp.len() != ws.len() {
        return Err(cheb2ord_error_with_detail(
            &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            "Wp and Ws must have the same length",
        ));
    }
    let kind = classify_kind(wp, ws)
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    crate::builtins::math::signal::order_selection::validate_domain(wp, analog, "Wp")
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    crate::builtins::math::signal::order_selection::validate_domain(ws, analog, "Ws")
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;

    let wp_work = transform_edges(wp, analog);
    let ws_work = transform_edges(ws, analog);
    let (nat, passband) =
        find_natural_frequency(kind, &wp_work, &ws_work, rp, rs, OrderFamily::Chebyshev).map_err(
            |detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail),
        )?;

    let log_ratio = db_excess_log_ratio(rp, rs).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    let v_pass_stop = chebyshev_v_pass_stop(log_ratio).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    let order = checked_order(v_pass_stop / nat.acosh())
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;

    let prototype_edge = 1.0 / (v_pass_stop / order as f64).cosh();
    if !prototype_edge.is_finite() || prototype_edge <= 0.0 {
        return Err(cheb2ord_error_with_detail(
            &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            "computed prototype edge is not finite and positive",
        ));
    }
    let natural = natural_frequency(kind, &passband, prototype_edge)?;
    let wn: Vec<f64> = if analog {
        natural
    } else {
        natural.into_iter().map(unwarp_frequency).collect()
    };
    validate_critical_frequencies(&wn, analog)
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    Ok(Cheb2ordResult { order, wn })
}

fn natural_frequency(
    kind: FilterKind,
    passband: &[f64],
    prototype_edge: f64,
) -> BuiltinResult<Vec<f64>> {
    match kind {
        FilterKind::Lowpass => Ok(vec![passband[0] / prototype_edge]),
        FilterKind::Highpass => Ok(vec![passband[0] * prototype_edge]),
        FilterKind::Bandpass => {
            let bandwidth = passband[1] - passband[0];
            let center_sq = passband[0] * passband[1];
            let disc = bandwidth * bandwidth / (4.0 * prototype_edge * prototype_edge) + center_sq;
            let root = disc.sqrt();
            let first = (passband[0] - passband[1]) / (2.0 * prototype_edge) + root;
            let second = center_sq / first;
            let mut edges = vec![first, second];
            edges.sort_by(|a, b| a.total_cmp(b));
            Ok(edges)
        }
        FilterKind::Bandstop => {
            let center_sq = passband[0] * passband[1];
            let disc = prototype_edge * prototype_edge * (passband[1] - passband[0]).powi(2) / 4.0
                + center_sq;
            let root = disc.sqrt();
            let first = prototype_edge * (passband[0] - passband[1]) / 2.0 + root;
            let second = center_sq / first;
            let mut edges = vec![first, second];
            edges.sort_by(|a, b| a.total_cmp(b));
            Ok(edges)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{builtin_catalog_entry_by_name, builtin_function_by_name};
    use runmat_builtins::{CHEB2ORD_EXTENSIONS, CHEB2ORD_INTEGER_CAPABILITIES};
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn integer_input_roles_bind_exact_extension_and_error_family() {
        for (role, label, extension, error) in [
            (
                OrderInputRole::Wp,
                "Wp",
                &CHEB2ORD_INTEGER_FREQUENCY_EXTENSION,
                &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            ),
            (
                OrderInputRole::Ws,
                "Ws",
                &CHEB2ORD_INTEGER_FREQUENCY_EXTENSION,
                &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            ),
            (
                OrderInputRole::Rp,
                "Rp",
                &CHEB2ORD_INTEGER_ATTENUATION_EXTENSION,
                &CHEB2ORD_ERROR_INVALID_ATTENUATION,
            ),
            (
                OrderInputRole::Rs,
                "Rs",
                &CHEB2ORD_INTEGER_ATTENUATION_EXTENSION,
                &CHEB2ORD_ERROR_INVALID_ATTENUATION,
            ),
        ] {
            assert_eq!(role.label(), label);
            assert_eq!(role.extension().id, extension.id);
            assert_eq!(role.error().identifier, error.identifier);
        }
    }

    fn call(
        wp: Value,
        ws: Value,
        rp: Value,
        rs: Value,
        rest: &[Value],
        outputs: Option<usize>,
    ) -> BuiltinResult<Value> {
        let _guard = outputs
            .map(|count| crate::output_count::push_output_count(Some(count)))
            .transpose()
            .expect("output-count scope");
        block_on(evaluate(wp, ws, rp, rs, rest))
    }

    fn outputs(value: Value) -> (f64, Vec<f64>) {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        let order = match &values[0] {
            Value::Num(value) => *value,
            other => panic!("expected scalar order, got {other:?}"),
        };
        let wn = match &values[1] {
            Value::Num(value) => vec![*value],
            Value::Tensor(tensor) => tensor.materialize_f64().clone(),
            other => panic!("expected Wn, got {other:?}"),
        };
        (order, wn)
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("typed integer tensor")
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_catalog_entry_by_name(BUILTIN_NAME).expect("cheb2ord catalog entry");
        assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
        assert_eq!(builtin.bindings.len(), 1);
        let compiled = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
        assert_eq!(compiled.len(), 1);
        assert_eq!(
            compiled[0].identity,
            builtin.binding_identity(&builtin.bindings[0])
        );
        let descriptor = builtin.descriptor;
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs)"));
    }

    #[test]
    fn production_highpass_repro_resolves() {
        let out = call(
            Value::Num(5000.0 / (20000.0 / 2.0)),
            Value::Num(4700.0 / (20000.0 / 2.0)),
            Value::Num(1.0),
            Value::Num(25.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 10.0);
        assert_eq!(wn.len(), 1);
        assert!((wn[0] - 0.472_175_327_294_252_1).abs() < 1e-12);
    }

    #[test]
    fn digital_lowpass_and_highpass_return_stopband_side_edges() {
        let low = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(low);
        assert_eq!(n, 6.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 0.2);
        assert!(wn[0] < 0.3);

        let high = call(
            Value::Num(0.5),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(high);
        assert_eq!(n, 5.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 0.3);
        assert!(wn[0] < 0.5);
    }

    #[test]
    fn bandpass_and_bandstop_return_two_edges() {
        let wp = Tensor::new(vec![0.3, 0.5], vec![1, 2]).unwrap();
        let ws = Tensor::new(vec![0.2, 0.6], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(wp),
            Value::Tensor(ws),
            Value::Num(1.0),
            Value::Num(30.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert!(n >= 1.0);
        assert_eq!(wn.len(), 2);
        assert!(wn[0] < 0.3);
        assert!(wn[1] > 0.5);
        assert!(wn[0] < wn[1]);

        let wp = Tensor::new(vec![0.1, 0.6], vec![1, 2]).unwrap();
        let ws = Tensor::new(vec![0.2, 0.5], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(wp),
            Value::Tensor(ws),
            Value::Num(3.0),
            Value::Num(60.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 7.0);
        assert_eq!(wn.len(), 2);
        assert!((wn[0] - 0.197_768_738_814_104_7).abs() < 1e-12);
        assert!((wn[1] - 0.503_814_419_296_416_3).abs() < 1e-12);
    }

    #[test]
    fn analog_lowpass_keeps_radian_cutoff() {
        let out = call(
            Value::Num(10.0),
            Value::Num(20.0),
            Value::Num(1.0),
            Value::Num(40.0),
            &[Value::from("s")],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 5.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 20.0);
    }

    #[test]
    fn analog_lowpass_reads_typed_integer_design_args_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = call(
            Value::Tensor(integer_tensor(IntegerStorage::I16(vec![10]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::I16(vec![20]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::U16(vec![40]), vec![1, 1])),
            &[Value::from("s")],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 5.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 20.0);
    }

    #[test]
    fn analog_design_executes_every_integer_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = [
            (
                IntValue::I8(10),
                IntValue::I8(20),
                IntValue::I8(1),
                IntValue::I8(40),
            ),
            (
                IntValue::I16(10),
                IntValue::I16(20),
                IntValue::I16(1),
                IntValue::I16(40),
            ),
            (
                IntValue::I32(10),
                IntValue::I32(20),
                IntValue::I32(1),
                IntValue::I32(40),
            ),
            (
                IntValue::I64(10),
                IntValue::I64(20),
                IntValue::I64(1),
                IntValue::I64(40),
            ),
            (
                IntValue::U8(10),
                IntValue::U8(20),
                IntValue::U8(1),
                IntValue::U8(40),
            ),
            (
                IntValue::U16(10),
                IntValue::U16(20),
                IntValue::U16(1),
                IntValue::U16(40),
            ),
            (
                IntValue::U32(10),
                IntValue::U32(20),
                IntValue::U32(1),
                IntValue::U32(40),
            ),
            (
                IntValue::U64(10),
                IntValue::U64(20),
                IntValue::U64(1),
                IntValue::U64(40),
            ),
        ];
        for (wp, ws, rp, rs) in cases {
            let out = call(
                Value::Int(wp),
                Value::Int(ws),
                Value::Int(rp),
                Value::Int(rs),
                &[Value::from("s")],
                Some(2),
            )
            .unwrap();
            let (order, wn) = outputs(out);
            assert_eq!(order, 5.0);
            assert!(wn[0] > 10.0 && wn[0] < 20.0);
        }
    }

    #[test]
    fn integer_metadata_and_compatibility_gates_are_role_specific() {
        assert_eq!(CHEB2ORD_INTEGER_CAPABILITIES.len(), 2);
        assert!(CHEB2ORD_INTEGER_CAPABILITIES.iter().all(|capability| {
            capability
                .inputs
                .iter()
                .all(|input| input.classes.len() == 8)
        }));
        assert_eq!(CHEB2ORD_EXTENSIONS.len(), 4);

        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let frequency = call(
            Value::Int(runmat_value::IntValue::I16(10)),
            Value::Num(20.0),
            Value::Num(1.0),
            Value::Num(40.0),
            &[Value::from("s")],
            Some(1),
        )
        .unwrap_err();
        assert_eq!(
            frequency.identifier(),
            Some("RunMat:compatibility:Cheb2ordIntegerFrequencyExtension")
        );

        let attenuation = call(
            Value::Num(10.0),
            Value::Num(20.0),
            Value::Int(runmat_value::IntValue::U16(1)),
            Value::Num(40.0),
            &[Value::from("s")],
            Some(1),
        )
        .unwrap_err();
        assert_eq!(
            attenuation.identifier(),
            Some("RunMat:compatibility:Cheb2ordIntegerAttenuationExtension")
        );
    }

    #[test]
    fn wide_integer_extension_rejects_before_lossy_double_computation() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = call(
            Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
            Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 2)),
            Value::Num(1.0),
            Value::Num(40.0),
            &[Value::from("s")],
            Some(1),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidFrequency"));
        assert!(err.message().contains("exactly representable as double"));
    }

    #[test]
    fn exact_wide_integer_multiple_passes_shared_binary64_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let exact = block_on(prepare_numeric_value(
            Value::Int(IntValue::U64((1_u64 << 53) + 2)),
            OrderInputRole::Wp,
            &mut cooperation,
        ))
        .expect("exact binary64 multiple is admissible");
        assert!(matches!(exact, Value::Int(IntValue::U64(value)) if value == (1_u64 << 53) + 2));
    }

    #[test]
    fn logical_and_resident_gates_run_before_provider_access() {
        {
            let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
            let logical = call(
                Value::Bool(true),
                Value::Num(20.0),
                Value::Num(1.0),
                Value::Num(40.0),
                &[Value::from("s")],
                Some(1),
            )
            .unwrap_err();
            assert_eq!(
                logical.identifier(),
                Some("RunMat:compatibility:Cheb2ordLogicalInputExtension")
            );
        }

        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![10]), vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            {
                let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = call(
                    Value::GpuTensor(handle.clone()),
                    Value::Num(20.0),
                    Value::Num(1.0),
                    Value::Num(40.0),
                    &[Value::from("s")],
                    Some(1),
                )
                .unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:Cheb2ordIntegerFrequencyExtension")
                );
            }
            let out = call(
                Value::GpuTensor(handle.clone()),
                Value::Num(20.0),
                Value::Num(1.0),
                Value::Num(40.0),
                &[Value::from("s")],
                Some(1),
            )
            .unwrap();
            assert!(matches!(out, Value::OutputList(values) if values.len() == 1));
            let _ = handle.release();
        });
    }

    #[test]
    fn output_count_and_invalid_specs() {
        let one = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(1),
        )
        .unwrap();
        let Value::OutputList(values) = one else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 1);

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(3),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:TooManyOutputs"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(40.0),
            Value::Num(1.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidAttenuation"));

        let err = call(
            Value::Num(1.2),
            Value::Num(1.4),
            Value::Num(1.0),
            Value::Num(20.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidFrequency"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(20.0),
            &[Value::from("z")],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidOption"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.2000001),
            Value::Num(1.0),
            Value::Num(1.0e6),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidFrequency"));
    }
}
