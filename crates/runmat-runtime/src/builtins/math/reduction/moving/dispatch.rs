//! Rooted call/control preparation, then the existing typed numeric actions.
use super::*;
use super::{
    admission::MovingCall,
    keywords::{keyword, Keyword},
};
use crate::call::BuiltinArguments;
use crate::context::RuntimeContext;

pub(super) async fn moving_builtin(
    runtime: Option<&RuntimeContext>,
    name: &'static str,
    op: MovingOp,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    let work = MovingCall::new(runtime, name)?;
    if let Some(Value::GpuTensor(handle)) = args.first() {
        gpu_helpers::require_resident_compute_owner(handle, || {
            invalid_input(
                name,
                format!("{name}: resident input has no exact owning provider"),
            )
        })?;
    }
    let result = async {
        if args.len() < 2 { return Err(invalid_argument(name, format!("{name}: expected at least A and k"))); }
        for argument in &args { crate::builtins::common::validation::reject_typed_complex_integer(argument, name)?; }
        if matches!(&args[0], Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved) {
            return Err(invalid_input(name, format!("{name}: complex GPU moving windows are unsupported; explicitly gather the input first")));
        }
        if op.uses_variance_normalization() && is_real_typed_integer_value(&args[0]) { return Err(invalid_input(name, format!("{name}: integer data input is not supported"))); }
        let rest = args.view().range(2..)?;
        if args.view().range(1..)?.iter().any(crate::builtins::common::validation::value_contains_explicit_gpu) {
            return Err(invalid_argument(name, format!("{name}: explicitly gather resident window and option controls before calling this form")));
        }
        if op.uses_variance_normalization() && moving_stdvar_uses_typed_integer_control(rest) {
            crate::compatibility::ensure_builtin_extension_enabled(stdvar_typed_integer_control_extension(op), name)?;
        }
        let resident_real = matches!(&args[0], Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real);
        let sample_points = rest.iter().any(|value| keyword(value) == Some(Keyword::SamplePoints));
        if !resident_real || sample_points { work.host()?; }
        // Retain original operands and roots across every control gather. The
        // data operand stays resident until the selected numeric action.
        let gathered = work.wait(args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
            if index == 0 { Ok(args[0].clone()) }
            else if matches!(&args[index], Value::GpuTensor(_)) {
                work.host()?;
                work.wait(crate::gather_if_needed_async(&args[index])).await
            } else { Ok(args[index].clone()) }
        })).await?;
        let parsed = work.wait(super::arguments::parse_moving_args(name, op, &gathered[1], gathered.view().range(2..)?)).await?;
        if let Value::GpuTensor(handle) = &gathered[0] {
            if parsed.sample_points.is_some() { crate::compatibility::ensure_builtin_extension_enabled(gpu_sample_points_extension(op), name)?; }
            if op == MovingOp::Median && parsed.window.checked_count_len(name)?.is_some_and(|length| length > 31) {
                crate::compatibility::ensure_builtin_extension_enabled(&movmedian::GPU_LARGE_WINDOW_EXTENSION, name)?;
            }
            return super::provider::moving_gpu(&work, name, op, handle.clone(), &parsed).await;
        }
        work.host()?;
        match &gathered[0] {
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(*re, *im)], vec![1, 1]).map_err(|err| internal(name, err))?;
                super::complex::apply(name, op, &tensor, &parsed)
            }
            Value::ComplexTensor(tensor) => super::complex::apply(name, op, tensor, &parsed),
            other => {
                let tensor = tensor::value_into_tensor_for(name, other.clone()).map_err(|err| invalid_input(name, err))?;
                moving_real(name, op, tensor, &parsed).map(tensor::tensor_into_value)
            }
        }
    }.await;
    work.finish_value(result)
}

fn moving_stdvar_uses_typed_integer_control(rest: crate::call::FunctionArgumentView<'_>) -> bool {
    let mut index = 0;
    while index < rest.len() {
        match keyword(&rest[index]) {
            Some(Keyword::Endpoints | Keyword::SamplePoints) => index = index.saturating_add(2),
            Some(Keyword::Omit | Keyword::Include) => index += 1,
            _ if is_real_typed_integer_value(&rest[index]) => return true,
            _ => index += 1,
        }
    }
    false
}
