use crate::interpreter::errors::mex;
use crate::interpreter::stack::pop_args;
use runmat_builtins::Value;
use runmat_hir::{QualifiedName, SymbolName};
use runmat_runtime::{build_runtime_error, RuntimeError};

#[cfg(feature = "native-accel")]
fn map_prepare_builtin_args_error(err: impl std::fmt::Display) -> RuntimeError {
    mex(
        "AccelerationOperationFailed",
        &format!("prepare builtin args: {err}"),
    )
}

#[derive(Clone, Copy)]
enum VmIntrinsicBuiltin {
    Nargin,
    Nargout,
    Narginchk,
    Nargoutchk,
}

impl VmIntrinsicBuiltin {
    fn classify(name: &str) -> Option<Self> {
        match name {
            runmat_hir::NARGIN_BUILTIN_NAME => Some(Self::Nargin),
            runmat_hir::NARGOUT_BUILTIN_NAME => Some(Self::Nargout),
            runmat_hir::NARGINCHK_BUILTIN_NAME => Some(Self::Narginchk),
            runmat_hir::NARGOUTCHK_BUILTIN_NAME => Some(Self::Nargoutchk),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Nargin => runmat_hir::NARGIN_BUILTIN_NAME,
            Self::Nargout => runmat_hir::NARGOUT_BUILTIN_NAME,
            Self::Narginchk => runmat_hir::NARGINCHK_BUILTIN_NAME,
            Self::Nargoutchk => runmat_hir::NARGOUTCHK_BUILTIN_NAME,
        }
    }
}

pub fn is_vm_intrinsic_builtin(name: &str) -> bool {
    VmIntrinsicBuiltin::classify(name).is_some()
}

#[derive(Clone, Copy)]
enum VmIntrinsicExceptionBuiltin {
    Rethrow,
}

impl VmIntrinsicExceptionBuiltin {
    fn classify(name: &str) -> Option<Self> {
        match name {
            "rethrow" => Some(Self::Rethrow),
            _ => None,
        }
    }
}

#[cfg(feature = "native-accel")]
pub async fn prepare_builtin_args(name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    runmat_accelerate::prepare_builtin_args(name, args)
        .await
        .map_err(map_prepare_builtin_args_error)
}

#[cfg(not(feature = "native-accel"))]
pub async fn prepare_builtin_args(_name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    Ok(args.to_vec())
}

pub fn collect_call_args(
    stack: &mut Vec<Value>,
    arg_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    pop_args(stack, arg_count)
}

#[derive(Clone, Copy)]
enum ArityBound {
    Finite(usize),
    Unbounded,
}

impl ArityBound {
    fn permits(self, actual: usize) -> bool {
        match self {
            Self::Finite(max) => actual <= max,
            Self::Unbounded => true,
        }
    }
}

fn parse_finite_arity_bound(
    value: &Value,
    builtin: &str,
    name: &str,
) -> Result<usize, RuntimeError> {
    let number = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(mex(
                &format!("{builtin}ArgumentInvalid"),
                &format!("{builtin}: {name} must be a nonnegative integer scalar, got {other:?}"),
            ))
        }
    };

    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(mex(
            &format!("{builtin}ArgumentInvalid"),
            &format!("{builtin}: {name} must be a nonnegative integer scalar"),
        ));
    }
    if number > usize::MAX as f64 {
        return Err(mex(
            &format!("{builtin}ArgumentInvalid"),
            &format!("{builtin}: {name} exceeds the platform argument-count range"),
        ));
    }
    Ok(number as usize)
}

fn parse_max_arity_bound(value: &Value, builtin: &str) -> Result<ArityBound, RuntimeError> {
    match value {
        Value::Num(value) if value.is_infinite() && value.is_sign_positive() => {
            Ok(ArityBound::Unbounded)
        }
        Value::Tensor(tensor)
            if tensor.data.len() == 1
                && tensor.data[0].is_infinite()
                && tensor.data[0].is_sign_positive() =>
        {
            Ok(ArityBound::Unbounded)
        }
        _ => parse_finite_arity_bound(value, builtin, "maxArgs").map(ArityBound::Finite),
    }
}

fn validate_arity_bounds(
    builtin: &str,
    min: &Value,
    max: &Value,
) -> Result<(usize, ArityBound), RuntimeError> {
    let min = parse_finite_arity_bound(min, builtin, "minArgs")?;
    let max = parse_max_arity_bound(max, builtin)?;
    if let ArityBound::Finite(max_value) = max {
        if min > max_value {
            return Err(mex(
                &format!("{builtin}BoundsInvalid"),
                &format!("{builtin}: minArgs must be less than or equal to maxArgs"),
            ));
        }
    }
    Ok((min, max))
}

fn validate_narginchk(args: &[Value], actual: usize) -> Result<(), RuntimeError> {
    let (min, max) = validate_arity_bounds("Narginchk", &args[0], &args[1])?;
    if actual < min {
        return Err(mex(
            "NotEnoughInputs",
            &format!("narginchk: expected at least {min} input arguments, got {actual}"),
        ));
    }
    if !max.permits(actual) {
        return Err(mex(
            "TooManyInputs",
            &format!("narginchk: input argument count {actual} exceeds maxArgs"),
        ));
    }
    Ok(())
}

fn validate_nargoutchk(args: &[Value], actual: usize) -> Result<(), RuntimeError> {
    let (min, max) = validate_arity_bounds("Nargoutchk", &args[0], &args[1])?;
    if actual < min {
        return Err(mex(
            "NotEnoughOutputs",
            &format!("nargoutchk: expected at least {min} output arguments, got {actual}"),
        ));
    }
    if !max.permits(actual) {
        return Err(mex(
            "TooManyOutputs",
            &format!("nargoutchk: output argument count {actual} exceeds maxArgs"),
        ));
    }
    Ok(())
}

fn validate_intrinsic_arg_count(
    builtin: &str,
    actual: usize,
    expected: usize,
) -> Result<(), RuntimeError> {
    if actual < expected {
        return Err(mex(
            "NotEnoughInputs",
            &format!("{builtin} takes {expected} arguments"),
        ));
    }
    if actual > expected {
        return Err(mex(
            "TooManyInputs",
            &format!("{builtin} takes {expected} arguments"),
        ));
    }
    Ok(())
}

pub fn vm_intrinsic_builtin(
    stack: &mut Vec<Value>,
    name: &str,
    arg_count: usize,
    call_counts: &[(usize, usize)],
) -> Result<Value, RuntimeError> {
    let Some(intrinsic) = VmIntrinsicBuiltin::classify(name) else {
        return Err(mex(
            "UndefinedFunction",
            &format!("unknown VM intrinsic builtin '{name}'"),
        ));
    };
    let args = collect_call_args(stack, arg_count)?;
    match intrinsic {
        VmIntrinsicBuiltin::Nargin => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 0)?;
            let (nin, _) = call_counts.last().cloned().unwrap_or((0, 0));
            Ok(Value::Num(nin as f64))
        }
        VmIntrinsicBuiltin::Nargout => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 0)?;
            let (_, nout) = call_counts.last().cloned().unwrap_or((0, 0));
            Ok(Value::Num(nout as f64))
        }
        VmIntrinsicBuiltin::Narginchk => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 2)?;
            let (nin, _) = call_counts.last().cloned().unwrap_or((0, 0));
            validate_narginchk(&args, nin)?;
            Ok(Value::Num(0.0))
        }
        VmIntrinsicBuiltin::Nargoutchk => {
            validate_intrinsic_arg_count(intrinsic.name(), args.len(), 2)?;
            let (_, nout) = call_counts.last().cloned().unwrap_or((0, 0));
            validate_nargoutchk(&args, nout)?;
            Ok(Value::Num(0.0))
        }
    }
}

pub enum ImportedBuiltinResolution {
    Resolved(Value),
    Ambiguous(RuntimeError),
    NotFound,
}

fn imported_builtin_qualified_name(path: &[String], leaf: Option<&str>) -> Option<String> {
    if path.iter().any(|segment| segment.is_empty()) {
        return None;
    }
    let mut segments: Vec<SymbolName> = path
        .iter()
        .map(|segment| SymbolName(segment.clone()))
        .collect();
    if let Some(leaf) = leaf {
        segments.push(SymbolName(leaf.to_string()));
    }
    QualifiedName(segments).display_name()
}

pub async fn resolve_imported_builtin(
    name: &str,
    imports: &[(Vec<String>, bool)],
    prepared_primary: &[Value],
    requested_outputs: usize,
) -> Result<ImportedBuiltinResolution, RuntimeError> {
    let mut specific_matches: Vec<(String, Value)> = Vec::new();
    for (path, wildcard) in imports {
        if *wildcard {
            continue;
        }
        if path.last().map(|s| s.as_str()) == Some(name) {
            let Some(qual) = imported_builtin_qualified_name(path, None) else {
                continue;
            };
            let qual_args = prepare_builtin_args(&qual, prepared_primary).await?;
            let result = runmat_runtime::call_builtin_async_with_outputs(
                &qual,
                &qual_args,
                requested_outputs,
            )
            .await;
            if let Ok(value) = result {
                specific_matches.push((qual, value));
            }
        }
    }
    if specific_matches.len() > 1 {
        let msg = specific_matches
            .iter()
            .map(|(q, _)| q.clone())
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ImportedBuiltinResolution::Ambiguous(mex(
            "AmbiguousBuiltinImport",
            &format!("ambiguous builtin '{}' via imports: {}", name, msg),
        )));
    }
    if let Some((_, value)) = specific_matches.pop() {
        return Ok(ImportedBuiltinResolution::Resolved(value));
    }

    let mut wildcard_matches: Vec<(String, Value)> = Vec::new();
    for (path, wildcard) in imports {
        if !*wildcard || path.is_empty() {
            continue;
        }
        let Some(qual) = imported_builtin_qualified_name(path, Some(name)) else {
            continue;
        };
        let qual_args = prepare_builtin_args(&qual, prepared_primary).await?;
        let result =
            runmat_runtime::call_builtin_async_with_outputs(&qual, &qual_args, requested_outputs)
                .await;
        if let Ok(value) = result {
            wildcard_matches.push((qual, value));
        }
    }
    if wildcard_matches.len() > 1 {
        let msg = wildcard_matches
            .iter()
            .map(|(q, _)| q.clone())
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ImportedBuiltinResolution::Ambiguous(mex(
            "AmbiguousBuiltinImport",
            &format!("ambiguous builtin '{}' via wildcard imports: {}", name, msg),
        )));
    }
    if let Some((_, value)) = wildcard_matches.pop() {
        return Ok(ImportedBuiltinResolution::Resolved(value));
    }

    Ok(ImportedBuiltinResolution::NotFound)
}

pub fn rethrow_without_explicit_exception(
    name: &str,
    args: &[Value],
    last_identifier: Option<&str>,
    last_message: Option<&str>,
) -> Option<RuntimeError> {
    match VmIntrinsicExceptionBuiltin::classify(name) {
        Some(VmIntrinsicExceptionBuiltin::Rethrow) if args.is_empty() => {
            if let (Some(identifier), Some(message)) = (last_identifier, last_message) {
                return Some(
                    build_runtime_error(message.to_string())
                        .with_identifier(identifier.to_string())
                        .build(),
                );
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{imported_builtin_qualified_name, rethrow_without_explicit_exception};

    #[test]
    fn imported_builtin_qualified_name_uses_typed_segments() {
        let specific = imported_builtin_qualified_name(&["PkgF".into(), "foo".into()], None);
        assert_eq!(specific.as_deref(), Some("PkgF.foo"));

        let wildcard = imported_builtin_qualified_name(&["PkgF".into()], Some("foo"));
        assert_eq!(wildcard.as_deref(), Some("PkgF.foo"));
    }

    #[test]
    fn imported_builtin_qualified_name_ignores_empty_segments_without_fallback() {
        let empty = imported_builtin_qualified_name(&[], None);
        assert_eq!(empty, None);

        let only_empty = imported_builtin_qualified_name(&["".into()], None);
        assert_eq!(only_empty, None);

        let mixed_empty = imported_builtin_qualified_name(&["PkgF".into(), "".into()], Some("foo"));
        assert_eq!(mixed_empty, None);
    }

    #[test]
    fn rethrow_preserves_last_exception_identifier() {
        let err = rethrow_without_explicit_exception(
            "rethrow",
            &[],
            Some("RunMat:Original"),
            Some("boom"),
        )
        .expect("rethrow should preserve prior exception");
        assert_eq!(err.identifier(), Some("RunMat:Original"));
        assert_eq!(err.message(), "boom");
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn prepare_builtin_args_error_maps_to_accel_identifier() {
        let err = super::map_prepare_builtin_args_error("boom");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
        assert!(err.message().contains("prepare builtin args"));
        assert!(err.message().contains("boom"));
    }
}
