//! MATLAB language identifier and keyword introspection builtins.

use crate::builtins::common::identifiers::{
    is_matlab_keyword, MATLAB_KEYWORDS, MATLAB_NAME_LENGTH_MAX,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::identifiers")]
pub const NAMELENGTHMAX_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "namelengthmax",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host language constant; no provider dispatch or GPU residency changes are needed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::identifiers")]
pub const NAMELENGTHMAX_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "namelengthmax",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Language constant resolved on the host; not a fusion operation.",
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::identifiers")]
pub const ISKEYWORD_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "iskeyword",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host text/language predicate; gpuArray inputs are invalid and no provider hooks apply.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::identifiers")]
pub const ISKEYWORD_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "iskeyword",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata/text predicate that does not participate in fusion planning.",
};

const OUT_NUMERIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "l",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Maximum supported MATLAB identifier length.",
}];

const OUT_BOOL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the text is a MATLAB keyword.",
}];

const OUT_KEYWORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell column of MATLAB keyword character vectors.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar or character vector to test.",
}];

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const NAMELENGTHMAX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "l = namelengthmax",
    inputs: &NO_INPUTS,
    outputs: &OUT_NUMERIC,
}];

const ISKEYWORD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = iskeyword(s)",
        inputs: &IN_TEXT,
        outputs: &OUT_BOOL,
    },
    BuiltinSignatureDescriptor {
        label: "k = iskeyword",
        inputs: &NO_INPUTS,
        outputs: &OUT_KEYWORDS,
    },
];

const ERROR_NAMELENGTHMAX_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NAMELENGTHMAX.ARG_COUNT",
    identifier: Some("RunMat:namelengthmax:InvalidArgument"),
    when: "Any input argument is supplied.",
    message: "namelengthmax: expected no input arguments",
};

const ERROR_ISKEYWORD_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISKEYWORD.ARG_COUNT",
    identifier: Some("RunMat:iskeyword:InvalidArgument"),
    when: "More than one input argument is supplied.",
    message: "iskeyword: expected zero or one input argument",
};

const ERROR_ISKEYWORD_TEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISKEYWORD.TEXT",
    identifier: Some("RunMat:iskeyword:InvalidText"),
    when: "Input is not a string scalar or character vector.",
    message: "iskeyword: input must be a string scalar or character vector",
};

const ERROR_ISKEYWORD_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISKEYWORD.INTERNAL",
    identifier: Some("RunMat:iskeyword:Internal"),
    when: "RunMat cannot construct the keyword-list cell array.",
    message: "iskeyword: failed to build keyword list",
};

pub const NAMELENGTHMAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NAMELENGTHMAX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ERROR_NAMELENGTHMAX_ARG_COUNT],
};

pub const ISKEYWORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISKEYWORD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ERROR_ISKEYWORD_ARG_COUNT,
        ERROR_ISKEYWORD_TEXT,
        ERROR_ISKEYWORD_INTERNAL,
    ],
};

#[runtime_builtin(
    name = "namelengthmax",
    category = "introspection",
    summary = "Return the maximum length for MATLAB identifiers.",
    keywords = "namelengthmax,identifier,variable name,field name,language",
    accel = "metadata",
    type_resolver(namelengthmax_type),
    descriptor(crate::builtins::introspection::identifiers::NAMELENGTHMAX_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::identifiers"
)]
fn namelengthmax_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(error("namelengthmax", &ERROR_NAMELENGTHMAX_ARG_COUNT));
    }
    Ok(Value::Num(MATLAB_NAME_LENGTH_MAX as f64))
}

#[runtime_builtin(
    name = "iskeyword",
    category = "introspection",
    summary = "Test MATLAB reserved keywords or return the keyword list.",
    keywords = "iskeyword,keyword,identifier,language",
    accel = "metadata",
    type_resolver(iskeyword_type),
    descriptor(crate::builtins::introspection::identifiers::ISKEYWORD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::identifiers"
)]
fn iskeyword_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => keyword_cell(),
        [value] => Ok(Value::Bool(is_matlab_keyword(&text_scalar(value)?))),
        _ => Err(error("iskeyword", &ERROR_ISKEYWORD_ARG_COUNT)),
    }
}

fn keyword_cell() -> BuiltinResult<Value> {
    let values = MATLAB_KEYWORDS
        .iter()
        .map(|keyword| Value::CharArray(CharArray::new_row(keyword)))
        .collect::<Vec<_>>();
    CellArray::new(values, MATLAB_KEYWORDS.len(), 1)
        .map(Value::Cell)
        .map_err(|err| {
            error_with_detail(
                "iskeyword",
                &ERROR_ISKEYWORD_INTERNAL,
                format!("iskeyword: failed to build keyword list ({err})"),
            )
        })
}

fn text_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows <= 1 => Ok(chars.data.iter().collect()),
        _ => Err(error("iskeyword", &ERROR_ISKEYWORD_TEXT)),
    }
}

fn error(builtin: &str, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    error_with_detail(builtin, descriptor, descriptor.message)
}

fn error_with_detail(
    builtin: &str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn namelengthmax_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
}

fn iskeyword_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        Type::cell_of(Type::String)
    } else {
        Type::Bool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::StringArray;

    #[test]
    fn namelengthmax_returns_current_identifier_limit() {
        assert_eq!(
            namelengthmax_builtin(Vec::new()).expect("namelengthmax"),
            Value::Num(2048.0)
        );
    }

    #[test]
    fn namelengthmax_rejects_arguments() {
        assert!(namelengthmax_builtin(vec![Value::Num(1.0)]).is_err());
    }

    #[test]
    fn namelengthmax_resolves_as_numeric_scalar() {
        assert_eq!(
            namelengthmax_type(&[], &ResolveContext::default()),
            Type::Num
        );
    }

    #[test]
    fn iskeyword_lists_keywords_as_cell_column() {
        let Value::Cell(cell) = iskeyword_builtin(Vec::new()).expect("iskeyword") else {
            panic!("expected cell")
        };
        assert_eq!(cell.shape, vec![20, 1]);
        assert_eq!(
            cell.data.first(),
            Some(&Value::CharArray(CharArray::new_row("break")))
        );
        assert_eq!(
            cell.data.last(),
            Some(&Value::CharArray(CharArray::new_row("while")))
        );
    }

    #[test]
    fn iskeyword_accepts_string_scalar_and_char_vector() {
        assert_eq!(
            iskeyword_builtin(vec![Value::String("for".to_string())]).expect("keyword"),
            Value::Bool(true)
        );
        assert_eq!(
            iskeyword_builtin(vec![Value::CharArray(CharArray::new_row("plot"))]).expect("keyword"),
            Value::Bool(false)
        );
        assert_eq!(
            iskeyword_builtin(vec![Value::StringArray(
                StringArray::new(vec!["end".to_string()], vec![1, 1]).expect("string")
            )])
            .expect("keyword"),
            Value::Bool(true)
        );
    }

    #[test]
    fn iskeyword_rejects_non_scalar_text_and_other_values() {
        assert!(iskeyword_builtin(vec![Value::StringArray(
            StringArray::new(vec!["for".to_string(), "end".to_string()], vec![1, 2])
                .expect("string array")
        )])
        .is_err());
        assert!(iskeyword_builtin(vec![Value::Num(1.0)]).is_err());
    }
}
