use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axes_metadata_snapshot, encode_axes_handle, figure_has_sg_title, legend_entries_snapshot,
    FigureHandle, PlotObjectKind,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_array_type;
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "ancestor";

const ANCESTOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching graphics ancestor handle, or an empty handle array when none exists.",
}];

const ANCESTOR_INPUTS_TYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle or array of graphics handles to inspect.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object type name or cell/string array of type names.",
    },
];

const ANCESTOR_INPUTS_TOPLEVEL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Graphics handle or array of graphics handles to inspect.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object type name or cell/string array of type names.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Use 'toplevel' to return the highest matching ancestor.",
    },
];

const ANCESTOR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "p = ancestor(h, type)",
        inputs: &ANCESTOR_INPUTS_TYPE,
        outputs: &ANCESTOR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = ancestor(h, type, 'toplevel')",
        inputs: &ANCESTOR_INPUTS_TOPLEVEL,
        outputs: &ANCESTOR_OUTPUT,
    },
];

const ANCESTOR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANCESTOR.INVALID_ARGUMENT",
    identifier: Some("RunMat:ancestor:InvalidArgument"),
    when: "Arguments are missing, the type selector is invalid, or the option is not 'toplevel'.",
    message: "ancestor: invalid argument",
};

const ANCESTOR_ERRORS: [BuiltinErrorDescriptor; 1] = [ANCESTOR_ERROR_INVALID_ARGUMENT];

pub const ANCESTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANCESTOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ANCESTOR_ERRORS,
};

#[runtime_builtin(
    name = "ancestor",
    category = "plotting",
    summary = "Find a graphics object ancestor with a requested type.",
    keywords = "ancestor,graphics,handle,parent,plotting",
    suppress_auto_output = true,
    type_resolver(handle_array_type),
    descriptor(crate::builtins::plotting::ancestor::ANCESTOR_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ancestor"
)]
pub fn ancestor_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(invalid_argument(
            "expected ancestor(h, type) or ancestor(h, type, 'toplevel')",
        ));
    }

    let types = TypeSelector::parse(&args[1])?;
    let toplevel = match args.get(2) {
        Some(value) => {
            let Some(option) = text_scalar(value) else {
                return Err(invalid_argument("option must be 'toplevel'"));
            };
            if !option.trim().eq_ignore_ascii_case("toplevel") {
                return Err(invalid_argument("option must be 'toplevel'"));
            }
            true
        }
        None => false,
    };

    match &args[0] {
        Value::Tensor(tensor) => ancestor_array(tensor, &types, toplevel),
        value => Ok(
            match scalar_handle(value).and_then(|handle| ancestor_scalar(handle, &types, toplevel))
            {
                Some(handle) => Value::Num(handle),
                None => empty_handle_array(),
            },
        ),
    }
}

fn ancestor_array(
    tensor: &Tensor,
    types: &TypeSelector,
    toplevel: bool,
) -> crate::BuiltinResult<Value> {
    let values = tensor_utils::tensor_values_f64_cow(tensor);
    if values.len() == 1 {
        return Ok(match ancestor_scalar(values[0], types, toplevel) {
            Some(handle) => Value::Num(handle),
            None => empty_handle_array(),
        });
    }

    let data: Vec<f64> = values
        .iter()
        .filter_map(|&handle| ancestor_scalar(handle, types, toplevel))
        .collect();
    if data.is_empty() {
        return Ok(empty_handle_array());
    }
    let len = data.len();
    Ok(Value::Tensor(
        Tensor::new(data, vec![1, len]).expect("ancestor handle row"),
    ))
}

fn ancestor_scalar(handle_value: f64, types: &TypeSelector, toplevel: bool) -> Option<f64> {
    let handle = resolve_plot_handle(&Value::Num(handle_value), BUILTIN_NAME).ok()?;
    if !ancestor_handle_exists(&handle) {
        return None;
    }
    let chain = ancestor_chain(handle_value, handle);
    if toplevel {
        chain
            .iter()
            .rev()
            .find(|candidate| types.matches(candidate.kind))
            .map(|candidate| candidate.handle)
    } else {
        chain
            .iter()
            .find(|candidate| types.matches(candidate.kind))
            .map(|candidate| candidate.handle)
    }
}

fn ancestor_handle_exists(handle: &PlotHandle) -> bool {
    match handle {
        PlotHandle::Legend(figure, axes_index) => legend_object_exists(*figure, *axes_index),
        PlotHandle::Text(figure, _, PlotObjectKind::SuperTitle) => figure_has_sg_title(*figure),
        _ => true,
    }
}

fn legend_object_exists(figure: FigureHandle, axes_index: usize) -> bool {
    axes_metadata_snapshot(figure, axes_index)
        .map(|meta| meta.legend_enabled)
        .unwrap_or(false)
        && legend_entries_snapshot(figure, axes_index)
            .map(|entries| !entries.is_empty())
            .unwrap_or(false)
}

fn ancestor_chain(raw_handle: f64, handle: PlotHandle) -> Vec<AncestorCandidate> {
    match handle {
        PlotHandle::Root => vec![AncestorCandidate::new("root", 0.0)],
        PlotHandle::Figure(figure) => vec![
            AncestorCandidate::new("figure", figure.as_u32() as f64),
            AncestorCandidate::new("root", 0.0),
        ],
        PlotHandle::Axes(figure, axes_index) => {
            chain_from_axes("axes", raw_handle, figure, axes_index)
        }
        PlotHandle::Ruler(figure, axes_index, _) => {
            chain_from_axes("numericruler", raw_handle, figure, axes_index)
        }
        PlotHandle::Text(figure, _, PlotObjectKind::SuperTitle) => vec![
            AncestorCandidate::new("text", raw_handle),
            AncestorCandidate::new("figure", figure.as_u32() as f64),
            AncestorCandidate::new("root", 0.0),
        ],
        PlotHandle::Text(figure, axes_index, _) => {
            chain_from_axes("text", raw_handle, figure, axes_index)
        }
        PlotHandle::Legend(figure, axes_index) => {
            chain_from_axes("legend", raw_handle, figure, axes_index)
        }
        PlotHandle::PlotChild(_, state) => {
            let (figure, axes_index) = state.figure_axes();
            chain_from_axes(state.type_name(), raw_handle, figure, axes_index)
        }
    }
}

fn chain_from_axes(
    kind: &'static str,
    handle: f64,
    figure: FigureHandle,
    axes_index: usize,
) -> Vec<AncestorCandidate> {
    vec![
        AncestorCandidate::new(kind, handle),
        AncestorCandidate::new("axes", encode_axes_handle(figure, axes_index)),
        AncestorCandidate::new("figure", figure.as_u32() as f64),
        AncestorCandidate::new("root", 0.0),
    ]
}

#[derive(Clone, Copy)]
struct AncestorCandidate {
    kind: &'static str,
    handle: f64,
}

impl AncestorCandidate {
    const fn new(kind: &'static str, handle: f64) -> Self {
        Self { kind, handle }
    }
}

struct TypeSelector {
    names: Vec<String>,
}

impl TypeSelector {
    fn parse(value: &Value) -> crate::BuiltinResult<Self> {
        let mut names = Vec::new();
        match value {
            Value::Cell(cell) => {
                for item in &cell.data {
                    push_type_name(item, &mut names)?;
                }
            }
            Value::StringArray(strings) => {
                for item in &strings.data {
                    push_normalized_name(item, &mut names)?;
                }
            }
            _ => push_type_name(value, &mut names)?,
        }
        if names.is_empty() {
            return Err(invalid_argument(
                "type selector must contain at least one type name",
            ));
        }
        Ok(Self { names })
    }

    fn matches(&self, kind: &str) -> bool {
        self.names
            .iter()
            .any(|name| name == kind || (name == "ruler" && kind == "numericruler"))
    }
}

fn push_type_name(value: &Value, names: &mut Vec<String>) -> crate::BuiltinResult<()> {
    match value {
        Value::StringArray(strings) => {
            for item in &strings.data {
                push_normalized_name(item, names)?;
            }
            Ok(())
        }
        _ => {
            let Some(text) = text_scalar(value) else {
                return Err(invalid_argument("type selector entries must be strings"));
            };
            push_normalized_name(&text, names)
        }
    }
}

fn push_normalized_name(raw: &str, names: &mut Vec<String>) -> crate::BuiltinResult<()> {
    let name = raw.trim().to_ascii_lowercase();
    if name.is_empty() {
        return Err(invalid_argument("type selector entries must not be empty"));
    }
    names.push(name);
    Ok(())
}

fn scalar_handle(value: &Value) -> Option<f64> {
    match value {
        Value::Num(v) => Some(*v),
        Value::Int(i) => Some(i.to_f64()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        Value::String(text) => Some(text.clone()),
        Value::StringArray(strings) if strings.data.len() == 1 => strings.data.first().cloned(),
        _ => None,
    }
}

fn empty_handle_array() -> Value {
    Value::Tensor(
        Tensor::new_with_dtype(Vec::new(), vec![0, 0], NumericDType::F64)
            .expect("empty handle tensor shape is valid"),
    )
}

fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        ANCESTOR_ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = ANCESTOR_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::legend::legend_builtin;
    use crate::builtins::plotting::plot::plot_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state::{
        current_axes_state, reset_hold_state_for_run, PlotTestLockGuard,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_plot_state};
    use futures::executor::block_on;
    use runmat_builtins::{CellArray, Tensor, Value};

    fn setup() -> PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn vec_tensor(data: Vec<f64>) -> Value {
        Value::Tensor(Tensor::new(data, vec![1, 2]).expect("test tensor"))
    }

    #[test]
    fn descriptor_lists_supported_forms() {
        let labels = ANCESTOR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"p = ancestor(h, type)"));
        assert!(labels.contains(&"p = ancestor(h, type, 'toplevel')"));
    }

    #[test]
    fn line_ancestor_includes_self_axes_figure_and_root() {
        let _guard = setup();
        let line_handle = block_on(plot_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )]))
        .unwrap();

        let axes = current_axes_state();
        let axes_handle = encode_axes_handle(axes.handle, axes.active_index);
        let figure_handle = axes.handle.as_u32() as f64;

        assert_eq!(
            ancestor_builtin(vec![Value::Num(line_handle), Value::String("line".into())]).unwrap(),
            Value::Num(line_handle)
        );
        assert_eq!(
            ancestor_builtin(vec![Value::Num(line_handle), Value::String("axes".into())]).unwrap(),
            Value::Num(axes_handle)
        );
        assert_eq!(
            ancestor_builtin(vec![
                Value::Num(line_handle),
                Value::String("figure".into())
            ])
            .unwrap(),
            Value::Num(figure_handle)
        );
        assert_eq!(
            ancestor_builtin(vec![Value::Num(line_handle), Value::String("root".into())]).unwrap(),
            Value::Num(0.0)
        );
    }

    #[test]
    fn type_lists_return_nearest_or_top_level_match() {
        let _guard = setup();
        let line_handle = block_on(plot_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )]))
        .unwrap();
        let axes = current_axes_state();
        let axes_handle = encode_axes_handle(axes.handle, axes.active_index);
        let figure_handle = axes.handle.as_u32() as f64;
        let types = CellArray::new(
            vec![Value::String("axes".into()), Value::String("figure".into())],
            1,
            2,
        )
        .unwrap();

        assert_eq!(
            ancestor_builtin(vec![Value::Num(line_handle), Value::Cell(types.clone())]).unwrap(),
            Value::Num(axes_handle)
        );
        assert_eq!(
            ancestor_builtin(vec![
                Value::Num(line_handle),
                Value::Cell(types),
                Value::String("toplevel".into())
            ])
            .unwrap(),
            Value::Num(figure_handle)
        );
    }

    #[test]
    fn pseudo_objects_use_encoded_parent_chain() {
        let _guard = setup();
        let _line = block_on(plot_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )]))
        .unwrap();
        let axes = current_axes_state();
        let axes_handle = encode_axes_handle(axes.handle, axes.active_index);
        let x_axis =
            get_builtin(vec![Value::Num(axes_handle), Value::String("XAxis".into())]).unwrap();
        let Value::Num(x_axis_handle) = x_axis else {
            panic!("XAxis should be numeric ruler handle");
        };

        assert_eq!(
            ancestor_builtin(vec![
                Value::Num(x_axis_handle),
                Value::String("numericruler".into())
            ])
            .unwrap(),
            Value::Num(x_axis_handle)
        );
        assert_eq!(
            ancestor_builtin(vec![
                Value::Num(x_axis_handle),
                Value::String("axes".into())
            ])
            .unwrap(),
            Value::Num(axes_handle)
        );
    }

    #[test]
    fn absent_optional_pseudo_objects_return_empty() {
        let _guard = setup();
        let axes = current_axes_state();
        let axes_handle = encode_axes_handle(axes.handle, axes.active_index);
        let legend = get_builtin(vec![
            Value::Num(axes_handle),
            Value::String("Legend".into()),
        ])
        .unwrap();
        let Value::Num(legend_handle) = legend else {
            panic!("Legend should be numeric pseudo-object handle");
        };

        let result = ancestor_builtin(vec![
            Value::Num(legend_handle),
            Value::String("legend".into()),
        ])
        .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected empty tensor for absent legend");
        };
        assert_eq!(tensor.shape, vec![0, 0]);
        assert!(tensor.materialize_f64().is_empty());
    }

    #[test]
    fn created_legend_matches_itself_and_axes() {
        let _guard = setup();
        let line_handle = block_on(plot_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )]))
        .unwrap();
        set_builtin(vec![
            Value::Num(line_handle),
            Value::String("DisplayName".into()),
            Value::String("signal".into()),
        ])
        .unwrap();
        let legend_handle = legend_builtin(vec![]).unwrap();
        let axes = current_axes_state();
        let axes_handle = encode_axes_handle(axes.handle, axes.active_index);

        assert_eq!(
            ancestor_builtin(vec![
                Value::Num(legend_handle),
                Value::String("legend".into())
            ])
            .unwrap(),
            Value::Num(legend_handle)
        );
        assert_eq!(
            ancestor_builtin(vec![
                Value::Num(legend_handle),
                Value::String("axes".into())
            ])
            .unwrap(),
            Value::Num(axes_handle)
        );
    }

    #[test]
    fn invalid_scalar_handle_returns_empty() {
        let _guard = setup();
        let result =
            ancestor_builtin(vec![Value::Num(f64::NAN), Value::String("axes".into())]).unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected empty tensor");
        };
        assert_eq!(tensor.shape, vec![0, 0]);
        assert!(tensor.materialize_f64().is_empty());
    }

    #[test]
    fn vector_handles_return_matches_without_placeholder_entries() {
        let _guard = setup();
        let line_handle = block_on(plot_builtin(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )]))
        .unwrap();
        let axes = current_axes_state();
        let axes_handle = encode_axes_handle(axes.handle, axes.active_index);

        let result = ancestor_builtin(vec![
            vec_tensor(vec![line_handle, f64::NAN]),
            Value::String("axes".into()),
        ])
        .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.materialize_f64()[0], axes_handle);
    }

    #[test]
    fn root_matches_itself() {
        let _guard = setup();
        assert_eq!(
            ancestor_builtin(vec![Value::Num(0.0), Value::String("root".into())]).unwrap(),
            Value::Num(0.0)
        );
    }

    #[test]
    fn typed_integer_tensor_handle_reads_storage_exactly() {
        let _guard = setup();
        let handle = Tensor::new_integer(runmat_builtins::IntegerStorage::U8(vec![0]), vec![1, 1])
            .expect("typed handle");

        assert_eq!(
            ancestor_builtin(vec![Value::Tensor(handle), Value::String("root".into())]).unwrap(),
            Value::Num(0.0)
        );
    }
}
