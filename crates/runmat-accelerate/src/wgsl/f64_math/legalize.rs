use std::collections::BTreeSet;
use std::fmt::{self, Write};
use std::ops::Range;

use super::intrinsic::{Intrinsic, Shape};
use super::include_family;

#[derive(Debug)]
pub(crate) enum PreparationError {
    Parse(String),
    Validate(String),
    InvalidSpan { intrinsic: Intrinsic },
    ConflictingEdits,
    RemainingIntrinsic { intrinsic: Intrinsic },
    ReservedHelperCollision(String),
    Allocation(String),
    SourceSizeOverflow,
    ConstantIntrinsic { intrinsic: Intrinsic },
}

impl fmt::Display for PreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(detail) => write!(f, "shader parsing failed: {detail}"),
            Self::Validate(detail) => write!(f, "shader validation failed: {detail}"),
            Self::InvalidSpan { intrinsic } => {
                write!(f, "invalid typed callee span for {intrinsic:?}")
            }
            Self::ConflictingEdits => f.write_str("overlapping typed shader edits"),
            Self::RemainingIntrinsic { intrinsic } => {
                write!(f, "unsupported binary64 intrinsic remains: {intrinsic:?}")
            }
            Self::ReservedHelperCollision(name) => {
                write!(f, "shader declares reserved math helper {name}")
            }
            Self::Allocation(detail) => write!(f, "shader preparation allocation failed: {detail}"),
            Self::SourceSizeOverflow => f.write_str("shader preparation source size overflows"),
            Self::ConstantIntrinsic { intrinsic } => write!(
                f,
                "unsupported unevaluated binary64 constant intrinsic {intrinsic:?}"
            ),
        }
    }
}
impl std::error::Error for PreparationError {}

pub(crate) struct PreparedShader {
    source: String,
    requires_f64_math: bool,
}

impl PreparedShader {
    pub(crate) fn source(&self) -> &str {
        &self.source
    }
    pub(crate) fn into_source(self) -> String {
        self.source
    }
    pub(crate) fn requires_f64_math(&self) -> bool {
        self.requires_f64_math
    }
}

struct Edit {
    span: Range<usize>,
    intrinsic: Intrinsic,
    shape: Shape,
}

fn parse(source: &str) -> Result<(naga::Module, naga::valid::ModuleInfo), PreparationError> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| PreparationError::Parse(error.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| PreparationError::Validate(error.emit_to_string(source)))?;
    Ok((module, info))
}

fn collect_function(
    source: &str,
    module: &naga::Module,
    function: &naga::Function,
    info: &naga::valid::FunctionInfo,
    edits: &mut Vec<Edit>,
) -> Result<(), PreparationError> {
    for (handle, expression) in function.expressions.iter() {
        let naga::Expression::Math { fun, arg, .. } = *expression else {
            continue;
        };
        let Some(intrinsic) = Intrinsic::from_math(fun) else {
            continue;
        };
        let Some(shape) = Shape::of(info[arg].ty.inner_with(&module.types)) else {
            continue;
        };
        let span = function
            .expressions
            .get_span(handle)
            .to_range()
            .ok_or(PreparationError::InvalidSpan { intrinsic })?;
        if source.get(span.clone()) != Some(intrinsic.spelling()) {
            return Err(PreparationError::InvalidSpan { intrinsic });
        }
        edits.push(Edit {
            span,
            intrinsic,
            shape,
        });
    }
    Ok(())
}

fn collect(
    source: &str,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<Vec<Edit>, PreparationError> {
    // A runtime helper call cannot appear in a WGSL constant/override
    // expression. Normal constant folding has removed these operations; fail
    // closed if a future frontend leaves one for SPIR-V emission.
    for (_, expression) in module.global_expressions.iter() {
        if let naga::Expression::Math { fun, arg, .. } = *expression {
            if let Some(intrinsic) = Intrinsic::from_math(fun) {
                if Shape::of(info[arg].inner_with(&module.types)).is_some() {
                    return Err(PreparationError::ConstantIntrinsic { intrinsic });
                }
            }
        }
    }
    let mut edits = Vec::new();
    for (handle, function) in module.functions.iter() {
        collect_function(source, module, function, &info[handle], &mut edits)?;
    }
    for (index, entry) in module.entry_points.iter().enumerate() {
        collect_function(
            source,
            module,
            &entry.function,
            info.get_entry_point(index),
            &mut edits,
        )?;
    }
    Ok(edits)
}

pub(crate) fn prepare(source: &str) -> Result<PreparedShader, PreparationError> {
    let (module, info) = parse(source)?;
    let mut edits = collect(source, &module, &info)?;
    let requires_f64_math = module.types.iter().any(|(_, ty)| match ty.inner {
        naga::TypeInner::Scalar(scalar)
        | naga::TypeInner::Vector { scalar, .. }
        | naga::TypeInner::Matrix { scalar, .. } => {
            scalar.kind == naga::ScalarKind::Float && scalar.width == 8
        }
        _ => false,
    });
    if edits.is_empty() {
        return Ok(PreparedShader {
            source: source.to_owned(),
            requires_f64_math,
        });
    }
    let selected: BTreeSet<_> = edits
        .iter()
        .map(|edit| (edit.intrinsic, edit.shape))
        .collect();
    let mut helpers = String::new();
    append_helpers(&mut helpers, &selected);
    // Resolve helper identity through parsed declarations, not substring markers.
    let helper_module = naga::front::wgsl::parse_str(&helpers)
        .map_err(|error| PreparationError::Parse(error.emit_to_string(&helpers)))?;
    let mut omit = Vec::new();
    for (helper_handle, helper) in helper_module.functions.iter() {
        let Some(name) = helper.name.as_deref() else {
            continue;
        };
        for (handle, function) in module.functions.iter() {
            if function.name.as_deref() != Some(name) {
                continue;
            }
            let original_span = module.functions.get_span(handle).to_range();
            let helper_span = helper_module.functions.get_span(helper_handle).to_range();
            let identical = original_span
                .and_then(|span| source.get(span))
                .zip(helper_span.clone().and_then(|span| helpers.get(span)))
                .is_some_and(|(a, b)| a == b);
            if !identical {
                return Err(PreparationError::ReservedHelperCollision(name.to_owned()));
            }
            if let Some(mut span) = helper_span {
                // Naga24's declaration span ends at the final body statement,
                // before whitespace and the function's closing brace. This
                // extension reads only our canonical helper source, never
                // searches a user source for a builtin name or call boundary.
                while helpers
                    .as_bytes()
                    .get(span.end)
                    .is_some_and(u8::is_ascii_whitespace)
                {
                    span.end += 1;
                }
                if helpers.as_bytes().get(span.end) != Some(&b'}') {
                    return Err(PreparationError::ReservedHelperCollision(name.to_owned()));
                }
                span.end += 1;
                omit.push(span);
            }
        }
    }
    omit.sort_by_key(|span| span.start);
    for span in omit.into_iter().rev() {
        helpers.replace_range(span, "");
    }
    edits.sort_by_key(|edit| (edit.span.start, edit.span.end));
    edits.dedup_by(|a, b| a.span == b.span && a.intrinsic == b.intrinsic && a.shape == b.shape);
    let replacement_bytes = edits
        .iter()
        .try_fold(0usize, |bytes, edit| {
            bytes.checked_add(edit.intrinsic.helper(edit.shape).len() - edit.span.len())
        })
        .ok_or(PreparationError::SourceSizeOverflow)?;
    // The compiler owns exactly one final-source allocation with a checked
    // bound; this does not alter the original producer's measured byte receipt.
    let source_bytes = source
        .len()
        .checked_add(helpers.len())
        .and_then(|bytes| bytes.checked_add(replacement_bytes))
        .and_then(|bytes| bytes.checked_add(1))
        .ok_or(PreparationError::SourceSizeOverflow)?;
    let mut output = String::new();
    output
        .try_reserve_exact(source_bytes)
        .map_err(|error| PreparationError::Allocation(error.to_string()))?;
    let mut cursor = 0;
    for edit in edits {
        if edit.span.start < cursor {
            return Err(PreparationError::ConflictingEdits);
        }
        output.push_str(&source[cursor..edit.span.start]);
        output.push_str(&edit.intrinsic.helper(edit.shape));
        cursor = edit.span.end;
    }
    output.push_str(&source[cursor..]);
    output.push('\n');
    output.push_str(&helpers);
    let (module, info) = parse(&output)?;
    if let Some(edit) = collect(&output, &module, &info)?.first() {
        return Err(PreparationError::RemainingIntrinsic {
            intrinsic: edit.intrinsic,
        });
    }
    Ok(PreparedShader {
        source: output,
        requires_f64_math: true,
    })
}

fn append_helpers(output: &mut String, selected: &BTreeSet<(Intrinsic, Shape)>) {
    // Family closure is deliberately explicit: a selected numerical family
    // owns the full dependency contract of all functions in its source bundle.
    let mut families = BTreeSet::new();
    for (intrinsic, _) in selected {
        include_family(*intrinsic, &mut families);
    }
    for family in families {
        output.push_str(family.source());
        output.push('\n');
    }
    for &(intrinsic, shape) in selected {
        let Some(width) = shape.vector_width() else {
            continue;
        };
        let name = intrinsic.helper(shape);
        let scalar = intrinsic.helper(Shape::Scalar);
        write!(output, "fn {name}(a: vec{width}<f64>").expect("String formatting");
        if intrinsic.arguments() == 2 {
            write!(output, ", b: vec{width}<f64>").expect("String formatting");
        }
        write!(output, ") -> vec{width}<f64> {{ return vec{width}<f64>(")
            .expect("String formatting");
        for index in 0..width {
            if index != 0 {
                output.push_str(", ");
            }
            write!(output, "{scalar}(a[{index}]").expect("String formatting");
            if intrinsic.arguments() == 2 {
                write!(output, ", b[{index}]").expect("String formatting");
            }
            output.push(')');
        }
        output.push_str("); }\n");
    }
}
