use super::intrinsic::{Intrinsic, Shape};
use super::{include_family, FamilySet};
use crate::wgsl::host_buffers::{self, ClaimedString, ClaimedVec};
use runmat_execution::resource::ProviderHostMetadataAdmission;
use std::{fmt, ops::Range};

#[derive(Debug)]
pub(crate) struct ParseFailure {
    error: naga::front::wgsl::ParseError,
    // Naga-produced diagnostic rendering is library-owned, not a RunMat copy.
    diagnostic: String,
}
impl fmt::Display for ParseFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.diagnostic)
    }
}
impl std::error::Error for ParseFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[derive(Debug)]
pub(crate) struct ValidationFailure {
    // Release the exact boxed payload before its retained allocation funding.
    error: Box<naga::WithSpan<naga::valid::ValidationError>>,
    diagnostic: String,
    _allocation: host_buffers::Claim,
}
impl fmt::Display for ValidationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.diagnostic)
    }
}
impl std::error::Error for ValidationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

#[derive(Debug)]
pub(crate) enum PreparationError {
    Parse(ParseFailure),
    Validate(ValidationFailure),
    InvalidSpan { intrinsic: Intrinsic },
    ConflictingEdits,
    RemainingIntrinsic { intrinsic: Intrinsic },
    ReservedHelperCollision(ClaimedString),
    Buffer(host_buffers::Error),
    SourceSizeOverflow,
    ConstantIntrinsic { intrinsic: Intrinsic },
}
impl From<host_buffers::Error> for PreparationError {
    fn from(error: host_buffers::Error) -> Self {
        Self::Buffer(error)
    }
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
            Self::Buffer(error) => fmt::Display::fmt(error, f),
            Self::SourceSizeOverflow => f.write_str("shader preparation source size overflows"),
            Self::ConstantIntrinsic { intrinsic } => write!(
                f,
                "unsupported unevaluated binary64 constant intrinsic {intrinsic:?}"
            ),
        }
    }
}
impl std::error::Error for PreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Validate(error) => Some(error),
            Self::Buffer(error) => Some(error),
            _ => None,
        }
    }
}

pub(crate) struct PreparedShader {
    source: ClaimedString,
    requires_f64_math: bool,
}
impl PreparedShader {
    #[cfg(any(test, all(feature = "wgpu", not(target_arch = "wasm32"))))]
    pub(crate) fn source(&self) -> &str {
        self.source.as_str()
    }
    pub(crate) fn into_source(self) -> ClaimedString {
        self.source
    }
    pub(crate) fn requires_f64_math(&self) -> bool {
        self.requires_f64_math
    }
}

#[derive(PartialEq, Eq)]
struct Edit {
    span: Range<usize>,
    intrinsic: Intrinsic,
    shape: Shape,
}

fn parse(
    source: &str,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<(naga::Module, naga::valid::ModuleInfo), PreparationError> {
    // Naga's parser/validator storage and its diagnostic output remain the
    // explicitly excluded library boundary; preserve original typed causes.
    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        let diagnostic = error.emit_to_string(source);
        PreparationError::Parse(ParseFailure { error, diagnostic })
    })?;
    // Reserve the known failure-wrapper layout before validation can produce
    // that payload. A refusal happens before validation, so it cannot replace
    // an already-produced typed Naga cause. On success the unused reservation
    // is released; on failure it follows the box until physical deallocation.
    // Naga's own parser/validator and rendered diagnostic allocations remain
    // the separately declared library boundary, not part of this claim.
    let error_allocation = host_buffers::claim_bytes(
        metadata,
        std::mem::size_of::<naga::WithSpan<naga::valid::ValidationError>>(),
    )?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| {
        let diagnostic = error.emit_to_string(source);
        PreparationError::Validate(ValidationFailure {
            error: Box::new(error),
            diagnostic,
            _allocation: error_allocation,
        })
    })?;
    Ok((module, info))
}

fn collect_function(
    source: &str,
    module: &naga::Module,
    function: &naga::Function,
    info: &naga::valid::FunctionInfo,
    edits: &mut ClaimedVec<Edit>,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
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
        edits.push(
            Edit {
                span,
                intrinsic,
                shape,
            },
            metadata,
        )?;
    }
    Ok(())
}

fn collect(
    source: &str,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<ClaimedVec<Edit>, PreparationError> {
    for (_, expression) in module.global_expressions.iter() {
        if let naga::Expression::Math { fun, arg, .. } = *expression {
            if let Some(intrinsic) = Intrinsic::from_math(fun) {
                if Shape::of(info[arg].inner_with(&module.types)).is_some() {
                    return Err(PreparationError::ConstantIntrinsic { intrinsic });
                }
            }
        }
    }
    let mut edits = ClaimedVec::with_capacity(0, metadata)?;
    for (handle, function) in module.functions.iter() {
        collect_function(
            source,
            module,
            function,
            &info[handle],
            &mut edits,
            metadata,
        )?;
    }
    for (index, entry) in module.entry_points.iter().enumerate() {
        collect_function(
            source,
            module,
            &entry.function,
            info.get_entry_point(index),
            &mut edits,
            metadata,
        )?;
    }
    Ok(edits)
}

pub(crate) fn prepare(
    source: &str,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<PreparedShader, PreparationError> {
    let (module, info) = parse(source, metadata)?;
    let mut edits = collect(source, &module, &info, metadata)?;
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
            source: ClaimedString::copy_from(source, metadata)?,
            requires_f64_math,
        });
    }
    // Sorted/deduplicated flat records replace private BTree node allocation.
    let mut selected = ClaimedVec::with_capacity(edits.len(), metadata)?;
    for edit in edits.as_slice() {
        selected.push((edit.intrinsic, edit.shape), metadata)?;
    }
    selected.as_mut_slice().sort_unstable();
    selected.dedup();
    let mut helpers = ClaimedString::with_capacity(0, metadata)?;
    append_helpers(&mut helpers, selected.as_slice(), metadata)?;
    let helper_module = naga::front::wgsl::parse_str(helpers.as_str()).map_err(|error| {
        let diagnostic = error.emit_to_string(helpers.as_str());
        PreparationError::Parse(ParseFailure { error, diagnostic })
    })?;
    let mut omit = ClaimedVec::with_capacity(0, metadata)?;
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
                .zip(
                    helper_span
                        .clone()
                        .and_then(|span| helpers.as_str().get(span)),
                )
                .is_some_and(|(a, b)| a == b);
            if !identical {
                return Err(PreparationError::ReservedHelperCollision(
                    ClaimedString::copy_from(name, metadata)?,
                ));
            }
            if let Some(mut span) = helper_span {
                // Extend only the canonical helper declaration span, exactly
                // as before; never discover operations by scanning user text.
                while helpers
                    .as_str()
                    .as_bytes()
                    .get(span.end)
                    .is_some_and(u8::is_ascii_whitespace)
                {
                    span.end += 1;
                }
                if helpers.as_str().as_bytes().get(span.end) != Some(&b'}') {
                    return Err(PreparationError::ReservedHelperCollision(
                        ClaimedString::copy_from(name, metadata)?,
                    ));
                }
                span.end += 1;
                omit.push(span, metadata)?;
            }
        }
    }
    omit.as_mut_slice().sort_unstable_by_key(|span| span.start);
    for span in omit.as_slice().iter().rev() {
        helpers.remove_range(span.clone())?;
    }
    edits
        .as_mut_slice()
        .sort_unstable_by_key(|edit| (edit.span.start, edit.span.end));
    edits.dedup();
    let replacement_bytes = edits.as_slice().iter().try_fold(0usize, |bytes, edit| {
        let count =
            host_buffers::formatted_len(format_args!("{}", edit.intrinsic.helper(edit.shape)))?;
        count
            .checked_sub(edit.span.len())
            .and_then(|added| bytes.checked_add(added))
            .ok_or(host_buffers::Error::Overflow)
    })?;
    let source_bytes = source
        .len()
        .checked_add(helpers.len())
        .and_then(|bytes| bytes.checked_add(replacement_bytes))
        .and_then(|bytes| bytes.checked_add(1))
        .ok_or(PreparationError::SourceSizeOverflow)?;
    let mut output = ClaimedString::with_capacity(source_bytes, metadata)?;
    let mut cursor = 0;
    for edit in edits.as_slice() {
        if edit.span.start < cursor {
            return Err(PreparationError::ConflictingEdits);
        }
        output.push_str(&source[cursor..edit.span.start], metadata)?;
        output.append_fmt(
            format_args!("{}", edit.intrinsic.helper(edit.shape)),
            metadata,
        )?;
        cursor = edit.span.end;
    }
    output.push_str(&source[cursor..], metadata)?;
    output.push_str("\n", metadata)?;
    output.push_str(helpers.as_str(), metadata)?;
    if output.len() != source_bytes {
        return Err(PreparationError::SourceSizeOverflow);
    }
    let (module, info) = parse(output.as_str(), metadata)?;
    if let Some(edit) = collect(output.as_str(), &module, &info, metadata)?
        .as_slice()
        .first()
    {
        return Err(PreparationError::RemainingIntrinsic {
            intrinsic: edit.intrinsic,
        });
    }
    Ok(PreparedShader {
        source: output,
        requires_f64_math: true,
    })
}

fn append_helpers(
    output: &mut ClaimedString,
    selected: &[(Intrinsic, Shape)],
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<(), PreparationError> {
    let mut families = FamilySet::default();
    for (intrinsic, _) in selected {
        include_family(*intrinsic, &mut families);
    }
    for family in families.iter() {
        output.push_str(family.source(), metadata)?;
        output.push_str("\n", metadata)?;
    }
    for &(intrinsic, shape) in selected {
        let Some(width) = shape.vector_width() else {
            continue;
        };
        let name = intrinsic.helper(shape);
        let scalar = intrinsic.helper(Shape::Scalar);
        output.append_fmt(format_args!("fn {name}(a: vec{width}<f64>"), metadata)?;
        if intrinsic.arguments() == 2 {
            output.append_fmt(format_args!(", b: vec{width}<f64>"), metadata)?;
        }
        output.append_fmt(
            format_args!(") -> vec{width}<f64> {{ return vec{width}<f64>("),
            metadata,
        )?;
        for index in 0..width {
            if index != 0 {
                output.push_str(", ", metadata)?;
            }
            output.append_fmt(format_args!("{scalar}(a[{index}]"), metadata)?;
            if intrinsic.arguments() == 2 {
                output.append_fmt(format_args!(", b[{index}]"), metadata)?;
            }
            output.push_str(")", metadata)?;
        }
        output.push_str("); }\n", metadata)?;
    }
    Ok(())
}
