//! Structured exception presentation retains actual source ordinals and
//! nested causes. No opaque live cause or executable authority is introduced.
use super::*;
pub(super) async fn exception(
    reader: &mut Reader,
    budget: &mut Budget,
    depth: u16,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<ExceptionValue> {
    budget
        .usage
        .depth(depth)
        .map_err(|source| with_source("selected exception depth", source))?;
    reader.token(b'{')?;
    reader.key("identifier", true)?;
    let identifier = budget.text(reader, c).await?;
    reader.key("message", false)?;
    let message = budget.text(reader, c).await?;
    reader.key("stack", false)?;
    let stack = super::leaves::texts(reader, budget, c).await?;
    reader.key("causes", false)?;
    reader.token(b'[')?;
    let mut causes = Vec::new();
    while !reader.is(b']') {
        budget.elements(1)?;
        budget.nodes(1)?;
        c.completed(1).await?;
        let nested = depth.checked_add(1).ok_or_else(|| {
            with_source(
                "selected exception depth overflow",
                runmat_execution::ContractError::Limit {
                    field: "value depth",
                    limit: budget.limits.max_depth.into(),
                },
            )
        })?;
        budget
            .usage
            .depth(nested)
            .map_err(|source| with_source("selected exception depth", source))?;
        if !causes.is_empty() {
            reader.token(b',')?;
        }
        causes
            .try_reserve(1)
            .map_err(|source| with_source("selected exception cause allocation failed", source))?;
        causes.push(Box::pin(exception(reader, budget, nested, c)).await?);
    }
    reader.token(b']')?;
    let runtime = if reader.is(b'}') {
        None
    } else {
        reader.key("runtime", false)?;
        if reader.null()? {
            None
        } else {
            Some(runtime(reader, budget, c).await?)
        }
    };
    reader.token(b'}')?;
    Ok(ExceptionValue {
        identifier,
        message,
        stack,
        causes,
        runtime,
    })
}
async fn runtime(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<ExceptionRuntimeMetadata> {
    reader.token(b'{')?;
    reader.key("span", true)?;
    let span = reader.metadata(64, c).await?;
    reader.key("builtin", false)?;
    let builtin = budget.optional_text(reader, c).await?;
    reader.key("task_id", false)?;
    let task_id = budget.optional_text(reader, c).await?;
    reader.key("call_frames", false)?;
    reader.token(b'[')?;
    let mut call_frames = Vec::new();
    while !reader.is(b']') {
        budget.elements(1)?;
        c.completed(1).await?;
        if !call_frames.is_empty() {
            reader.token(b',')?;
        }
        call_frames
            .try_reserve(1)
            .map_err(|source| with_source("selected exception frame allocation failed", source))?;
        call_frames.push(frame(reader, budget, c).await?);
    }
    reader.token(b']')?;
    reader.key("call_frames_elided", false)?;
    let call_frames_elided = reader.metadata(20, c).await?;
    reader.key("call_stack", false)?;
    let call_stack = super::leaves::texts(reader, budget, c).await?;
    reader.key("phase", false)?;
    let phase = budget.optional_text(reader, c).await?;
    reader.key("gpu_gather_retry", false)?;
    let gpu_gather_retry = reader.metadata(32, c).await?;
    reader.key("failure_origin", false)?;
    let failure_origin = reader.metadata(64, c).await?;
    reader.token(b'}')?;
    Ok(ExceptionRuntimeMetadata {
        span,
        builtin,
        task_id,
        call_frames,
        call_frames_elided,
        call_stack,
        phase,
        gpu_gather_retry,
        failure_origin,
    })
}
async fn frame(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<ExceptionCallFrame> {
    reader.token(b'{')?;
    reader.key("function", true)?;
    let function = budget.text(reader, c).await?;
    reader.key("source_id", false)?;
    let source_id = reader.metadata(20, c).await?;
    reader.key("span", false)?;
    let span = reader.metadata(64, c).await?;
    let detached = if reader.is(b'}') {
        None
    } else {
        reader.key("detached_presentation", false)?;
        if reader.null()? {
            None
        } else {
            reader.token(b'{')?;
            reader.key("display_name", true)?;
            let display_name = budget.text(reader, c).await?;
            reader.key("line", false)?;
            let line: Option<u64> = reader.metadata(20, c).await?;
            reader.key("column", false)?;
            let column: Option<u64> = reader.metadata(20, c).await?;
            reader.token(b'}')?;
            if line.is_some() != column.is_some() {
                return Err(error(
                    "detached exception coordinates must be present together",
                ));
            }
            Some(DetachedDiagnosticFrame {
                display_name,
                line,
                column,
            })
        }
    };
    reader.token(b'}')?;
    let source = match (source_id, detached) {
        (Some(id), None) => ExceptionFrameSource::Admitted(id),
        (None, Some(presentation)) => ExceptionFrameSource::Detached(presentation),
        (None, None) => ExceptionFrameSource::Unknown,
        (Some(_), Some(_)) => {
            return Err(error(
                "exception frame cannot claim admitted and detached sources",
            ))
        }
    };
    Ok(ExceptionCallFrame {
        function,
        source,
        span,
    })
}
