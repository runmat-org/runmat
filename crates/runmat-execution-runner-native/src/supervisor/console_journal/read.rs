//! Bounded pages borrow one original attempt, not its complete execution history.
use super::*;

pub(in crate::supervisor) struct ConsoleReadPlan {
    root: PathBuf,
    original: Header,
}
impl ConsoleReadPlan {
    /// Bind the original assignment while borrowing scheduler authority; retain no
    /// arguments, program, outputs or execution history across filesystem reads.
    pub(in crate::supervisor) fn bind(
        job: &Path,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<Self> {
        Ok(Self {
            root: directory(job, request),
            original: header(request)?,
        })
    }
}

fn load(plan: &ConsoleReadPlan) -> NativeExecutionResult<Option<(PathBuf, Head)>> {
    let root = &plan.root;
    let Some(original): Option<Header> =
        read_optional_json_bounded(&root.join("original.json"), CONSOLE_HEAD_BYTES)?
    else {
        return Ok(None);
    };
    if original != plan.original {
        return Err(invalid("console journal differs from original assignment"));
    }
    let head: Head = read_optional_json_bounded(&root.join("head.json"), CONSOLE_HEAD_BYTES)?
        .ok_or_else(|| invalid("console journal publication head is absent"))?;
    if head.inventory.limits != original.limits
        || head.inventory.record_count > u64::from(original.limits.max_records)
        || head.inventory.text_bytes > original.limits.max_text_bytes
        || head.last_record.is_some() != (head.inventory.record_count > 0)
        || match &head.source_receipt {
            Some(r) => r != &head.inventory,
            None => head.inventory.completion != ConsoleCompletion::Interrupted,
        }
    {
        return Err(invalid("invalid original console prefix head"));
    }
    Ok(Some((root.clone(), head)))
}

fn record(
    root: &Path,
    sequence: u64,
    limits: ConsoleLimits,
) -> NativeExecutionResult<StoredRecord> {
    let maximum = limits.max_text_bytes * 6 + 512;
    let record: StoredRecord =
        read_optional_json_bounded(&record_path(root, sequence), maximum)?
            .ok_or_else(|| invalid("published original console write is absent"))?;
    if record.record.sequence != sequence
        || record.record.text.len() as u64 > limits.max_text_bytes
        || (record.record.stream == ConsoleStream::ClearScreen && !record.record.text.is_empty())
    {
        return Err(invalid("invalid original console write"));
    }
    Ok(record)
}

pub(in crate::supervisor) fn read_transcript(
    job: &Path,
    request: &AttemptRequest,
) -> NativeExecutionResult<Option<ConsoleTranscript>> {
    if request.task.console.is_none() {
        return Ok(None);
    }
    let plan = ConsoleReadPlan::bind(job, request)?;
    let Some((root, head)) = load(&plan)? else {
        return Ok(None);
    };
    let mut receiver = ConsoleReceiver::new(head.inventory.limits).map_err(invalid)?;
    let mut inventory = ConsoleInventory::new(head.inventory.limits).map_err(invalid)?;
    let mut previous = None;
    for sequence in 1..=head.inventory.record_count {
        let item = record(&root, sequence, head.inventory.limits)?;
        if item.previous != previous {
            return Err(invalid("console write chain differs from original order"));
        }
        previous = Some(json_digest(&item)?);
        inventory.push(&item.record).map_err(invalid)?;
        receiver.push(item.record).map_err(invalid)?;
    }
    if previous != head.last_record
        || inventory.receipt(head.inventory.completion) != head.inventory
    {
        return Err(invalid(
            "console head differs from retained original writes",
        ));
    }
    receiver
        .finish(head.source_receipt.as_ref())
        .map(Some)
        .map_err(invalid)
}

impl ConsoleReadPlan {
    pub(in crate::supervisor) fn read_page(
        &self,
        cursor: ConsoleCursor,
        budget: ConsoleReadBudget,
    ) -> NativeExecutionResult<Option<ConsolePage>> {
        budget.validate().map_err(invalid)?;
        let Some((root, head)) = load(self)? else {
            return Ok(None);
        };
        if cursor.sequence == 0
            || cursor.sequence > head.inventory.record_count + 1
            || (cursor.sequence == head.inventory.record_count + 1 && cursor.text_offset != 0)
        {
            return Err(invalid(
                "console cursor is outside the observed original prefix",
            ));
        }
        let mut page = ConsolePage {
            limits: head.inventory.limits,
            budget,
            cursor,
            next_cursor: cursor,
            fragments: vec![],
            observed_records: head.inventory.record_count,
            observed_text_bytes: head.inventory.text_bytes,
            completion: head.inventory.completion,
            source_receipt: head.source_receipt.clone(),
        };
        let mut remaining = budget.max_text_bytes;
        while page.next_cursor.sequence <= page.observed_records
            && page.fragments.len() < budget.max_fragments as usize
            && remaining >= 4
        {
            let item = record(&root, page.next_cursor.sequence, page.limits)?;
            // Check adjacent immutable links and the observed tail; a page is partial
            // consistency evidence, not an independent whole-inventory proof.
            let previous = if item.record.sequence == 1 {
                None
            } else {
                Some(json_digest(&record(
                    &root,
                    item.record.sequence - 1,
                    page.limits,
                )?)?)
            };
            if item.previous != previous
                || (item.record.sequence == page.observed_records
                    && Some(json_digest(&item)?) != head.last_record)
            {
                return Err(invalid("console page differs from original write chain"));
            }
            let fragment = ConsoleFragment::from_record(&item.record, page.next_cursor, remaining)
                .map_err(invalid)?;
            remaining -= fragment.text.len() as u64;
            page.next_cursor = fragment.next_cursor().map_err(invalid)?;
            page.fragments.push(fragment);
        }
        page.validate().map_err(invalid)?;
        Ok(Some(page))
    }
}
