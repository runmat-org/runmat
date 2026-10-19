use super::*;

/// One source's bounded ordered inventory. Overflow permanently rejects the
/// capture: later writes cannot hide a missing write or restore completeness.
pub struct ConsoleCapture {
    limits: ConsoleLimits,
    records: Vec<ConsoleRecord>,
    text_bytes: u64,
    rejected: Option<ConsoleRejection>,
}

impl ConsoleCapture {
    pub fn new(limits: ConsoleLimits) -> Result<Self, ContractError> {
        limits.validate()?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(limits.max_records as usize)
            .map_err(|_| ContractError::invalid("console capture", "record allocation failed"))?;
        Ok(Self {
            limits,
            records,
            text_bytes: 0,
            rejected: None,
        })
    }

    pub fn limits(&self) -> ConsoleLimits {
        self.limits
    }
    pub fn text_bytes(&self) -> u64 {
        self.text_bytes
    }

    pub fn records(&self) -> &[ConsoleRecord] {
        &self.records
    }

    pub fn rejected(&self) -> Option<ConsoleRejection> {
        self.rejected
    }

    pub fn push(&mut self, stream: ConsoleStream, text: String) -> Result<(), ContractError> {
        if let Some(limit) = self.rejected {
            return Err(limit_error(limit, self.limits));
        }
        if stream == ConsoleStream::ClearScreen && !text.is_empty() {
            self.rejected = Some(ConsoleRejection::InvalidControlRecord);
            return Err(limit_error(
                ConsoleRejection::InvalidControlRecord,
                self.limits,
            ));
        }
        let limit = if self.records.len() >= self.limits.max_records as usize {
            Some(ConsoleRejection::RecordLimit)
        } else if self
            .text_bytes
            .checked_add(text.len() as u64)
            .is_none_or(|bytes| bytes > self.limits.max_text_bytes)
        {
            Some(ConsoleRejection::TextByteLimit)
        } else {
            None
        };
        if let Some(limit) = limit {
            self.rejected = Some(limit);
            return Err(limit_error(limit, self.limits));
        }
        self.text_bytes += text.len() as u64;
        self.records.push(ConsoleRecord {
            sequence: self.records.len() as u64 + 1,
            stream,
            text: text.into_boxed_str(),
        });
        Ok(())
    }

    /// Move the original inventory, retaining a rejected prefix even if the
    /// program otherwise returned successfully. No transcript clone is needed.
    pub fn finish(self, completed: bool) -> ConsoleTranscript {
        ConsoleTranscript {
            schema_version: CONSOLE_TRANSCRIPT_SCHEMA_VERSION,
            source_receipt: None,
            limits: self.limits,
            records: self.records,
            text_bytes: self.text_bytes,
            completion: match self.rejected {
                Some(reason) => ConsoleCompletion::Rejected { reason },
                None if completed => ConsoleCompletion::Complete,
                None => ConsoleCompletion::Interrupted,
            },
        }
    }
}

fn limit_error(limit: ConsoleRejection, limits: ConsoleLimits) -> ContractError {
    let (field, limit) = match limit {
        ConsoleRejection::InvalidControlRecord => {
            return ContractError::invalid("console record", "clear screen carries text")
        }
        ConsoleRejection::RecordLimit => ("console records", u64::from(limits.max_records)),
        ConsoleRejection::TextByteLimit => ("console UTF-8 bytes", limits.max_text_bytes),
    };
    ContractError::Limit { field, limit }
}
