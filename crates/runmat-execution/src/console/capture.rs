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
        self.precheck(stream, &text)?;
        self.push_prechecked(stream, text);
        Ok(())
    }

    /// Reject a borrowed write before allocating its retained text. A host
    /// must not clone an oversized or already-rejected record first.
    pub fn push_borrowed(
        &mut self,
        stream: ConsoleStream,
        text: &str,
    ) -> Result<(), ContractError> {
        self.precheck(stream, text)?;
        let mut retained = String::new();
        retained
            .try_reserve_exact(text.len())
            .map_err(|_| ContractError::invalid("console capture", "text allocation failed"))?;
        retained.push_str(text);
        self.push_prechecked(stream, retained);
        Ok(())
    }

    fn precheck(&mut self, stream: ConsoleStream, text: &str) -> Result<(), ContractError> {
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
        Ok(())
    }

    fn push_prechecked(&mut self, stream: ConsoleStream, text: String) {
        self.text_bytes += text.len() as u64;
        self.records.push(ConsoleRecord {
            sequence: self.records.len() as u64 + 1,
            stream,
            text: text.into_boxed_str(),
        });
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

#[cfg(test)]
mod borrowed_tests {
    use super::*;

    #[test]
    fn oversized_borrowed_text_rejects_before_retention_and_remains_sticky() {
        let mut capture = ConsoleCapture::new(ConsoleLimits {
            max_records: 1,
            max_text_bytes: 2,
        })
        .unwrap();
        let source = "borrowed text that cannot fit";
        assert!(capture
            .push_borrowed(ConsoleStream::Stdout, source)
            .is_err());
        assert_eq!(capture.rejected(), Some(ConsoleRejection::TextByteLimit));
        assert!(capture.records().is_empty());
        assert_eq!(capture.text_bytes(), 0);
        assert!(capture.push_borrowed(ConsoleStream::Stdout, "ok").is_err());
        assert!(capture.records().is_empty());
    }
}
