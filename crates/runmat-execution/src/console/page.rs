//! Bounded transport fragments preserve original write ordinals and UTF-8 byte positions.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleCursor {
    pub sequence: u64,
    pub text_offset: u64,
}
impl Default for ConsoleCursor {
    fn default() -> Self {
        Self {
            sequence: 1,
            text_offset: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleReadBudget {
    pub max_fragments: u32,
    pub max_text_bytes: u64,
}
impl ConsoleReadBudget {
    pub fn validate(self) -> Result<(), ContractError> {
        if !(1..=256).contains(&self.max_fragments)
            || !(4..=1024 * 1024).contains(&self.max_text_bytes)
        {
            return Err(ContractError::invalid(
                "console read budget",
                "outside supported bounds",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleFragment {
    pub sequence: u64,
    pub stream: ConsoleStream,
    pub text_offset: u64,
    pub original_text_bytes: u64,
    pub text: Box<str>,
}
impl ConsoleFragment {
    pub fn from_record(
        record: &ConsoleRecord,
        cursor: ConsoleCursor,
        maximum_bytes: u64,
    ) -> Result<Self, ContractError> {
        let start = usize::try_from(cursor.text_offset)
            .ok()
            .filter(|start| record.text.is_char_boundary(*start));
        if cursor.sequence != record.sequence || start.is_none() {
            return Err(ContractError::invalid(
                "console cursor",
                "not an original UTF-8 position",
            ));
        }
        let start = start.expect("validated position");
        if start == record.text.len() && start != 0 {
            return Err(ContractError::invalid(
                "console cursor",
                "finished write must advance sequence",
            ));
        }
        let mut end = start
            .saturating_add(usize::try_from(maximum_bytes).unwrap_or(usize::MAX))
            .min(record.text.len());
        while !record.text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start && start < record.text.len() {
            return Err(ContractError::invalid(
                "console read budget",
                "cannot contain next UTF-8 character",
            ));
        }
        Ok(Self {
            sequence: record.sequence,
            stream: record.stream,
            text_offset: cursor.text_offset,
            original_text_bytes: record.text.len() as u64,
            text: record.text[start..end].into(),
        })
    }
    pub fn next_cursor(&self) -> Result<ConsoleCursor, ContractError> {
        let offset = self
            .text_offset
            .checked_add(self.text.len() as u64)
            .filter(|offset| *offset <= self.original_text_bytes)
            .ok_or_else(|| ContractError::invalid("console fragment", "invalid UTF-8 inventory"))?;
        if offset == self.original_text_bytes {
            Ok(ConsoleCursor {
                sequence: self.sequence.checked_add(1).ok_or_else(|| {
                    ContractError::invalid("console fragment", "sequence overflow")
                })?,
                text_offset: 0,
            })
        } else {
            Ok(ConsoleCursor {
                sequence: self.sequence,
                text_offset: offset,
            })
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsolePage {
    pub limits: ConsoleLimits,
    pub budget: ConsoleReadBudget,
    pub cursor: ConsoleCursor,
    pub next_cursor: ConsoleCursor,
    pub fragments: Vec<ConsoleFragment>,
    pub observed_records: u64,
    pub observed_text_bytes: u64,
    pub completion: ConsoleCompletion,
    pub source_receipt: Option<ConsoleReceipt>,
}
impl ConsolePage {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.limits.validate()?;
        self.budget.validate()?;
        if self.observed_records > u64::from(self.limits.max_records)
            || self.observed_text_bytes > self.limits.max_text_bytes
            || self.fragments.len() > self.budget.max_fragments as usize
        {
            return Err(ContractError::invalid(
                "console page",
                "observed source exceeds controls",
            ));
        }
        let mut cursor = self.cursor;
        let mut bytes = 0_u64;
        let mut previous: Option<&ConsoleFragment> = None;
        for f in &self.fragments {
            if f.sequence != cursor.sequence
                || f.text_offset != cursor.text_offset
                || f.original_text_bytes > self.observed_text_bytes
                || previous.is_some_and(|p| {
                    p.sequence == f.sequence
                        && (p.stream != f.stream || p.original_text_bytes != f.original_text_bytes)
                })
                || f.text_offset
                    .checked_add(f.text.len() as u64)
                    .is_none_or(|n| n > f.original_text_bytes)
                || (f.original_text_bytes > 0 && f.text.is_empty())
                || (f.stream == ConsoleStream::ClearScreen && f.original_text_bytes != 0)
                || f.sequence > self.observed_records
            {
                return Err(ContractError::invalid(
                    "console page",
                    "invalid original fragment",
                ));
            }
            bytes = bytes
                .checked_add(f.text.len() as u64)
                .filter(|bytes| *bytes <= self.budget.max_text_bytes)
                .ok_or_else(|| {
                    ContractError::invalid("console page", "text exceeds read budget")
                })?;
            cursor = f.next_cursor()?;
            previous = Some(f);
        }
        let receipt_valid = match &self.source_receipt {
            Some(r) => {
                r.limits == self.limits
                    && r.record_count == self.observed_records
                    && r.text_bytes == self.observed_text_bytes
                    && r.completion == self.completion
            }
            None => self.completion == ConsoleCompletion::Interrupted,
        };
        if !receipt_valid
            || self.cursor.sequence == 0
            || self.cursor.sequence > self.observed_records + 1
            || (self.cursor.sequence == self.observed_records + 1 && self.cursor.text_offset != 0)
            || self.next_cursor != cursor
        {
            return Err(ContractError::invalid(
                "console page",
                "invalid bounded page inventory",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
