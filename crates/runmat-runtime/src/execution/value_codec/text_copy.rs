/// Incremental UTF-8 copy for ordinary-policy codec text. Managed destinations
/// use the Value-domain exact copy and retain its original allocation claim.
pub(super) struct TextCopy<'a> {
    source: &'a str,
    source_offset: usize,
    output: String,
}
impl<'a> TextCopy<'a> {
    pub(super) fn new(source: &'a str) -> Self {
        Self {
            source,
            source_offset: 0,
            output: String::with_capacity(source.len()),
        }
    }
    pub(super) fn advance(&mut self, maximum_work: usize) -> usize {
        let mut completed = 0;
        for character in self.source[self.source_offset..].chars().take(maximum_work) {
            self.output.push(character);
            self.source_offset += character.len_utf8();
            completed += 1;
        }
        completed.max(1)
    }
    pub(super) fn is_complete(&self) -> bool {
        self.source_offset == self.source.len()
    }
    pub(super) fn finish(self) -> String {
        self.output
    }
}
