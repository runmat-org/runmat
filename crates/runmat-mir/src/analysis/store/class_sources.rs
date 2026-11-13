//! Function IDs are projected into analysis coordinates; original class
//! source coordinates and declaration identities are not function IDs.
use super::AnalysisStore;
impl AnalysisStore {
    /// Preserve declaration/source identity across function-ordinal projection.
    pub fn validate_class_sources(&self, mir: &crate::MirAssembly) -> Result<(), String> {
        if self.classes.len() != mir.classes.len()
            || self
                .classes
                .iter()
                .zip(&mir.classes)
                .any(|(analysis, declaration)| {
                    analysis.declaration.id != declaration.id
                        || analysis.declaration.source != declaration.source
                        || analysis.declaration.name != declaration.name
                })
        {
            return Err(
                "analysis class declaration sources differ from their original MIR owners".into(),
            );
        }
        Ok(())
    }
}
