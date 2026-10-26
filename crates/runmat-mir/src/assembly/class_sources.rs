//! Class provenance is one original declaration fact. Publication and cold
//! admission share this allocation-free structural consistency check.
use super::MirAssembly;
use runmat_types::ProgramSourceId;

impl MirAssembly {
    /// Validate original class source membership and each declared method's owner.
    pub fn validate_class_sources(
        &self,
        contains: impl Fn(ProgramSourceId) -> bool,
    ) -> Result<(), String> {
        if self
            .classes
            .windows(2)
            .any(|classes| classes[0].id >= classes[1].id)
        {
            return Err("MIR class declaration identities are duplicated or unordered".into());
        }
        for class in &self.classes {
            if !contains(class.source) {
                return Err("class declaration source is outside its original source map".into());
            }
            for method in &class.methods {
                let metadata = self
                    .functions
                    .get(&method.function)
                    .ok_or("class method has no original MIR metadata")?;
                let owner = metadata
                    .class_method_owner
                    .as_ref()
                    .ok_or("class method has no original class owner")?;
                if metadata.source != class.source
                    || metadata.enclosing_class != Some(class.id)
                    || !owner.declaring_class.matches_qualified_name(&class.name)
                    || owner.method != method.name
                    || owner.is_static != method.is_static
                    || owner.role != method.role
                {
                    return Err(
                        "class method disagrees with its original declaration source or owner"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
}
