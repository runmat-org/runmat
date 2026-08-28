use crate::ObjectNamespace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheNamespace(pub ObjectNamespace);

impl CacheNamespace {
    pub fn directory(self) -> &'static str {
        match self.0 {
            ObjectNamespace::ProgramSource => "program-source",
            ObjectNamespace::PackageRelease => "package-release",
            ObjectNamespace::ProgramArtifact => "program-artifact",
            ObjectNamespace::InputValue => "input-value",
            ObjectNamespace::ResultValue => "result-value",
            ObjectNamespace::DetailedEvent => "detailed-event",
            ObjectNamespace::Log => "log",
            ObjectNamespace::Checkpoint => "checkpoint",
        }
    }
}
