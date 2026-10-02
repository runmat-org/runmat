use super::{LocalFeaExecutionOptions, PreparedFeaStudy, PreparedFeaStudyRoot};

pub(super) fn validate_prepared_execution(
    prepared: &PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
) -> Result<(), String> {
    if prepared.workers == 0 || options.workers == Some(0) {
        return Err("prepared study execution requires nonzero worker metadata".into());
    }
    // Metadata describe the original root, never a replacement scientific authority.
    macro_rules! metadata {
        ($root:expr, $backend:expr) => {
            (
                $root.study == prepared.realized,
                $root.program_revision == prepared.revision,
                $backend,
                $root.analysis_accelerator_requests == options.accelerators,
            )
        };
    }
    let (study_matches, revision_matches, backend, accelerators_match) = match &prepared.root {
        PreparedFeaStudyRoot::Electrical(root) => {
            metadata!(root, runmat_analysis_fea::ComputeBackend::Cpu)
        }
        PreparedFeaStudyRoot::Magnetic(root) => {
            metadata!(root, runmat_analysis_fea::ComputeBackend::Cpu)
        }
        PreparedFeaStudyRoot::Maxwell(root) => {
            metadata!(root, runmat_analysis_fea::ComputeBackend::Cpu)
        }
        PreparedFeaStudyRoot::Fluid(root) => {
            metadata!(root, runmat_analysis_fea::ComputeBackend::Cpu)
        }
        PreparedFeaStudyRoot::Acoustic(root) => {
            metadata!(root, runmat_analysis_fea::ComputeBackend::Cpu)
        }
        PreparedFeaStudyRoot::Thermal(root) => metadata!(root, root.planning.backend),
        PreparedFeaStudyRoot::Structural(root) => {
            let expected = root.result_expectation();
            metadata!(expected, expected.planning.backend)
        }
    };
    if !study_matches || !revision_matches {
        return Err("prepared study metadata differs from original root study/revision".into());
    }
    if options.backend != backend || !accelerators_match {
        return Err("execution options differ from prepared backend/accelerator admission".into());
    }
    if !matches!(prepared.root, PreparedFeaStudyRoot::Structural(_))
        && (backend != runmat_analysis_fea::ComputeBackend::Cpu || !options.accelerators.is_empty())
    {
        return Err("prepared nonstructural study has no admitted accelerator envelope".into());
    }
    Ok(())
}
