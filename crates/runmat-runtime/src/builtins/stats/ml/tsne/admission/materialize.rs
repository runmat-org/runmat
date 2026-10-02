//! Download only typed controls that need host values for execution.

use super::super::{cooperation::TsneCooperation, gather, invalid, options};
use super::{reserve_admission, AdmittedOption, AdmittedStatsetField};
use crate::BuiltinResult;

pub(in crate::builtins::stats::ml::tsne) async fn materialize_options(
    admitted: Vec<AdmittedOption>,
    cooperation: &mut TsneCooperation,
) -> BuiltinResult<Vec<AdmittedOption>> {
    let mut materialized = Vec::new();
    reserve_admission(
        &mut materialized,
        admitted.len(),
        "tsne: materialized options exceed available memory",
    )?;
    for option in admitted {
        materialized.push(match option {
            AdmittedOption::CacheSizeNumeric(value) => {
                AdmittedOption::CacheSizeNumeric(gather(value).await?)
            }
            AdmittedOption::Exaggeration(value) => {
                AdmittedOption::Exaggeration(gather(value).await?)
            }
            AdmittedOption::NumDimensions(value) => {
                AdmittedOption::NumDimensions(gather(value).await?)
            }
            AdmittedOption::NumPcaComponents(value) => {
                AdmittedOption::NumPcaComponents(gather(value).await?)
            }
            AdmittedOption::Perplexity(value) => AdmittedOption::Perplexity(gather(value).await?),
            AdmittedOption::Standardize(value) => AdmittedOption::Standardize(gather(value).await?),
            AdmittedOption::InitialY(value) => AdmittedOption::InitialY(gather(value).await?),
            AdmittedOption::LearnRate(value) => AdmittedOption::LearnRate(gather(value).await?),
            AdmittedOption::NumPrint(value) => AdmittedOption::NumPrint(gather(value).await?),
            AdmittedOption::Theta(value) => AdmittedOption::Theta(gather(value).await?),
            AdmittedOption::Verbose(value) => AdmittedOption::Verbose(gather(value).await?),
            AdmittedOption::Options(fields) => {
                let mut values = Vec::new();
                reserve_admission(
                    &mut values,
                    fields.len(),
                    "tsne: materialized statset fields exceed available memory",
                )?;
                for field in fields {
                    values.push(match field {
                        AdmittedStatsetField::MaxIter(value) => {
                            AdmittedStatsetField::MaxIter(gather(value).await?)
                        }
                        AdmittedStatsetField::TolFun(value) => {
                            AdmittedStatsetField::TolFun(gather(value).await?)
                        }
                        AdmittedStatsetField::EmptyOutputFcnResident(value) => {
                            let value = gather(value).await?;
                            if !options::is_empty_numeric(&value) {
                                return Err(invalid("tsne: Options.OutputFcn must be empty"));
                            }
                            AdmittedStatsetField::EmptyOutputFcn
                        }
                        other => other,
                    });
                    cooperation.completed(1).await?;
                }
                AdmittedOption::Options(values)
            }
            AdmittedOption::OptionsEmptyResident(value) => {
                let value = gather(value).await?;
                if !options::is_empty_numeric(&value) {
                    return Err(invalid(
                        "tsne: Options must be an empty numeric array or struct",
                    ));
                }
                AdmittedOption::Options(Vec::new())
            }
            other => other,
        });
        cooperation.completed(1).await?;
    }
    Ok(materialized)
}
