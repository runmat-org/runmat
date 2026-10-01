//! Download only typed controls that need host values for execution.

use super::super::{gather, invalid, options};
use super::{AdmittedOption, AdmittedStatsetField};
use crate::BuiltinResult;

pub(in crate::builtins::stats::ml::tsne) async fn materialize_options(
    admitted: Vec<AdmittedOption>,
) -> BuiltinResult<Vec<AdmittedOption>> {
    let mut materialized = Vec::with_capacity(admitted.len());
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
                let mut values = Vec::with_capacity(fields.len());
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
    }
    Ok(materialized)
}
