use super::NativeLibraryMetadata;

pub fn normalize_metadata(mut metadata: NativeLibraryMetadata) -> NativeLibraryMetadata {
    for library in &mut metadata.libraries {
        library.dependencies.sort();
        library.dependencies.dedup();
        library.symbols.sort_by(|left, right| {
            (&left.name, &left.exported_name).cmp(&(&right.name, &right.exported_name))
        });
    }
    metadata
        .libraries
        .sort_by(|left, right| left.name.cmp(&right.name));
    metadata
        .structures
        .sort_by(|left, right| left.name.cmp(&right.name));
    metadata
        .enumerations
        .sort_by(|left, right| left.name.cmp(&right.name));
    metadata
        .aliases
        .sort_by(|left, right| left.name.cmp(&right.name));
    metadata
}
