/**
 * Regenerates a RunMat MCAD part or assembly into its exact STEP AP242 study geometry.
 *
 * `rootPath` names the design and is a key of `files`, which holds the text of the design and of
 * every file it reads (parts, sub-assemblies, imported STEP files) keyed by slash-separated
 * paths. Resolve to the bytes the OCCT kernel's `study_geometry(rootPath, files, null)` writes;
 * reject with the reason the design does not regenerate, which the FEA check reports at the
 * geometry's `file =`.
 */
export type StudyGeometryProvider = (
  rootPath: string,
  files: Record<string, string>
) => Promise<Uint8Array>;

/** Native generated export behind {@link installStudyGeometryProvider}. */
export interface StudyGeometryNativeExports {
  setStudyGeometryProvider?: (provider: StudyGeometryProvider | null) => void;
}

/**
 * Installs the browser's study-geometry provider for the runtime worker that loaded `native`, or
 * removes it with `null`. Without a provider, a study on a part or assembly reports at once that
 * this browser session can't regenerate RunMat MCAD geometry.
 */
export function installStudyGeometryProvider(
  native: StudyGeometryNativeExports,
  provider: StudyGeometryProvider | null
): void {
  if (typeof native.setStudyGeometryProvider !== "function") {
    throw new Error(
      "The loaded RunMat module does not support setStudyGeometryProvider; rebuild the browser runtime assets."
    );
  }
  native.setStudyGeometryProvider(provider);
}
