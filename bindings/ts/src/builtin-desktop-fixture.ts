import type { BuiltinExampleFixtureId, BuiltinFilesystemFixture } from "./builtins.js";

/** Serialized projection of catalog/documentation/example/fixture/desktop.rs.
 * Paths are fixture-relative; the host runner validates and resolves them. */
export type BuiltinDesktopHostFixture = {
  id: BuiltinExampleFixtureId;
  entries: BuiltinFilesystemFixture["entries"];
  interactions: BuiltinDesktopHostInteraction[];
};

type DialogRequest = { title: string | null; default_path: string | null };
type FileDialogRequest = DialogRequest & {
  filters: Array<{ patterns: string[]; description: string | null }>;
};
type DialogOutcome<T> = "Cancel" | { Error: string } | { Selection: T };

export type BuiltinDocumentExpectation =
  | { OpenFile: { relative_path: string } }
  | { OpenVariable: { name: string } }
  | { Navigate: {
      relative_path: string;
      /** Positive one-based positions, validated by the fixture schema. */
      line: number;
      column: number;
      selection: "Caret" | "Select";
    } };

export type BuiltinDesktopHostInteraction =
  | { Document: BuiltinDocumentExpectation }
  | { OpenFileDialog: {
      request: FileDialogRequest & { multiselect: boolean };
      outcome: DialogOutcome<{ paths: string[]; filter_index: number | null }>;
    } }
  | { SaveFileDialog: {
      request: FileDialogRequest;
      outcome: DialogOutcome<{ path: string; filter_index: number | null }>;
    } }
  | { DirectoryDialog: {
      request: DialogRequest;
      outcome: DialogOutcome<{ path: string }>;
    } }
  | { LineInput: { prompt: string; echo: boolean; outcome: { Line: string } | { Error: string } } }
  | { KeyInput: { prompt: string; outcome: "KeyPress" | { Error: string } } }
  | { FigurePresentation: {
      figure_ordinal: number;
      event: "Created" | "Updated" | "Cleared" | "Closed";
      snapshot: { title: string | null; axes_rows: number; axes_cols: number } | null;
    } };
