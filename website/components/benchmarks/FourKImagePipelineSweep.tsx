type SweepRow = {
  batchSize: number;
  runMatMs: number;
  vsPyTorch: string;
  vsNumPy: string;
};

const SWEEP_ROWS: SweepRow[] = [
  { batchSize: 4, runMatMs: 204, vsPyTorch: "4.5× faster", vsNumPy: "2.6× faster" },
  { batchSize: 8, runMatMs: 265, vsPyTorch: "3.3× faster", vsNumPy: "3.5× faster" },
  { batchSize: 16, runMatMs: 299, vsPyTorch: "3.3× faster", vsNumPy: "6.1× faster" },
  { batchSize: 32, runMatMs: 493, vsPyTorch: "2.2× faster", vsNumPy: "7.6× faster" },
  { batchSize: 64, runMatMs: 871, vsPyTorch: "1.5× faster", vsNumPy: "8.3× faster" },
];

export default function FourKImagePipelineSweep() {
  return (
    <div className="mx-auto max-w-[40rem] rounded-2xl border border-border/60 bg-background/60 overflow-hidden shadow-lg">
      {/* Header */}
      <div className="bg-muted/70 px-4 sm:px-6 py-4 border-b border-border/60">
        <div className="text-xs sm:text-sm font-semibold uppercase tracking-wide text-foreground">
          4K Image Pipeline
        </div>
        <div className="mt-1 text-[11px] sm:text-xs text-muted-foreground">
          Batch size sweep: 4 → 64 images
        </div>
      </div>

      {/* Table */}
      <div className="bg-background">
        <div className="overflow-x-auto">
          <table className="w-full text-xs sm:text-sm md:text-base">
            <thead className="border-b border-border/60 bg-muted/40 text-center">
              <tr>
                <th className="px-4 sm:px-6 py-3 text-center text-muted-foreground font-medium">
                  Benchmark
                </th>
                <th className="px-4 sm:px-6 py-3 text-center font-medium text-muted-foreground">
                  RunMat (ms)
                </th>
                <th className="px-4 sm:px-6 py-3 text-center font-medium">
                  <span className="inline-flex items-center justify-center gap-2 text-[11px] sm:text-xs text-purple-100">
                    <span className="h-2 w-2 rounded-full bg-purple-400" />
                    <span>vs PyTorch</span>
                  </span>
                </th>
                <th className="px-4 sm:px-6 py-3 text-center font-medium">
                  <span className="inline-flex items-center justify-center gap-2 text-[11px] sm:text-xs text-blue-100">
                    <span className="h-2 w-2 rounded-full bg-blue-400" />
                    <span>vs NumPy</span>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody className="bg-background">
              {SWEEP_ROWS.map((row) => (
                <tr
                  key={row.batchSize}
                  className="border-b border-border/40 last:border-b-0"
                >
                  <td className="px-4 sm:px-6 py-3 text-center">
                    <span className="font-mono text-sm sm:text-base">
                      {row.batchSize}
                    </span>
                  </td>
                  <td className="px-4 sm:px-6 py-3 text-center font-mono text-sm sm:text-base">
                    {row.runMatMs}
                  </td>
                  <td className="px-4 sm:px-6 py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 px-3 py-1 text-[11px] sm:text-xs text-purple-100">
                      {row.vsPyTorch}
                    </span>
                  </td>
                  <td className="px-4 sm:px-6 py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10 px-3 py-1 text-[11px] sm:text-xs text-blue-100">
                      {row.vsNumPy}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 text-white text-[11px] sm:text-xs">
                <td className="px-4 sm:px-6 py-3 font-medium" colSpan={2}>
                  RunMat speedup range
                </td>
                <td className="px-4 sm:px-6 py-3 text-center">
                  1.5× – 4.5× vs PyTorch
                </td>
                <td className="px-4 sm:px-6 py-3 text-center">
                  2.6× – 8.3× vs NumPy
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
}


