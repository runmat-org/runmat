type ElementwiseRow = {
  points: string;
  runMatMs: number;
  vsPyTorch: string;
  vsNumPy: string;
};

const ELEMENTWISE_ROWS: ElementwiseRow[] = [
  { points: "1M", runMatMs: 197, vsPyTorch: "4.2× faster", vsNumPy: "0.35× as fast" },
  { points: "5M", runMatMs: 208, vsPyTorch: "5.3× faster", vsNumPy: "0.54× as fast" },
  { points: "10M", runMatMs: 174, vsPyTorch: "8.2× faster", vsNumPy: "0.96× as fast" },
  { points: "100M", runMatMs: 171, vsPyTorch: "98.8× faster", vsNumPy: "6.4× faster" },
  { points: "1B", runMatMs: 199, vsPyTorch: "113.6× faster", vsNumPy: "63.0× faster" },
];

export default function ElementwiseMathSweep() {
  return (
    <div className="mx-auto w-full max-w-[40rem] rounded-2xl border border-border/60 bg-background/60 overflow-hidden shadow-lg">
      <div className="bg-muted/70 px-4 sm:px-6 py-4 border-b border-border/60">
        <div className="text-sm sm:text-base font-semibold uppercase tracking-wide text-foreground">
          Elementwise Math
        </div>
        <div className="mt-1 text-xs sm:text-sm text-muted-foreground">
          Problem size sweep: 1M → 1B points
        </div>
      </div>
      <div className="bg-background">
        <div className="overflow-x-auto">
          <table className="w-full text-sm sm:text-base">
            <thead className="border-b border-border/60 bg-muted/40 text-center">
              <tr>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center text-muted-foreground font-medium">
                  Points (elements)
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-muted-foreground">
                  RunMat (ms)
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-muted-foreground">
                  <span className="inline-flex items-center justify-center gap-2 text-sm sm:text-base text-purple-100">
                    <span className="h-2 w-2 rounded-full bg-purple-400" />
                    <span>vs PyTorch</span>
                  </span>
                </th>
                <th className="px-3 sm:px-6 py-2 sm:py-3 text-center font-medium text-muted-foreground">
                  <span className="inline-flex items-center justify-center gap-2 text-sm sm:text-base text-blue-100">
                    <span className="h-2 w-2 rounded-full bg-blue-400" />
                    <span>vs NumPy</span>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody className="bg-background">
              {ELEMENTWISE_ROWS.map((row) => (
                <tr key={row.points} className="border-b border-border/40 last:border-b-0">
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="font-mono text-sm sm:text-base">{row.points}</span>
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center font-mono text-sm sm:text-base">
                    {row.runMatMs}
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 px-3 py-1 text-sm sm:text-base text-purple-100">
                      {row.vsPyTorch}
                    </span>
                  </td>
                  <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                    <span className="inline-flex items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10 px-3 py-1 text-sm sm:text-base text-blue-100">
                      {row.vsNumPy}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 text-white text-sm sm:text-base">
                <td className="px-3 sm:px-6 py-2 sm:py-3 font-medium" colSpan={2}>
                  RunMat speedup range
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  4.2× – 113.6× vs PyTorch
                </td>
                <td className="px-3 sm:px-6 py-2 sm:py-3 text-center">
                  0.35× – 63.0× vs NumPy
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
}

