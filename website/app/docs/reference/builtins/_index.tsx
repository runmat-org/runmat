import { Metadata } from 'next';
import BuiltinsExplorer from '@/components/BuiltinsExplorer';
import { loadBuiltins } from '@/lib/builtins';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'Built-in Functions | Docs',
  description: 'Reference for built-in functions in RunMat.'
};

export default function BuiltinsIndexPage() {
  const all = loadBuiltins();

  return (
    <div className="min-h-screen">
      <h1 className="text-3xl font-bold mb-2">Built-in Functions</h1>
      <p className="text-muted-foreground mb-4">RunMat implements a slim standard library of built-in functions with canonical behavior.</p>
      <div className="mb-6">
        <div className="h-2 bg-muted rounded overflow-hidden">
          <div className="h-2 bg-green-600 rounded" style={{ width: '100%' }} />
        </div>
        <div className="text-sm text-muted-foreground mt-1">{all.filter(b => !b.internal).length} built-in functions</div>
      </div>

      <BuiltinsExplorer builtins={all} />
    </div>
  );
}


