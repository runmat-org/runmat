'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';

type GlobalErrorProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  return (
    <html>
      <body className="min-h-screen bg-background text-foreground flex items-center justify-center px-6">
        <div className="max-w-lg text-center space-y-6">
          <div>
            <p className="text-sm uppercase tracking-wide text-muted-foreground">
              Something went wrong
            </p>
            <h1 className="mt-2 text-3xl font-semibold">We hit a snag</h1>
          </div>

          <p className="text-muted-foreground">
            {error.message || 'An unexpected error occurred. Please try again.'}
          </p>

          {error.digest && (
            <p className="text-xs text-muted-foreground">Error ID: {error.digest}</p>
          )}

          <div className="flex flex-wrap gap-3 justify-center">
            <Button onClick={reset}>Try again</Button>
            <Button variant="outline" asChild>
              <Link href="/">Go home</Link>
            </Button>
          </div>
        </div>
      </body>
    </html>
  );
}

