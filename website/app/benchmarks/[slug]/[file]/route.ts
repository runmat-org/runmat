import { NextRequest, NextResponse } from 'next/server';
import { join } from 'path';
import { existsSync, readFileSync } from 'fs';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string; file: string }> }
) {
  const { slug, file } = await params;

  // Security check: ensure no directory traversal
  if (file.includes('..') || slug.includes('..')) {
    return new NextResponse('Invalid path', { status: 400 });
  }

  const benchmarksDir = join(process.cwd(), '..', 'benchmarks');
  const filePath = join(benchmarksDir, slug, file);

  if (!existsSync(filePath)) {
    return new NextResponse('File not found', { status: 404 });
  }

  const content = readFileSync(filePath);
  // Default to text/plain so browsers display the code instead of downloading
  const contentType = 'text/plain; charset=utf-8';

  return new NextResponse(content, {
    headers: {
      'Content-Type': contentType,
    },
  });
}

