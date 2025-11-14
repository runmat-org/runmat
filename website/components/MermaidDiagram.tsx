"use client";

import { useEffect, useRef, useState } from 'react';
import { useTheme } from 'next-themes';

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

export function MermaidDiagram({ chart, className }: MermaidDiagramProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Ensure we're on the client side
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const renderChart = async () => {
      // Early return if ref is not available
      if (!ref.current) {
        console.warn('MermaidDiagram: ref.current is not available');
        return;
      }

      // Check if chart content is valid
      if (!chart || chart.trim().length === 0) {
        console.warn('MermaidDiagram: chart content is empty');
        setError('Mermaid chart content is empty');
        setIsLoading(false);
        return;
      }

      // Dynamically import mermaid only on client side
      let mermaid;
      try {
        mermaid = (await import('mermaid')).default;
      } catch (err) {
        console.error('Failed to load mermaid:', err);
        setError('Failed to load Mermaid library');
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        // Initialize mermaid with theme. Keep securityLevel default (strict) for safety.
        mermaid.initialize({
          startOnLoad: true,
          theme: theme === 'dark' ? 'dark' : 'default',
          flowchart: {
            htmlLabels: false,
            useMaxWidth: true,
            curve: 'basis',
          },
          themeVariables: {
            primaryColor: theme === 'dark' ? '#111827' : '#ffffff',
            primaryTextColor: theme === 'dark' ? '#E6E7E9' : '#202124',
            primaryBorderColor: theme === 'dark' ? '#334155' : '#CBD5E1',
            lineColor: theme === 'dark' ? '#64748B' : '#94A3B8',
            nodeBorder: theme === 'dark' ? '#334155' : '#CBD5E1',
            clusterBkg: theme === 'dark' ? '#0b1220' : '#F8FAFC',
            clusterBorder: theme === 'dark' ? '#334155' : '#CBD5E1',
            edgeLabelBackground: theme === 'dark' ? '#0f172a' : '#ffffff',
            tertiaryColor: theme === 'dark' ? '#0b1220' : '#ffffff',
            fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
            noteBkgColor: theme === 'dark' ? '#0b1220' : '#F8FAFC',
            noteBorderColor: theme === 'dark' ? '#334155' : '#CBD5E1',
          },
        });

        // Double-check ref is still available before rendering
        if (ref.current) {
          const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
          const { svg } = await mermaid.render(id, chart);
          
          // Triple-check ref is still available before setting innerHTML
          if (ref.current) {
            ref.current.innerHTML = svg;
            setIsLoading(false);
          }
        }
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        setError(errorMsg);
        setIsLoading(false);
        
        // Fallback: show the raw chart text if rendering fails
        if (ref.current) {
          ref.current.innerHTML = `<pre style="color: red; padding: 1rem; border: 1px solid red; border-radius: 4px;">Mermaid diagram failed to render: ${errorMsg}</pre>`;
        }
      }
    };

    // Add a small delay to ensure DOM is ready
    const timeoutId = setTimeout(renderChart, 100);
    
    return () => clearTimeout(timeoutId);
  }, [chart, theme, mounted]);

  // Don't render anything during SSR
  if (!mounted) {
    return (
      <div className={`w-full ${className ?? ''}`}>
        <div className="flex items-center justify-center p-8">
          <div className="animate-pulse text-gray-500 dark:text-gray-400">Loading diagram...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`${className} p-4 border border-red-300 rounded bg-red-50 dark:bg-red-900/20 dark:border-red-700`}>
        <p className="text-red-600 dark:text-red-400">Failed to render diagram: {error}</p>
      </div>
    );
  }

  return (
    <div className={`w-full ${className ?? ''}`}>
      {isLoading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-pulse text-gray-500 dark:text-gray-400">Loading diagram...</div>
        </div>
      )}
      <div ref={ref} className={`${isLoading ? 'hidden' : ''} w-full mermaid`} />
    </div>
  );
}