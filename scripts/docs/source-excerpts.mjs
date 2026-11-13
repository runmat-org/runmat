/** Presentation excerpts are exact slices of the packaged source, never a second study. */
export function sourceExcerpts(source, sections = []) {
  return sections.map(({ title, start, end }) => {
    const first = source.indexOf(start);
    const last = end === null ? source.length : source.indexOf(end, first + start.length);
    if (first < 0 || source.indexOf(start, first + start.length) !== -1 || last <= first) {
      throw new Error(`Example section '${title}' no longer matches its canonical source`);
    }
    return { title, content: source.slice(first, last).trim() };
  });
}
