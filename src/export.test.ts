import { describe, expect, it } from 'vitest';
import { csvCell } from './export';
describe('review exports', () => {
  it('preserves multiline notes and escapes quotes', () => {
    expect(csvCell('He said "check"\nagain')).toBe('"He said ""check""\nagain"');
  });
  it('keeps spreadsheet formulas in reviewer notes inert', () => {
    expect(csvCell('=HYPERLINK("https://example.com")')).toBe('"\'=HYPERLINK(""https://example.com"")"');
    expect(csvCell(' @SUM(1,2)')).toBe('"\' @SUM(1,2)"');
    expect(csvCell(null)).toBe('""');
  });
});
