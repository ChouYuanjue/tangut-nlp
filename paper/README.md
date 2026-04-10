# Paper Draft

This directory contains a compile-friendly LaTeX draft derived from the current Markdown manuscript.

## Scope

- The draft intentionally uses a plain `article` layout first.
- It avoids venue-specific style files until the related-work and citation pass is complete.
- Tables are already separated into reusable `.tex` files.

## Compile

```bash
cd paper
lualatex main.tex
lualatex main.tex
```

The draft now expects a Unicode engine for inline Chinese examples:

- `lualatex`: supported on the current machine
- `xelatex`: also works on systems where XeTeX is installed

If `xelatex` is installed on your machine, you can compile with:

```bash
cd paper
xelatex main.tex
xelatex main.tex
```

## Pending before submission

1. Replace the plain article style with the final venue template.
2. Add verified citations and a real `references.bib`.
3. Decide whether to include reference-aware judge results.
4. Decide whether to include a multitask-base DPO follow-up.
