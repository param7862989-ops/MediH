# MediHaven: project report and A3 poster

LaTeX sources and compiled PDFs for the MediHaven project report, prepared to the SPIT report template
and the *Mini Project Report Preparation Guidelines (T.E. Computer Engineering, AY 2026–27)*.

**Before submitting, read `MISSING_INFORMATION.md`.** Identity fields, signatures, the NIL AI content
image and the plagiarism report are still placeholders. There is also an AI-authorship issue you need to
resolve before signing anything.

## Deliverables

| File | Description |
| :--- | :--- |
| `Final_Report.pdf` | The report: 62 pages, A4, built 26 Oct 2026. Page 62 (the last page) is the NIL AI content page, currently a placeholder. |
| `A3_Poster.pdf` | The poster: 1 page, A3 landscape (420 × 297 mm). |
| `main.tex` | Report master file. |
| `frontmatter/` | Cover, certificate, approval certificate, declaration, acknowledgement, abstract, abbreviations, NIL page. `details.tex` holds every identity field. |
| `chapters/` | Chapters 1–11 and the template's research-paper and plagiarism-report sections. |
| `diagrams/` | TikZ sources for all 11 report diagrams, the compact poster diagram, and shared styles. |
| `figures/` | SPIT logo (from the official template) and genuine application screenshots. |
| `appendices/` | Reproducibility appendix, the load-test script `perf_measure.py`, and its raw results `perf_results.json`. |
| `poster/poster.tex` | Poster source. |
| `bibliography.bib` | 23 references, verified against DOI/Crossref, PubMed, PMLR, JMLR, DataCite or the RFC Editor. Style: IEEEtran. |

## Build

The PDFs were built with **Tectonic 0.15.0**, a self-contained LaTeX engine. It downloads the LaTeX
packages it needs on first use and runs BibTeX automatically.

```powershell
# from this folder
tectonic main.tex                      # -> main.pdf
Copy-Item main.pdf Final_Report.pdf

cd poster
tectonic poster.tex                    # -> poster.pdf
Copy-Item poster.pdf ..\A3_Poster.pdf
```

To get Tectonic, download the release for your system from
https://github.com/tectonic-typesetting/tectonic/releases (for example
`tectonic-0.15.0-x86_64-pc-windows-msvc.zip`) and put `tectonic.exe` on your PATH. The first build needs
internet access to fetch packages.

The sources use only standard packages (geometry, tikz/pgfplots, tcolorbox, tocloft, titlesec,
adjustbox, booktabs, longtable, algorithmicx, hyperref, cite) and IEEEtran.bst, so
`latexmk -pdf main.tex` with TeX Live or MiKTeX should also work. That route was **not** tested.

## Fill in the identity details

Edit `frontmatter/details.tex` and replace every `\missing{...}` value: surnames, UIDs, group number,
guide, Head of Department, Principal and submission month. Both PDFs read from this one file, so rebuild
both afterwards. Each `\missing{...}` prints as a red bracketed marker (yellow on the poster header), so
an unfilled field cannot go unnoticed.

Then rename this folder from `GroupNo-TBD_MediHaven` to the official `GroupNo_TitleOfTheProject` form,
once the group number and approved title are confirmed.

## Insert the NIL AI content image

1. Save the **genuine** image issued for this report as `figures/nil_ai_declaration.png`. If you have a
   `.jpg` or `.pdf` instead, change the file name in `\NILImageFile` in `frontmatter/nil_ai_declaration.tex`.
2. Rebuild with `tectonic main.tex` and copy to `Final_Report.pdf`.
3. Check that the image now appears on the last page and that nothing follows it.

Read the AI-authorship note in `MISSING_INFORMATION.md` (item 1) first.

## Verification performed

- **Report build:** compiled without errors. BibTeX reported no warnings, and there are no undefined
  references or citations. The report is 62 A4 pages.
- **Poster build:** compiled to exactly one A3 landscape page.
- **Rendering:** every page was rendered and inspected. In particular: cover, certificate, declaration,
  abstract, contents, diagram and table pages, results and screenshot pages, bibliography and the final
  page. Overflowing tables and colliding diagrams were fixed after inspection.
- **Content scan:** an automated scan of both PDFs and all sources found no `.env` secret values, no
  API-key-like strings, no MIMIC-IV identifiers and no leftover template text.
- **Evidence behind the numbers:**
  - Test count: an actual run of the full suite (354 passed).
  - Load-test figures: a measurement documented in Chapter 8 (`appendices/perf_results.json`).
  - Research metrics: the verified Phase 2 run v2.0.0.
  - Cohort counts: the Phase 1 dataset manifest.

## Known limitations

- **Page count:** 62 pages, above the 30–40 target. The reasons are in `MISSING_INFORMATION.md`, item 12.
- **Docker image:** the image recipe described in the report was not built, because Docker is not
  installed on the development machine.
