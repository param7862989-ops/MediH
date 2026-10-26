# Missing information and open issues

Each item says what is missing or unverified, what must be provided, and where it goes. Nothing listed
here was guessed or filled in with invented content.

## 1. AI authorship and the NIL AI content declaration (resolve first)

- **The report text, the diagrams and the poster were drafted by an AI assistant (Claude).** They are
  based on the project's code, test results and data, and on verified references.
- A **NIL AI content** declaration, or the template's statement that the report "represents our ideas in
  our own words", would therefore not be accurate for this text as written.
- Before anyone signs the declaration or attaches a NIL AI image, the team must do one of the following:
  - rewrite the report in its own words; or
  - follow SPIT's policy on disclosing AI assistance.

  Ask your guide which applies.
- The development history also records AI assistance: 13 of the 25 non-merge commits carry an AI
  co-author trailer. Chapter 7 states this.

## 2. Genuine NIL AI content image (missing)

- **Status:** not found in `Project report guidelines.docx`, which has no embedded images; not in the
  template ZIP, whose only image is `spitlogo.pdf`; and not in the repository. Nothing was recreated or
  simulated.
- **Provide:** the genuine image, saved as `figures/nil_ai_declaration.png`.
- **Then:** rebuild (README.md, "Insert the NIL AI content image"). It replaces the placeholder on the last
  page (page 62).

## 3. Plagiarism report (missing)

- **Status:** not supplied. The template's "Plagiarism Report" section holds a placeholder. No score is
  stated or implied.
- **Provide:** the genuine institutional report.
- **Where:** `chapters/research_paper_plagiarism.tex`, for example via `\includegraphics` of the report
  pages.

## 4. Research paper (none)

- **Status:** the template has a "Research Paper" section. No paper exists for this project.
- **If one is written:** insert it, its plagiarism report and any acceptance receipt in
  `chapters/research_paper_plagiarism.tex`.

## 5. Identity and approval details

All of these go in `frontmatter/details.tex`. Do not fill in any of them without confirmation.

| Item | Status |
| :--- | :--- |
| Group number | Not found anywhere. |
| Approved project title | The repository README's title is used. Confirm it matches the approved title. |
| Student surnames | Only the first names Vidhi, Nirupam and Param appear in the repository (README). Surnames are not recorded. |
| Student UIDs | Not found. |
| Guide name and designation | Not found. Also check whether the template's co-guide line is needed. |
| Head of Department and Principal names | Not found. The template's names are from a 2014 Information Technology certificate, so they were not reused. |
| Submission month and year | Not found (certificate page). |

**Signatures:** the certificate, approval certificate and declaration have empty signature lines. No
signature or approval is implied.

**Folder name:** this folder is `GroupNo-TBD_MediHaven`. Rename it to the official
`GroupNo_TitleOfTheProject` form once the group number and title are confirmed.

## 6. Course designation conflict

| Source | Says |
| :--- | :--- |
| Guidelines DOCX (AY 2026–27) | "Mini Project Report Preparation Guidelines, T.E. Computer Engineering" |
| Repository README | "TE Computer Engineering, Semester V, Mini Project I" |
| Template ZIP | A **B.Tech Major Project** template for **Information Technology** |
| Your instructions | "major project" |

- **What I used:** the guidelines and README designation, "T.E. Mini Project (Semester V)", Computer
  Engineering.
- **Structure:** the guidelines' 11-chapter structure, with the template's cover, certificate,
  declaration and acknowledgement wording, as the guidelines instruct.
- **Formatting:** the template's 12 pt A4 layout with 20/20/20/50 mm margins.
- **To do:** confirm the course name. It is set in `\CourseName` in `details.tex`.

## 7. Reference file locations

The reference files were found at `docs/report-references/` (plural), not the expected
`docs/report-reference/`. The template was extracted to `docs/report-references/template/`. The original
files were not modified.

## 8. Individual contributions (Chapter 7.8)

- **Required:** the guidelines ask for each member's contribution.
- **What Git shows:** commits by `paramthakkar24-sudo` (13), `Nirupam` (10) and `param7862989-ops` (7,
  including 5 merges). No commits appear under a Vidhi identity.
- **Why that isn't enough:** Git identities are not verified mappings to students.
- **To do:** the group should state its contributions in Section 7.8.

## 9. Literature review

- **Count:** 18 papers reviewed, meeting the template's minimum of 12. The bibliography also includes
  the dataset, PhysioNet, scikit-learn and two RFCs.
- **Verification:** all were verified against DOI/Crossref, PubMed, PMLR, JMLR, DataCite or the RFC
  Editor, and findings are quoted only from the abstracts that were read.
- **Gaps:**
  - The ACM article number of Kaufman et al. (2012) came from a secondary record, so it is not printed.
  - The MedRec summary rests on a partial abstract.
  - No verified paper specifically on QR- or token-based personal health record sharing was found;
    MedRec is the closest.

## 10. Evidence limitations

- **Test count:** the run cited is 354 passed (26 Oct 2026). It is on the working tree of branch
  `param`, which includes the research triage view, implemented and tested but **not yet committed**.
  The previously recorded clean-state result of 301 passed was not re-run separately.
- **Load test:** done on one laptop over loopback. Individual results varied by up to a factor of two
  between runs; Chapter 8 says so.
- **Docker:** the image recipe was not built (Docker not installed).
- **Clinical evidence:** there is none. Research metrics come from a 107-stay demo cohort with 11 deaths
  and are labelled as pipeline validation only.

## 11. Compilation

- **Tested:** both PDFs compile with Tectonic 0.15.0.
- **Not tested:** pdflatex/latexmk with TeX Live or MiKTeX.

## 12. Page count (62 pages, target 30–40)

The report is above the target for three reasons:

- **Front matter:** 12 pages that the guidelines and template require. These are the cover,
  certificate, approval certificate, declaration, acknowledgement, abstract, contents (3 pages), list of
  figures, list of tables, and abbreviations.
- **Structure:** the guidelines mandate 11 chapters with about 80 named subsections, and the template
  asks for a page break after each chapter.
- **Layout:** the template's 12 pt text and 50 mm bottom margin.

The body (Chapters 1–11, pages 13–57) is 45 pages, slightly above the template's "25–40". It was tightened
without shrinking fonts, making diagrams illegible or dropping mandated sections. To shorten further, the
team could:

- condense the literature paragraphs into a table;
- move the appendix out of the report;
- ask the guide whether the 50 mm bottom margin is intended (25 mm would save about 6 pages).
