# Public release checklist

Complete these items before making the GitHub repository public or minting a DOI.

1. **Corpus redistribution rights** — confirm that University of Isfahan source documents may be redistributed as extracted JSONL text. Public availability of a PDF does not automatically imply permission to republish extracted content. Note that `indices/*/*/all_meta.json` also contains chunk text, so removing only `data/corpus/` is not sufficient if text redistribution is restricted.
2. **Privacy review** — run `python scripts/privacy_scan.py` and manually inspect all email-like strings, personal names, phone numbers, IDs, internal hostnames, and support examples. The current corpus contains email-like strings that require human review before release.
3. **Secrets** — make sure no API token, password, cookie, SSH key, Hugging Face token, or private server path is committed.
4. **Model weights** — do not commit third-party model weights. Reference exact model IDs/revisions and download them under their original licenses.
5. **License** — choose an institution-approved code license and a separate data/annotation license or redistribution statement. Replace the placeholder `LICENSE` file.
6. **Repository identifiers** — replace `OWNER`, repository URL, and DOI placeholders in `README.md` and `CITATION.cff`.
7. **Exact environment** — after reproducing the final runs from a clean environment, commit `requirements-lock.txt`, Python version, OS, and CPU information.
8. **Release tag** — create an immutable tag such as `v1.0.0-paper`, then archive that release in Zenodo (or an equivalent repository) and cite the DOI rather than a moving branch.
9. **Checksums** — regenerate `SHA256SUMS.txt` for the archived release and keep the final Dev/Test hashes frozen.
10. **Blind review** — if the journal requires anonymous review, confirm that a public repository, commit history, DOI metadata, or author names do not reveal identity before the review stage permits it.
