"""Render reports/*.md to PDF: markdown -> HTML -> headless Chrome. Needs Google Chrome installed."""
import subprocess, sys
from pathlib import Path
import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;max-width:100%;color:#222}
h1{font-size:20pt}h2{font-size:14pt;border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:22px;page-break-after:avoid}
table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:8px 0;page-break-inside:avoid}th,td{border:1px solid #bbb;padding:3px 5px;text-align:left;vertical-align:top}
th{background:#eef2f7}img{max-width:100%}hr{border:0;border-top:1px solid #ddd}@page{size:A4;margin:16mm}"""


def render(md_path: Path) -> Path:
    html = markdown.markdown(md_path.read_text(), extensions=["tables"])
    h = md_path.with_suffix(".html")
    h.write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>")
    pdf = md_path.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf.resolve()}", h.resolve().as_uri()],
                   check=True, capture_output=True)
    h.unlink()
    return pdf


def executive_summary() -> Path:
    """Section 0 of the validation report as its own one-page PDF."""
    text = Path("reports/validation_report.md").read_text()
    sec = text.split("## 0. Executive summary")[1].split("\n---\n")[0]
    md = Path("reports/executive_summary.md")
    md.write_text("# Executive summary: mortgage PD model validation" + sec)
    pdf = render(md); md.unlink()
    return pdf


if __name__ == "__main__":
    for f in sys.argv[1:] or ["reports/model_development.md", "reports/validation_report.md"]:
        print(render(Path(f)))
    if not sys.argv[1:]:
        print(executive_summary())
