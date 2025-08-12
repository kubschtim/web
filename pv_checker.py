import csv
import time
import os
from openai import OpenAI

# Use the existing client pattern
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CSV_PATH = "/Users/timkubsch/Downloads/vibecoding/web/lars_leads_top_rows.csv"

def _norm(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s == "None":
        return ""
    return s

def build_company_info(row: dict, fieldnames: list[str]) -> str:
    lines = []
    for key in fieldnames:
        val = _norm(row.get(key, ""))
        lines.append(f"{key}: {val}")
    return "\n".join(lines)

def analyze_row(row: dict, fieldnames: list[str]) -> str:
    company_info = build_company_info(row, fieldnames)

    website = _norm(row.get("Website", ""))
    page_url = _norm(row.get("Page_URL", ""))

    browse_hint = []
    if website:
        browse_hint.append(f"Primäre Quelle (Website): {website}")
    if page_url:
        browse_hint.append(f"Sekundäre Quelle (Google Maps): {page_url}")
    browse_hint_text = "\n".join(browse_hint) if browse_hint else "Keine URLs verfügbar."

    prompt_text = (
        "Lies und berücksichtige alle bereitgestellten Felder über das Unternehmen.\n\n"
        f"{company_info}\n\n"
        "Wenn verfügbar, nutze die URLs unterhalb zur Verifikation (Website bevorzugt, sonst Google Maps):\n"
        f"{browse_hint_text}\n\n"
        "Aufgabe:\n"
        "- Antworte ausschließlich mit:\n"
        '  - "Ja" (wenn das Unternehmen den gesamten Projektprozess selbst verantwortet, einschließlich Beratung, Planung, Installation und After-Sales — auch wenn die Montage durch Partner erfolgt)\n'
        '  - "Nein" (wenn es lediglich vermittelt oder berät, ohne eigene Projektverantwortung)\n'
        "- Füge eine prägnante Begründung (ein Satz) und einen Confidence-Score in Prozent hinzu."
    )

    resp = client.responses.create(
        model="gpt-5",
        tools=[{"type": "web_search_preview"}],
        input=[{"role": "user", "content": prompt_text}],
    )
    return resp.output_text

def main():
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        count = 0
        for row in reader:
            count += 1
            title = _norm(row.get("Title", f"Eintrag {count}"))
            print(f"Analysiere: {title}")
            try:
                out = analyze_row(row, fieldnames)
                print(f"Ergebnis: {out}")
            except Exception as e:
                print(f"Fehler bei '{title}': {e}")
            print("-" * 80)
            time.sleep(1.5)  # leichte Pause, um Limits zu vermeiden

if __name__ == "__main__":
    main()