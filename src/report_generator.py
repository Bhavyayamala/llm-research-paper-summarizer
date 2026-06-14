def generate_report(summary, contribution, limitations):

    report = f"""
RESEARCH PAPER REPORT

SUMMARY:
{summary}

MAIN CONTRIBUTION:
{contribution}

LIMITATIONS:
{limitations}
"""

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Report generated!")