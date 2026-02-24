import csv

import yaml

with open("/Users/andreasmartinson/repos/tuva/models/readmissions/readmissions_models.yml", "r") as f:
    data = yaml.safe_load(f)

rows = []
for table in data["models"]:
    table_name = table.get("name", "")
    columns = table.get("columns", [])

    # Find primary key (column with unique test)
    primary_keys = []
    for col in columns:
        tests = col.get("tests", [])
        for test in tests:
            if test == "unique" or (isinstance(test, dict) and "unique" in test):
                primary_keys.append(col["name"])

    primary_keys_str = ", ".join(primary_keys)

    for col in columns:
        rows.append(
            {
                "table": table_name,
                "primary_keys": primary_keys_str,
                "column_name": col.get("name", ""),
                "column_description": col.get("description", "").strip(),
            }
        )

with open("output.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["table", "primary_keys", "column_name", "column_description"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done! {len(rows)} rows written to output.csv")
