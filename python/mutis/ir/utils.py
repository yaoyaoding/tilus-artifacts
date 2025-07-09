from typing import List, Any


def table_prune_empty_column(headers: List[str], rows: List[List[Any]]):
    columns_to_skip = []

    for column in range(len(headers)):
        if all(isinstance(row[column], str) and row[column].strip() == '' for row in rows):
            columns_to_skip.append(column)

    headers = [header for idx, header in enumerate(headers) if idx not in columns_to_skip]
    rows = [[cell for idx, cell in enumerate(row) if idx not in columns_to_skip] for row in rows]
    return headers, rows
