import datetime
import numbers

def analyze_column_types(rows, columns):
    """
    Simple heuristic-based column type detection from rows.
    Returns: dict {col_name: 'numeric' | 'categorical' | 'datetime' | 'text'}
    """
    types = {}
    if not rows:
        return {c: 'unknown' for c in columns}

    for col_index, col_name in enumerate(columns):
        sample_vals = [r[col_index] for r in rows if r[col_index] is not None]
        if not sample_vals:
            types[col_name] = 'unknown'
            continue

        # Detect datetimes
        if all(isinstance(v, (datetime.date, datetime.datetime)) for v in sample_vals[:10]):
            types[col_name] = 'datetime'
        # Detect numeric
        elif all(isinstance(v, numbers.Number) or str(v).replace('.', '', 1).isdigit() for v in sample_vals[:10]):
            types[col_name] = 'numeric'
        # Detect categorical (few unique values)
        elif len(set(sample_vals[:20])) < len(sample_vals[:20]) / 2:
            types[col_name] = 'categorical'
        else:
            types[col_name] = 'text'
    return types