import datetime
import decimal
import uuid
import numpy as np

def universal_serialize(value):
    """Safely convert any SQL/JSON value to a JSON-serializable form."""
    try:
        if value is None:
            return None
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.isoformat()
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, (bytes, bytearray)):
            return value.decode(errors="ignore")
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, (np.int64, np.int32, np.float32, np.float64)):
            return value.item()
        return value
    except Exception:
        return str(value)

def serialize_value(value):
    """Convert complex SQLAlchemy values into JSON-serializable Python values."""
    import datetime, decimal

    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    elif isinstance(value, decimal.Decimal):
        return float(value)
    elif isinstance(value, bytes):
        return value.decode(errors="ignore")
    elif value is None:
        return None
    else:
        return value