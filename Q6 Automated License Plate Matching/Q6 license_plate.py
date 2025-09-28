# license_plate.py

import re

def is_valid_license_plate(plate: str) -> bool:
    pattern = r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$"
    return bool(re.match(pattern, plate))
