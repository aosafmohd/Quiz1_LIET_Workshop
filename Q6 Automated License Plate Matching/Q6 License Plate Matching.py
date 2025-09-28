import pytest
import random
import string
import re

# -------------------------------
# License Plate Validator Function
# -------------------------------
def is_valid_license_plate(plate: str) -> bool:
    """
    Validates modern Indian license plates.
    Format: [State][District][Series][Number]
    Example: MH12AB1234, DL05C6789
    """
    pattern = r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$"
    return bool(re.match(pattern, plate))


# -------------------------------
# Helper Functions to Generate Test Data
# -------------------------------
def generate_valid_plate():
    state = ''.join(random.choices(string.ascii_uppercase, k=2))
    district = ''.join(random.choices(string.digits, k=2))
    series = ''.join(random.choices(string.ascii_uppercase, k=random.choice([1, 2])))
    number = ''.join(random.choices(string.digits, k=4))
    return state + district + series + number

def generate_invalid_plate():
    # Randomly break the format
    choices = [
        ''.join(random.choices(string.ascii_lowercase, k=2)) + "12AB1234",  # lowercase state code
        ''.join(random.choices(string.ascii_uppercase, k=3)) + "12345",     # wrong length
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=9)), # random 9 chars
        "MH" + ''.join(random.choices(string.digits, k=5)),                  # missing series
    ]
    return random.choice(choices)


# -------------------------------
# Generate Test Cases
# -------------------------------
valid_plates = [generate_valid_plate() for _ in range(500)]
invalid_plates = [generate_invalid_plate() for _ in range(500)]


# -------------------------------
# Pytest Test Cases
# -------------------------------
@pytest.mark.parametrize("plate", valid_plates)
def test_valid_plates(plate):
    assert is_valid_license_plate(plate) == True

@pytest.mark.parametrize("plate", invalid_plates)
def test_invalid_plates(plate):
    assert is_valid_license_plate(plate) == False


# -------------------------------
# Run pytest directly if script is run
# -------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])
