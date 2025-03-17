import re


def extract_field_info(text_list):
    """Extract relevant information from detected text"""
    id_number = None
    name = None
    dob = None
    address = None
    nationality = None
    expiry_date = None
    print(text_list)
    # Pattern for Vietnamese ID card number
    id_pattern = r'\b\d{9,12}\b'

    # Pattern for date (DD/MM/YYYY)
    date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'

    for text in text_list:
        text_lower = text.lower()

        # Find ID number
        id_match = re.search(id_pattern, text)
        if id_match and not id_number:
            id_number = id_match.group(0)

        # Check for name
        if "họ và tên" in text_lower or "họ tên" in text_lower:
            # Get the next line which likely contains the name
            idx = text_list.index(text)
            if idx + 1 < len(text_list):
                name = text_list[idx + 1]

        # Check for date of birth
        if "ngày sinh" in text_lower or "sinh ngày" in text_lower:
            dob_match = re.search(date_pattern, text)
            if dob_match:
                dob = dob_match.group(0)
            else:
                # Get the next line which likely contains the DOB
                idx = text_list.index(text)
                if idx + 1 < len(text_list):
                    next_text = text_list[idx + 1]
                    dob_match = re.search(date_pattern, next_text)
                    if dob_match:
                        dob = dob_match.group(0)

        # Check for address
        if "nơi cư trú" in text_lower or "địa chỉ" in text_lower:
            # The address might span multiple lines
            idx = text_list.index(text)
            if idx + 1 < len(text_list):
                address = text_list[idx + 1]

        # Check for nationality
        if "quốc tịch" in text_lower:
            # Get the next word or line
            idx = text_list.index(text)
            if idx + 1 < len(text_list):
                nationality = text_list[idx + 1]

        # Check for expiry date
        if "giá trị đến" in text_lower or "có giá trị đến" in text_lower:
            expiry_match = re.search(date_pattern, text)
            if expiry_match:
                expiry_date = expiry_match.group(0)

    return {
        "ID Number": id_number,
        "Name": name,
        "Date of Birth": dob,
        "Address": address,
        "Nationality": nationality,
        "Expiry Date": expiry_date
    }
